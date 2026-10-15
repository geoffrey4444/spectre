// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Spherepack.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <InterpTargetTestHelpers::ValidPoints ValidPoints>
domain::creators::Sphere make_sphere() {
  if constexpr (ValidPoints == InterpTargetTestHelpers::ValidPoints::All) {
    return {0.9, 4.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false};
  }
  if constexpr (ValidPoints == InterpTargetTestHelpers::ValidPoints::None) {
    return {4.9, 8.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false};
  }
  return {3.4, 4.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false};
}

struct SphereTag : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
  using temporal_id = ::Tags::TimeStepId;
  using vars_to_interpolate_to_target = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using compute_items_on_target = tmpl::list<>;
  using compute_target_points =
      ::intrp::TargetPoints::Sphere<SphereTag, ::Frame::Inertial>;
  using post_interpolation_callbacks = tmpl::list<>;
};

template <InterpTargetTestHelpers::ValidPoints ValidPoints, typename Generator>
void test_interpolation_target_sphere(
    const gsl::not_null<Generator*> generator, const size_t number_of_spheres,
    const ylm::AngularOrdering angular_ordering) {
  // Keep bounds a bit inside than inner and outer radius of shell below so the
  // offset-sphere is still within the domain
  std::uniform_real_distribution<double> dist{1.2, 4.5};
  std::vector<double> radii(number_of_spheres);
  for (size_t i = 0; i < number_of_spheres; i++) {
    double radius = dist(*generator);
    while (alg::find(radii, radius) != radii.end()) {
      radius = dist(*generator);
    }
    radii[i] = radius;
  }
  const size_t l_max = 18;
  const std::array<double, 3> center = {{0.05, 0.06, 0.07}};

  CAPTURE(l_max);
  CAPTURE(center);
  CAPTURE(radii);
  CAPTURE(number_of_spheres);
  CAPTURE(angular_ordering);

  // Options for Sphere
  std::string radii_str;
  intrp::OptionHolders::Sphere sphere_opts;
  std::stringstream ss;
  ss << std::setprecision(std::numeric_limits<double>::max_digits10);
  if (number_of_spheres == 1) {
    // Test the double variant
    sphere_opts =
        intrp::OptionHolders::Sphere(l_max, center, radii[0], angular_ordering);
    ss << radii[0];
  } else {
    // Test the vector variant
    sphere_opts =
        intrp::OptionHolders::Sphere(l_max, center, radii, angular_ordering);
    ss << "[" << radii[0];
    for (size_t i = 1; i < number_of_spheres; i++) {
      ss << "," << radii[i];
    }
    ss << "]";
  }
  radii_str = ss.str();

  // Test creation of options
  const auto created_opts =
      TestHelpers::test_creation<intrp::OptionHolders::Sphere>(
          "Center: [0.05, 0.06, 0.07]\n"
          "Radius: " +
          radii_str +
          "\n"
          "LMax: 18\n"
          "AngularOrdering: " +
          std::string(MakeString{} << angular_ordering));
  CHECK(created_opts == sphere_opts);

  const auto domain_creator = make_sphere<ValidPoints>();

  TestHelpers::db::test_simple_tag<intrp::Tags::Sphere<SphereTag>>("Sphere");

  const auto expected_block_coord_holders = [&domain_creator, &radii, &center,
                                             &angular_ordering,
                                             &number_of_spheres]() {
    // How many points are supposed to be in a Strahlkorper,
    // reproduced here by hand for the test.
    const size_t n_theta = l_max + 1;
    const size_t n_phi = 2 * l_max + 1;

    // Have to turn this into a set to guarantee ordering
    const std::set<double> radii_set(radii.begin(), radii.end());

    tnsr::I<DataVector, 3, Frame::Inertial> points(number_of_spheres * n_theta *
                                                   n_phi);

    size_t s = 0;
    for (const double radius : radii_set) {
      // The theta points of a Strahlkorper are Gauss-Legendre points.
      const std::vector<double> theta_points = []() {
        std::vector<double> thetas(n_theta);
        std::vector<double> work(n_theta + 1);
        std::vector<double> unused_weights(n_theta);
        int err = 0;
        gaqd_(static_cast<int>(n_theta), thetas.data(), unused_weights.data(),
              work.data(), static_cast<int>(n_theta + 1), &err);
        return thetas;
      }();

      const double two_pi_over_n_phi = 2.0 * M_PI / n_phi;
      if (angular_ordering == ylm::AngularOrdering::Strahlkorper) {
        for (size_t i_phi = 0; i_phi < n_phi; ++i_phi) {
          const double phi = two_pi_over_n_phi * i_phi;
          for (size_t i_theta = 0; i_theta < n_theta; ++i_theta) {
            const double theta = theta_points[i_theta];
            points.get(0)[s] = radius * sin(theta) * cos(phi) + center[0];
            points.get(1)[s] = radius * sin(theta) * sin(phi) + center[1],
            points.get(2)[s] = radius * cos(theta) + center[2];
            ++s;
          }
        }
      } else {
        for (size_t i_theta = 0; i_theta < n_theta; ++i_theta) {
          for (size_t i_phi = 0; i_phi < n_phi; ++i_phi) {
            const double phi = two_pi_over_n_phi * i_phi;
            const double theta = theta_points[i_theta];
            points.get(0)[s] = radius * sin(theta) * cos(phi) + center[0];
            points.get(1)[s] = radius * sin(theta) * sin(phi) + center[1],
            points.get(2)[s] = radius * cos(theta) + center[2];
            ++s;
          }
        }
      }
    }
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }();

  InterpTargetTestHelpers::test_interpolation_target<
      SphereTag, 3, intrp::Tags::Sphere<SphereTag>>(
      created_opts, expected_block_coord_holders);
}

void test_sphere_on_excision_boundary() {
  // A constant sphere on the boundary of a thin logarithmic shell must have
  // every target point assigned. A physical-to-spectral round trip of the
  // constant radius can add enough error to lose points at this boundary.
  constexpr size_t l_max = 28;
  constexpr double radius = 1.01;
  using metavars = InterpTargetTestHelpers::MockMetavars<SphereTag, 3>;
  using target_component =
      InterpTargetTestHelpers::mock_interpolation_target<metavars, SphereTag>;

  for (const auto angular_ordering :
       {ylm::AngularOrdering::Strahlkorper, ylm::AngularOrdering::Cce}) {
    CAPTURE(angular_ordering);
    // This is the map and topology used for a SphericalShells block.
    std::vector<Block<3>> blocks;
    blocks.emplace_back(
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            domain::CoordinateMaps::ProductOf2Maps<
                domain::CoordinateMaps::Interval,
                domain::CoordinateMaps::Identity<2>>{
                domain::CoordinateMaps::Interval{
                    -1.0, 1.0, radius, 1.0765580818594012,
                    domain::CoordinateMaps::Distribution::Logarithmic, 0.0},
                domain::CoordinateMaps::Identity<2>{}},
            domain::CoordinateMaps::SphericalToCartesianPfaffian{}),
        0, DirectionMap<3, BlockNeighbors<3>>{}, "Shell0",
        domain::topologies::spherical_shell);
    ActionTesting::MockRuntimeSystem<metavars> runner{
        {intrp::OptionHolders::Sphere{
             l_max, {0.0, 0.0, 0.0}, radius, angular_ordering},
         Domain<3>{std::move(blocks)}, ::Verbosity::Silent}};
    ActionTesting::set_phase(make_not_null(&runner),
                             Parallel::Phase::Initialization);
    ActionTesting::emplace_component<target_component>(&runner, 0);
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<target_component>(make_not_null(&runner), 0);
    }
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
    auto& target_box =
        ActionTesting::get_databox<target_component>(make_not_null(&runner), 0);
    const auto& cache = ActionTesting::cache<target_component>(runner, 0_st);
    const Slab slab{0.0, 1.0};
    const TimeStepId temporal_id{true, 0, ::Time{slab, 0}};
    const auto block_coord_holders =
        intrp::InterpolationTarget_detail::block_logical_coords<SphereTag>(
            target_box, cache, temporal_id);

    REQUIRE(block_coord_holders.size() == (l_max + 1) * (2 * l_max + 1));
    REQUIRE(
        std::count_if(block_coord_holders.begin(), block_coord_holders.end(),
                      [](const auto& point) { return point.has_value(); }) ==
        static_cast<std::ptrdiff_t>(block_coord_holders.size()));
    for (const auto& point : block_coord_holders) {
      CHECK(point->id == domain::BlockId{0});
      CHECK(get<0>(point->data) == approx(-1.0));
    }
  }
}

void test_sphere_errors() {
  CHECK_THROWS_WITH(
      ([]() {
        const auto created_opts =
            TestHelpers::test_creation<intrp::OptionHolders::Sphere>(
                "Center: [0.05, 0.06, 0.07]\n"
                "Radius: [1.0, 1.0]\n"
                "LMax: 18\n"
                "AngularOrdering: Cce");
      })(),
      Catch::Matchers::ContainsSubstring(
          "into radii for Sphere interpolation target. It already "
          "exists. Existing radii are"));
  CHECK_THROWS_WITH(
      ([]() {
        const auto created_opts =
            TestHelpers::test_creation<intrp::OptionHolders::Sphere>(
                "Center: [0.05, 0.06, 0.07]\n"
                "Radius: [-1.0]\n"
                "LMax: 18\n"
                "AngularOrdering: Cce");
      })(),
      Catch::Matchers::ContainsSubstring("Radius must be positive"));
  CHECK_THROWS_WITH(
      ([]() {
        const auto created_opts =
            TestHelpers::test_creation<intrp::OptionHolders::Sphere>(
                "Center: [0.05, 0.06, 0.07]\n"
                "Radius: -1.0\n"
                "LMax: 18\n"
                "AngularOrdering: Cce");
      })(),
      Catch::Matchers::ContainsSubstring("Radius must be positive"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.Sphere",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  test_sphere_errors();
  test_sphere_on_excision_boundary();
  MAKE_GENERATOR(gen);
  for (size_t num_spheres : {1_st, 2_st, 3_st}) {
    test_interpolation_target_sphere<InterpTargetTestHelpers::ValidPoints::All>(
        make_not_null(&gen), num_spheres, ylm::AngularOrdering::Cce);
    test_interpolation_target_sphere<InterpTargetTestHelpers::ValidPoints::All>(
        make_not_null(&gen), num_spheres, ylm::AngularOrdering::Strahlkorper);
    test_interpolation_target_sphere<
        InterpTargetTestHelpers::ValidPoints::None>(
        make_not_null(&gen), num_spheres, ylm::AngularOrdering::Strahlkorper);
    // ValidPoints::Some is not tested as the radii of the
    // interpolation targets are set randomly so it is difficult to
    // arrange that only a subset of the target points are
    // valid/invalid.
  }
}
