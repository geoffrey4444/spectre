// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"
#include "NumericalAlgorithms/Strahlkorper/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/InvokeCallbacks.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/TimeExtrapolation.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr size_t l_max = 6;
constexpr double radius = 1.34;

template <typename HorizonMetavars>
struct TestCallback : tt::ConformsTo<ah::protocols::Callback> {
 private:
  using Fr = typename HorizonMetavars::frame;

 public:
  template <typename DbTags, typename Metavariables>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const FastFlow::Status failure_reason) {
    const auto& time = db::get<ah::Tags::CurrentTime>(box);
    CHECK(time == std::optional{LinkedMessageId<double>{2.0, {1.0}}});
    CHECK(failure_reason == FastFlow::Status::TruncationTol);
    CHECK(db::get<ah::Tags::FastFlow>(box) == FastFlow{FastFlow::FlowType::Fast,
                                                       1.0, 0.5, 1.e-12, 1.e-2,
                                                       1.2, 5, 100});
    CHECK(db::get<ah::Tags::Dependency>(box) ==
          std::optional{"FakeDependency"});

    const auto& strahlkorper = db::get<ylm::Tags::Strahlkorper<Fr>>(box);
    CHECK(strahlkorper ==
          ylm::Strahlkorper<Fr>{l_max, radius, std::array{0.0, 0.0, 0.0}});

    // Check that we restricted properly
    const size_t expected_size = ylm::Spherepack::physical_size(l_max, l_max);
    tmpl::for_each<ah::vars_to_interpolate_to_target<3, Fr>>(
        [&]<typename Tag>(tmpl::type_<Tag>) {
          const auto& var = db::get<Tag>(box);
          for (size_t i = 0; i < var.size(); i++) {
            CHECK(var[i].size() == expected_size);
          }
        });
  }
};

template <typename Fr>
struct HorizonMetavars : tt::ConformsTo<ah::protocols::HorizonMetavars> {
  using temporal_id_tag = ::Tags::TimeAndPrevious<0>;
  using frame = Fr;

  using horizon_find_callbacks = tmpl::list<TestCallback<HorizonMetavars>>;
  using horizon_find_failure_callbacks = tmpl::list<>;

  using compute_tags_on_element = tmpl::list<>;

  static constexpr ah::Destination destination = ah::Destination::ControlSystem;

  static std::string name() { return "TestingHorizonMetavars"; }
};

template <typename Fr>
struct Metavariables {
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<3>,
                 ah::Tags::ApparentHorizonOptions<HorizonMetavars<Fr>>>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize,
                 ah::Tags::PreviousSurface<HorizonMetavars<Fr>>>;
  using component_list = tmpl::list<>;
};

template <typename Fr>
void run_test(const std::optional<size_t> number_of_points,
              const size_t previous_finds) {
  CAPTURE(number_of_points);
  CAPTURE(previous_finds);
  const domain::creators::Sphere sphere_creator{
      0.9, 2.0, domain::creators::Sphere::Excision{nullptr}, 0_st, 4_st, true};

  ah::HorizonOptions<Fr> options{};
  options.number_of_points_for_time_extrapolation = number_of_points;
  Parallel::GlobalCache<Metavariables<Fr>> cache{
      {sphere_creator.create_domain(), std::move(options)},
      {sphere_creator.functions_of_time(),
       ah::Storage::LockedPreviousSurface<Fr>{}}};

  const LinkedMessageId<double> time{2.0, {1.0}};
  const FastFlow fast_flow{
      FastFlow::FlowType::Fast, 1.0, 0.5, 1.e-12, 1.e-2, 1.2, 5, 100};
  const std::optional<std::string> dependency{"FakeDependency"};

  ah::Storage::Iteration<Fr> current_iteration{};
  current_iteration.strahlkorper =
      ylm::Strahlkorper<Fr>{l_max, radius, std::array{0.0, 0.0, 0.0}};
  current_iteration.intersecting_element_ids = {ElementId<3>{0},
                                                ElementId<3>{1}};
  const size_t l_mesh =
      fast_flow.current_l_mesh(current_iteration.strahlkorper);
  // The actual values don't matter for this test
  current_iteration.interpolated_vars =
      Variables<ah::vars_to_interpolate_to_target<3, Fr>>{
          ylm::Spherepack::physical_size(l_mesh, l_mesh), 3.21};

  ah::Storage::SingleTimeStorage<Fr> current_time_storage{};
  current_time_storage.current_iteration = current_iteration;
  current_time_storage.dependency = dependency;

  std::unordered_map<LinkedMessageId<double>,
                     ah::Storage::SingleTimeStorage<Fr>>
      all_storage{};
  all_storage[time] = current_time_storage;

  const FastFlow::Status status = FastFlow::Status::TruncationTol;

  // A surface with r(t) = radius + 0.01 (t - 2)^3 distinguishes three-point
  // and four-point derivatives at the accepted time t = 2.
  std::deque<ah::Storage::PreviousSurface<Fr>> previous_surfaces{};
  for (size_t i = 1; i <= previous_finds; ++i) {
    const double dt = static_cast<double>(i);
    previous_surfaces.emplace_back(
        LinkedMessageId<double>{.id = time.id - dt,
                                .previous = time.id - dt - 1.0},
        ylm::Strahlkorper<Fr>{l_max, radius - 0.01 * dt * dt * dt,
                              std::array{0.0, 0.0, 0.0}},
        std::unordered_set<ElementId<3>>{});
  }

  auto box = db::create<db::AddSimpleTags<tmpl::list<
      ah::Tags::CurrentTime, ah::Tags::Storage<Fr>,
      ah::Tags::PreviousSurfaces<Fr>, ah::Tags::ExtrapolationPredictionErrors,
      ah::Tags::FastFlow, ah::Tags::Verbosity, ylm::Tags::Strahlkorper<Fr>,
      ylm::Tags::TimeDerivStrahlkorper<Fr>, ah::Tags::Dependency,
      ylm::Tags::CartesianCoords<Frame::Inertial>,
      ::Tags::Variables<ah::vars_to_interpolate_to_target<3, Fr>>>>>(
      std::optional{time}, all_storage, std::move(previous_surfaces),
      ah::PredictionErrors{},
      FastFlow{FastFlow::FlowType::Fast, 1.0, 0.5, 1.e-12, 1.e-2, 1.2, 5, 100},
      ::Verbosity::Quiet, ylm::Strahlkorper<Fr>{}, ylm::Strahlkorper<Fr>{},
      std::optional<std::string>{}, tnsr::I<DataVector, 3>{},
      Variables<ah::vars_to_interpolate_to_target<3, Fr>>{});

  ah::invoke_callbacks<HorizonMetavars<Fr>>(make_not_null(&box), cache, status);

  const auto& box_previous_surfaces =
      db::get<ah::Tags::PreviousSurfaces<Fr>>(box);
  // Each fixture starts with either an empty history or a full history.
  const size_t expected_history_size = std::max(size_t{1}, previous_finds);
  REQUIRE(box_previous_surfaces.size() == expected_history_size);
  for (size_t i = 0; i < expected_history_size; ++i) {
    CHECK(box_previous_surfaces[i].time.id == time.id - static_cast<double>(i));
  }
  CHECK(box_previous_surfaces.front().time == time);
  CHECK(box_previous_surfaces.front().surface ==
        current_iteration.strahlkorper);
  CHECK(box_previous_surfaces.front().intersecting_element_ids ==
        current_iteration.intersecting_element_ids);
  const auto& cache_previous_surface =
      get<ah::Tags::PreviousSurface<HorizonMetavars<Fr>>>(cache);
  CHECK(cache_previous_surface.surface == box_previous_surfaces.front());
  CHECK(db::get<ylm::Tags::Strahlkorper<Fr>>(box) ==
        current_iteration.strahlkorper);
  const auto& derivative = db::get<ylm::Tags::TimeDerivStrahlkorper<Fr>>(box);
  if (previous_finds == 0) {
    CHECK(
        derivative.coefficients() ==
        DataVector{current_iteration.strahlkorper.coefficients().size(), 0.0});
  } else {
    // The three-point derivative is -0.02. The four-point derivative of the
    // cubic is exact and would give zero.
    CHECK(derivative.average_radius() == approx(-0.02));
  }

  const auto& errors = db::get<ah::Tags::ExtrapolationPredictionErrors>(box);
  if (not number_of_points.has_value() and previous_finds == 4) {
    for (const auto& error : errors) {
      REQUIRE(error.has_value());
    }
    CHECK(*errors[0] > 1.e-4);
    CHECK(*errors[1] > 1.e-4);
    CHECK(*errors[2] == approx(0.0));

    const auto restored_box = serialize_and_deserialize(box);
    CHECK(db::get<ah::Tags::PreviousSurfaces<Fr>>(restored_box) ==
          box_previous_surfaces);
    CHECK(db::get<ah::Tags::ExtrapolationPredictionErrors>(restored_box) ==
          errors);
  } else {
    CHECK(errors == ah::PredictionErrors{});
  }
  CHECK(db::get<ah::Tags::Dependency>(box) == dependency);
  CHECK(
      db::get<::Tags::Variables<ah::vars_to_interpolate_to_target<3, Fr>>>(box)
          .number_of_grid_points() ==
      ylm::Spherepack::physical_size(l_max, l_max));
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.InvokeCallbacks",
                  "[ApparentHorizonFinder][Unit]") {
  domain::creators::register_derived_with_charm();
  run_test<Frame::Grid>(3, 0);
  run_test<Frame::Inertial>(3, 3);
  run_test<Frame::Inertial>(2, 3);
  run_test<Frame::Inertial>(4, 4);
  run_test<Frame::Inertial>(std::nullopt, 0);
  run_test<Frame::Inertial>(std::nullopt, 4);
}
}  // namespace
