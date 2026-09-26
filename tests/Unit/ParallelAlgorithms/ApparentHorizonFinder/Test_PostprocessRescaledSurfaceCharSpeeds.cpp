// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeRescaledSurfaceCharSpeedVars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeVarsToInterpolateToTarget.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/PostprocessRescaledSurfaceCharSpeeds.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/SampleRescaledSurfaceCharSpeeds.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"

namespace {
void test_postprocessing(const bool moving) {
  CAPTURE(moving);
  auto domain =
      domain::creators::Sphere{
          1.0,  3.0,   domain::creators::Sphere::Excision{nullptr},
          0_st, 12_st, true}
          .create_domain();
  domain::FunctionsOfTimeMap functions_of_time{};
  if (moving) {
    using Scale = domain::CoordinateMaps::TimeDependent::CubicScale<3>;
    using Function = domain::FunctionsOfTime::PiecewisePolynomial<2>;
    functions_of_time["GridScale"] = std::make_unique<Function>(
        0.0, std::array{DataVector{1.1}, DataVector{0.3}, DataVector{0.0}},
        2.0);
    functions_of_time["InertialScale"] = std::make_unique<Function>(
        0.0, std::array{DataVector{1.2}, DataVector{0.1}, DataVector{0.0}},
        2.0);
    const Scale grid_scale{100.0, "GridScale", "GridScale"};
    const Scale inertial_scale{100.0, "InertialScale", "InertialScale"};
    for (size_t id = 0; id < domain.blocks().size(); ++id) {
      domain.inject_time_dependent_map_for_block(
          id,
          domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
              grid_scale, inertial_scale),
          domain::make_coordinate_map_base<Frame::Grid, Frame::Distorted>(
              grid_scale),
          domain::make_coordinate_map_base<Frame::Distorted, Frame::Inertial>(
              inertial_scale));
    }
  }
  const Mesh<3> mesh{12, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  tnsr::aa<DataVector, 3> metric{mesh.number_of_grid_points(), 0.0};
  get<0, 0>(metric) = -1.0;
  get<1, 1>(metric) = 1.0;
  get<2, 2>(metric) = 1.0;
  get<3, 3>(metric) = 1.0;
  std::vector<ElementId<3>> ids{};
  std::vector<Mesh<3>> meshes{};
  std::vector<tnsr::aa<DataVector, 3>> metrics{};
  for (size_t id = 0; id < domain.blocks().size(); ++id) {
    ids.emplace_back(id);
    meshes.push_back(mesh);
    metrics.push_back(metric);
  }
  const ylm::Strahlkorper<Frame::Distorted> horizon{4_st, 2.0,
                                                    std::array{0.0, 0.0, 0.0}};
  auto derivative = horizon;
  derivative.coefficients() *= 0.1;
  const auto sample = [&](const auto& dt, const auto& element_ids,
                          const auto& element_meshes, const auto& fields) {
    return ah::postprocess_rescaled_surface_char_speeds(
        horizon, dt, element_ids, element_meshes, fields, domain,
        functions_of_time, 0.0,
        ah::RescaledSurfaceCharSpeedOptions{"ExcisionSphere", 3, 1.e-7});
  };
  const auto row = sample(derivative, ids, meshes, metrics);
  REQUIRE(row.size() == 11);
  CHECK(row[1] == 0.0);
  const double q_min = (moving ? 0.55 : 0.5) * (1.0 + 1.e-7);
  const std::array factors{1.0, 1.0 - 0.25 * (1.0 - q_min), q_min};
  const auto interpolation_approx = Approx::custom().epsilon(2.e-5).scale(1.0);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(row[2 + 3 * i] == approx(gsl::at(factors, i)));
    // Grid-to-distorted velocity must not enter the surface speed.
    const double expected =
        -1.0 + gsl::at(factors, i) * (moving ? 0.1 * 2.0 + 1.2 * 0.2 : 0.2);
    CHECK(row[3 + 3 * i] == interpolation_approx(expected));
    CHECK(row[4 + 3 * i] == interpolation_approx(expected));
  }
  const auto missing_derivative = sample(std::nullopt, ids, meshes, metrics);
  CHECK(missing_derivative[1] == 1.0);
  CHECK(std::isnan(missing_derivative[3]));

  // Compare the discrete result with the evolution's nodal preparation. A
  // nonconstant metric distinguishes this from deriving fields after
  // interpolation to the surfaces.
  std::unordered_map<ElementId<3>,
                     ah::Storage::VolumeVariables<Frame::Distorted>>
      volumes{};
  std::unordered_set<std::string> blocks{};
  const auto num_points = mesh.number_of_grid_points();
  const tnsr::aa<DataVector, 3> pi{num_points, 0.0};
  const tnsr::iaa<DataVector, 3> phi{num_points, 0.0};
  const tnsr::ijaa<DataVector, 3> deriv_phi{num_points, 0.0};
  for (size_t i = 0; i < ids.size(); ++i) {
    for (size_t p = 0; p < num_points; ++p) {
      get<0, 0>(metrics[i])[p] = -1.0 - 0.01 * static_cast<double>(p % 5);
      get<1, 1>(metrics[i])[p] = 1.0 + 0.03 * static_cast<double>(p % 7);
    }
    auto& volume = volumes[ids[i]];
    volume.mesh = mesh;
    ah::compute_vars_to_interpolate_to_target(
        make_not_null(&volume.vars_to_interpolate_to_target), metrics[i], pi,
        phi, deriv_phi, LinkedMessageId<double>{0.0, std::nullopt}, domain,
        mesh, ids[i], functions_of_time);
    volume.rescaled_surface_vars.emplace();
    ah::compute_rescaled_surface_char_speed_vars(
        make_not_null(&*volume.rescaled_surface_vars), metrics[i], domain, mesh,
        ids[i], 0.0, functions_of_time);
    blocks.insert(domain.blocks()[ids[i].block_id()].name());
  }
  ah::Storage::RescaledSurfaceCharSpeeds online{};
  ah::initialize_rescaled_surface_char_speeds(
      make_not_null(&online), horizon, derivative, true,
      ah::RescaledSurfaceCharSpeedOptions{"ExcisionSphere", 3, 1.e-7}, domain,
      functions_of_time, 0.0);
  REQUIRE(ah::sample_rescaled_surface_char_speeds(
      make_not_null(&online), volumes, domain, functions_of_time, 0.0, blocks));
  CHECK_ITERABLE_APPROX(sample(derivative, ids, meshes, metrics),
                        ah::rescaled_surface_char_speed_row(0.0, online));

  auto duplicate_ids = ids;
  duplicate_ids.back() = duplicate_ids.front();
  CHECK_THROWS_WITH(sample(derivative, duplicate_ids, meshes, metrics),
                    Catch::Matchers::ContainsSubstring("Duplicate element"));
  if (moving) {
    functions_of_time.erase("GridScale");
    CHECK_THROWS_WITH(sample(derivative, ids, meshes, metrics),
                      Catch::Matchers::ContainsSubstring("Missing function"));
    return;
  }
  ids.pop_back();
  meshes.pop_back();
  metrics.pop_back();
  CHECK(sample(derivative, ids, meshes, metrics)[1] == 4.0);
  metrics.pop_back();
  CHECK_THROWS_WITH(sample(derivative, ids, meshes, metrics),
                    Catch::Matchers::ContainsSubstring("same size"));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.ApparentHorizonFinder.PostprocessRescaledSurfaceCharSpeeds",
    "[Unit][ApparentHorizonFinder]") {
  test_postprocessing(false);
  test_postprocessing(true);
}
