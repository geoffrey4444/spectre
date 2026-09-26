// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeRescaledSurfaceCharSpeedVars.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void test_volume_fields(const bool moving) {
  CAPTURE(moving);
  const double time = 0.4;
  const Mesh<3> mesh{4, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  auto domain = domain::creators::Brick{
      {2.0, -1.0, 0.5},
      {3.0, 1.0, 1.5},
      {1, 1, 1},
      {4, 4, 4}}.create_domain();
  const ElementId<3> element_id{
      0, std::array{SegmentId{1, 1}, SegmentId{1, 0}, SegmentId{1, 1}}};
  domain::FunctionsOfTimeMap functions_of_time{};
  const double initial_scale = 1.2;
  const double scale_velocity = 0.13;
  const DataVector grid_translation{0.1, -0.2, 0.3};
  const DataVector grid_velocity{-0.7, 0.8, -0.9};
  const DataVector inertial_translation{-0.2, 0.4, 0.1};
  const DataVector inertial_velocity{0.2, -0.3, 0.5};
  if (moving) {
    using Translation = domain::CoordinateMaps::TimeDependent::Translation<3>;
    using Scale = domain::CoordinateMaps::TimeDependent::CubicScale<3>;
    const Translation grid_to_distorted{"GridTranslation"};
    const Scale distorted_scale{100.0, "Scale", "Scale"};
    const Translation distorted_translation{"InertialTranslation"};
    domain.inject_time_dependent_map_for_block(
        0,
        domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            grid_to_distorted, distorted_scale, distorted_translation),
        domain::make_coordinate_map_base<Frame::Grid, Frame::Distorted>(
            grid_to_distorted),
        domain::make_coordinate_map_base<Frame::Distorted, Frame::Inertial>(
            distorted_scale, distorted_translation));
    using FunctionOfTime = domain::FunctionsOfTime::PiecewisePolynomial<2>;
    functions_of_time["Scale"] = std::make_unique<FunctionOfTime>(
        0.0,
        std::array{DataVector{initial_scale}, DataVector{scale_velocity},
                   DataVector{0.0}},
        2.0);
    functions_of_time["GridTranslation"] = std::make_unique<FunctionOfTime>(
        0.0, std::array{grid_translation, grid_velocity, DataVector(3, 0.0)},
        2.0);
    functions_of_time["InertialTranslation"] = std::make_unique<FunctionOfTime>(
        0.0,
        std::array{inertial_translation, inertial_velocity, DataVector(3, 0.0)},
        2.0);
  }
  const auto& block = domain.blocks()[0];
  const auto logical_coords = logical_coordinates(mesh);
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{};
  tnsr::I<DataVector, 3, Frame::Distorted> distorted_coords{};
  if (moving) {
    const ElementMap<3, Frame::Grid> element_map{
        element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
    const auto grid_coords = element_map(logical_coords);
    distorted_coords = block.moving_mesh_grid_to_distorted_map()(
        grid_coords, time, functions_of_time);
    inertial_coords = block.moving_mesh_grid_to_inertial_map()(
        grid_coords, time, functions_of_time);
  } else {
    const ElementMap<3, Frame::Inertial> element_map{
        element_id, block.stationary_map().get_clone()};
    inertial_coords = element_map(logical_coords);
  }
  const gr::Solutions::KerrSchild solution{
      1.0, {0.1, -0.2, 0.3}, {0.0, 0.0, 0.0}};
  using Lapse = gr::Tags::Lapse<DataVector>;
  using InertialShift = gr::Tags::Shift<DataVector, 3>;
  using SpatialMetric = gr::Tags::SpatialMetric<DataVector, 3>;
  const auto solution_vars = solution.variables(
      inertial_coords, time, tmpl::list<Lapse, InertialShift, SpatialMetric>{});
  const auto metric = gr::spacetime_metric(get<Lapse>(solution_vars),
                                           get<InertialShift>(solution_vars),
                                           get<SpatialMetric>(solution_vars));
  Variables<ah::rescaled_surface_char_speed_vars> result{};
  ah::compute_rescaled_surface_char_speed_vars(make_not_null(&result), metric,
                                               domain, mesh, element_id, time,
                                               functions_of_time);
  CHECK_ITERABLE_APPROX(get<Lapse>(result), get<Lapse>(solution_vars));
  const auto& shift =
      get<gr::Tags::Shift<DataVector, 3, Frame::Distorted>>(result);
  for (size_t i = 0; i < 3; ++i) {
    DataVector expected_shift = get<InertialShift>(solution_vars).get(i);
    if (moving) {
      expected_shift +=
          scale_velocity * distorted_coords.get(i) + inertial_velocity[i];
      expected_shift /= initial_scale + time * scale_velocity;
    }
    CHECK_ITERABLE_APPROX(shift.get(i), expected_shift);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.RescaledSurfaceVolumeVars",
                  "[ApparentHorizonFinder][Unit]") {
  test_volume_fields(false);
  test_volume_fields(true);
}
