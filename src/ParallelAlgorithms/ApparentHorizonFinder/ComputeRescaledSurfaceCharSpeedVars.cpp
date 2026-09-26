// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeRescaledSurfaceCharSpeedVars.hpp"

#include <cstddef>

#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace ah {
void compute_rescaled_surface_char_speed_vars(
    const gsl::not_null<Variables<rescaled_surface_char_speed_vars>*> result,
    const tnsr::aa<DataVector, 3>& spacetime_metric, const Domain<3>& domain,
    const Mesh<3>& mesh, const ElementId<3>& element_id, const double time,
    const domain::FunctionsOfTimeMap& functions_of_time) {
  const auto number_of_points = get<0, 0>(spacetime_metric).size();
  result->initialize(number_of_points);
  using SpatialMetric = gr::Tags::SpatialMetric<DataVector, 3>;
  using InverseSpatialMetric = gr::Tags::InverseSpatialMetric<DataVector, 3>;
  using InertialShift = gr::Tags::Shift<DataVector, 3>;
  TempBuffer<tmpl::list<SpatialMetric, InverseSpatialMetric, InertialShift>>
      buffer{number_of_points};
  auto& spatial_metric = get<SpatialMetric>(buffer);
  auto& inverse_spatial_metric = get<InverseSpatialMetric>(buffer);
  auto& inertial_shift = get<InertialShift>(buffer);
  auto& lapse = get<gr::Tags::Lapse<DataVector>>(*result);
  auto& distorted_shift =
      get<gr::Tags::Shift<DataVector, 3, Frame::Distorted>>(*result);
  gr::spatial_metric(make_not_null(&spatial_metric), spacetime_metric);
  // Use the lapse storage for the determinant before computing the lapse.
  determinant_and_inverse(make_not_null(&lapse),
                          make_not_null(&inverse_spatial_metric),
                          spatial_metric);
  gr::shift(make_not_null(&inertial_shift), spacetime_metric,
            inverse_spatial_metric);
  gr::lapse(make_not_null(&lapse), inertial_shift, spacetime_metric);

  const auto& block = domain.blocks()[element_id.block_id()];
  if (not block.is_time_dependent()) {
    // All physical frames coincide for a stationary domain.
    for (size_t i = 0; i < 3; ++i) {
      distorted_shift.get(i) = inertial_shift.get(i);
    }
    return;
  }
  ASSERT(block.has_distorted_frame(),
         "Rescaled-surface characteristic speeds need a distorted frame in "
         "block "
             << element_id.block_id() << ".");
  const ElementMap<3, Frame::Grid> element_to_grid{
      element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
  const auto distorted_coords = block.moving_mesh_grid_to_distorted_map()(
      element_to_grid(logical_coordinates(mesh)), time, functions_of_time);
  const auto& distorted_to_inertial =
      block.moving_mesh_distorted_to_inertial_map();
  const auto [inertial_coords, inverse_jacobian, jacobian, frame_velocity] =
      distorted_to_inertial.coords_frame_velocity_jacobians(
          distorted_coords, time, functions_of_time);
  tenex::evaluate<ti::I>(make_not_null(&distorted_shift),
                         inverse_jacobian(ti::I, ti::j) *
                             (inertial_shift(ti::J) + frame_velocity(ti::J)));
}
}  // namespace ah
