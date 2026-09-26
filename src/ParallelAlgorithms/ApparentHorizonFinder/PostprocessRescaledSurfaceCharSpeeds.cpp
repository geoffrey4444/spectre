// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/PostprocessRescaledSurfaceCharSpeeds.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/FrameTransform.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeRescaledSurfaceCharSpeedVars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/SampleRescaledSurfaceCharSpeeds.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace ah {
std::vector<double> postprocess_rescaled_surface_char_speeds(
    const ylm::Strahlkorper<Frame::Distorted>& horizon,
    const std::optional<ylm::Strahlkorper<Frame::Distorted>>&
        time_deriv_horizon,
    const std::vector<ElementId<3>>& element_ids,
    const std::vector<Mesh<3>>& meshes,
    const std::vector<tnsr::aa<DataVector, 3>>& spacetime_metrics,
    const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time, const double time,
    const RescaledSurfaceCharSpeedOptions& options,
    const std::optional<std::vector<std::string>>& blocks) {
  if (element_ids.size() != meshes.size() or
      element_ids.size() != spacetime_metrics.size()) {
    throw std::invalid_argument(
        "Element IDs, meshes, and spacetime metrics must have the same size.");
  }
  if (not std::isfinite(time)) {
    throw std::invalid_argument("The observation time must be finite.");
  }
  std::vector<std::string> all_block_names{};
  std::unordered_set<std::string> selected_blocks{};
  for (const auto& block : domain.blocks()) {
    all_block_names.push_back(block.name());
    if (not block.is_time_dependent() or block.has_distorted_frame()) {
      selected_blocks.insert(block.name());
    }
    if (block.is_time_dependent() and block.has_distorted_frame()) {
      const auto check_functions_of_time = [&functions_of_time,
                                            time](const auto& map) {
        for (const auto& name : map.function_of_time_names()) {
          const auto function = functions_of_time.find(name);
          if (function == functions_of_time.end() or
              function->second == nullptr) {
            throw std::invalid_argument("Missing function of time '" + name +
                                        "'.");
          }
          const auto bounds = function->second->time_bounds();
          if (time < bounds[0] or time > bounds[1]) {
            throw std::invalid_argument(
                "Function of time '" + name +
                "' is not valid at the observation time.");
          }
        }
      };
      check_functions_of_time(block.moving_mesh_grid_to_distorted_map());
      check_functions_of_time(block.moving_mesh_distorted_to_inertial_map());
    }
  }
  if (blocks.has_value()) {
    selected_blocks = domain::expand_block_groups_to_block_names(
        *blocks, all_block_names, domain.block_groups());
  }
  Storage::RescaledSurfaceCharSpeeds data{};
  initialize_rescaled_surface_char_speeds(
      make_not_null(&data), horizon, time_deriv_horizon.value_or(horizon),
      time_deriv_horizon.has_value(), options, domain, functions_of_time, time);
  if (data.status != Storage::RescaledSurfaceStatus::Valid) {
    return rescaled_surface_char_speed_row(time, data);
  }
  std::unordered_map<ElementId<3>, Storage::VolumeVariables<Frame::Distorted>>
      volume_variables{};
  for (size_t i = 0; i < element_ids.size(); ++i) {
    const auto& id = element_ids[i];
    if (id.block_id() >= domain.blocks().size()) {
      throw std::invalid_argument("Element block ID is outside the domain.");
    }
    const auto& block = domain.blocks()[id.block_id()];
    if (not selected_blocks.contains(block.name())) {
      continue;
    }
    if (block.is_time_dependent() and not block.has_distorted_frame()) {
      throw std::invalid_argument("Selected block '" + block.name() +
                                  "' has no distorted frame.");
    }
    const auto& mesh = meshes[i];
    const auto& metric = spacetime_metrics[i];
    for (const auto& component : metric) {
      if (component.size() != mesh.number_of_grid_points()) {
        throw std::invalid_argument(
            "Spacetime metric size must match its element mesh.");
      }
    }
    auto [entry, inserted] = volume_variables.try_emplace(id);
    if (not inserted) {
      throw std::invalid_argument("Duplicate element in saved volume data.");
    }
    auto& volume = entry->second;
    volume.mesh = mesh;
    // The shared horizon interpolator also carries fields that the speed
    // diagnostic does not use. Initialize them to avoid interpolating
    // uninitialized memory.
    volume.vars_to_interpolate_to_target.initialize(
        mesh.number_of_grid_points(), 0.0);
    volume.rescaled_surface_vars.emplace();
    compute_rescaled_surface_char_speed_vars(
        make_not_null(&*volume.rescaled_surface_vars), metric, domain, mesh, id,
        time, functions_of_time);
    const auto inertial_metric = gr::spatial_metric(metric);
    auto& spatial_metric =
        get<gr::Tags::SpatialMetric<DataVector, 3, Frame::Distorted>>(
            volume.vars_to_interpolate_to_target);
    if (block.is_time_dependent()) {
      const ElementMap<3, Frame::Grid> element_to_grid{
          id, block.moving_mesh_logical_to_grid_map().get_clone()};
      const auto distorted_coords = block.moving_mesh_grid_to_distorted_map()(
          element_to_grid(logical_coordinates(mesh)), time, functions_of_time);
      transform::to_different_frame(
          make_not_null(&spatial_metric), inertial_metric,
          block.moving_mesh_distorted_to_inertial_map().jacobian(
              distorted_coords, time, functions_of_time));
    } else {
      for (size_t component = 0; component < spatial_metric.size();
           ++component) {
        spatial_metric[component] = inertial_metric[component];
      }
    }
    Scalar<DataVector> determinant{};
    determinant_and_inverse(
        make_not_null(&determinant),
        make_not_null(&get<gr::Tags::InverseSpatialMetric<DataVector, 3,
                                                          Frame::Distorted>>(
            volume.vars_to_interpolate_to_target)),
        spatial_metric);
  }
  if (not sample_rescaled_surface_char_speeds(
          make_not_null(&data), volume_variables, domain, functions_of_time,
          time, selected_blocks)) {
    data.status = Storage::RescaledSurfaceStatus::MissingBlockCoverage;
  }
  return rescaled_surface_char_speed_row(time, data);
}
}  // namespace ah
