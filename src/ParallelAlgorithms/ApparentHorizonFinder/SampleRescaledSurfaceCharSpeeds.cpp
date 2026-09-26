// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/SampleRescaledSurfaceCharSpeeds.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CoordsToDifferentFrame.hpp"
#include "Domain/Domain.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolateVolumeVars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/RescaledSurfaceCharSpeeds.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace ah {
namespace {
bool positive_finite_radii(const DataVector& radii) {
  return std::all_of(radii.begin(), radii.end(), [](const double radius) {
    return std::isfinite(radius) and radius > 0.0;
  });
}

std::vector<BlockLogicalCoords<3>> checked_surface_coordinates(
    const gsl::not_null<Storage::RescaledSurfaceStatus*> status,
    const ylm::Strahlkorper<Frame::Distorted>& surface, const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time, const double time,
    const std::unordered_set<std::string>& blocks_for_interpolation) {
  if (not positive_finite_radii(get(ylm::radius(surface)))) {
    *status = Storage::RescaledSurfaceStatus::InvalidGeometry;
    return {};
  }
  auto coordinates = block_logical_coordinates(
      domain, ylm::cartesian_coords(surface), time, functions_of_time);
  for (const auto& coordinate : coordinates) {
    if (not coordinate.has_value()) {
      *status = Storage::RescaledSurfaceStatus::OutsideDomain;
      return {};
    }
    if (not blocks_for_interpolation.contains(
            domain.blocks()[coordinate->id.get_index()].name())) {
      *status = Storage::RescaledSurfaceStatus::MissingBlockCoverage;
      return {};
    }
  }
  return coordinates;
}
}  // namespace

void initialize_rescaled_surface_char_speeds(
    const gsl::not_null<Storage::RescaledSurfaceCharSpeeds*> data,
    const ylm::Strahlkorper<Frame::Distorted>& horizon,
    const ylm::Strahlkorper<Frame::Distorted>& time_deriv_horizon,
    const bool time_derivative_is_available,
    const RescaledSurfaceCharSpeedOptions& options, const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time, const double time) {
  *data = Storage::RescaledSurfaceCharSpeeds{};
  data->horizon = horizon;
  data->time_deriv_horizon = time_deriv_horizon;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  data->radius_factors.assign(options.number_of_surfaces, nan);
  data->min_speeds.assign(options.number_of_surfaces, nan);
  data->max_speeds.assign(options.number_of_surfaces, nan);
  const auto horizon_radius = ylm::radius(horizon);
  const auto& excision_spheres = domain.excision_spheres();
  if (not positive_finite_radii(get(horizon_radius)) or
      not excision_spheres.contains(options.excision_sphere) or
      (time_derivative_is_available and
       (horizon.l_max() != time_deriv_horizon.l_max() or
        horizon.m_max() != time_deriv_horizon.m_max() or
        not equal_within_roundoff(horizon.expansion_center(),
                                  time_deriv_horizon.expansion_center())))) {
    data->status = Storage::RescaledSurfaceStatus::InvalidGeometry;
    return;
  }
  const auto& excision = excision_spheres.at(options.excision_sphere);
  const auto& center = horizon.expansion_center();
  if (not std::isfinite(excision.radius()) or excision.radius() <= 0.0) {
    data->status = Storage::RescaledSurfaceStatus::InvalidGeometry;
    return;
  }
  for (size_t d = 0; d < 3; ++d) {
    if (not std::isfinite(gsl::at(center, d)) or
        not equal_within_roundoff(gsl::at(center, d),
                                  excision.center().get(d))) {
      data->status = Storage::RescaledSurfaceStatus::InvalidGeometry;
      return;
    }
  }
  const auto number_of_points = horizon.ylm_spherepack().physical_size();
  // Stationary domains identify the physical frames without storing an
  // explicit grid-to-distorted map.
  DataVector excision_radius(number_of_points, excision.radius());
  if (domain.is_time_dependent()) {
    const ylm::Strahlkorper<Frame::Grid> grid_excision{
        horizon.l_max(), horizon.m_max(), excision_radius, center};
    const auto grid_coords = ylm::cartesian_coords(grid_excision);
    const auto grid_block_coords =
        block_logical_coordinates(domain, grid_coords, time, functions_of_time);
    for (const auto& coordinate : grid_block_coords) {
      if (not coordinate.has_value()) {
        data->status = Storage::RescaledSurfaceStatus::OutsideDomain;
        return;
      }
      if (not domain.blocks()[coordinate->id.get_index()]
                  .has_distorted_frame()) {
        data->status = Storage::RescaledSurfaceStatus::InvalidGeometry;
        return;
      }
    }
    tnsr::I<DataVector, 3, Frame::Distorted> distorted_coords{number_of_points};
    coords_to_different_frame(make_not_null(&distorted_coords), grid_coords,
                              domain, functions_of_time, time);
    for (size_t p = 0; p < number_of_points; ++p) {
      excision_radius[p] = std::hypot(get<0>(distorted_coords)[p] - center[0],
                                      get<1>(distorted_coords)[p] - center[1],
                                      get<2>(distorted_coords)[p] - center[2]);
      if (not std::isfinite(excision_radius[p]) or excision_radius[p] <= 0.0) {
        data->status = Storage::RescaledSurfaceStatus::InvalidGeometry;
        return;
      }
      // Radii can be compared at the original collocation angles only when the
      // grid-to-distorted map preserves rays about the common center.
      for (size_t d = 0; d < 3; ++d) {
        if (not equal_within_roundoff(
                (distorted_coords.get(d)[p] - gsl::at(center, d)) /
                    excision_radius[p],
                (grid_coords.get(d)[p] - gsl::at(center, d)) /
                    excision.radius())) {
          data->status = Storage::RescaledSurfaceStatus::InvalidGeometry;
          return;
        }
      }
    }
  }
  auto factors = rescaled_surface_factors(get(horizon_radius), excision_radius,
                                          options.number_of_surfaces,
                                          options.relative_excision_margin);
  if (not factors.has_value()) {
    data->status = Storage::RescaledSurfaceStatus::InvalidGeometry;
    return;
  }
  data->radius_factors = std::move(*factors);
  if (not time_derivative_is_available) {
    data->status = Storage::RescaledSurfaceStatus::MissingTimeDerivative;
  }
}

bool sample_rescaled_surface_char_speeds(
    const gsl::not_null<Storage::RescaledSurfaceCharSpeeds*> data,
    const std::unordered_map<ElementId<3>,
                             Storage::VolumeVariables<Frame::Distorted>>&
        volume_variables,
    const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time, const double time,
    const std::unordered_set<std::string>& blocks_for_interpolation) {
  if (data->status != Storage::RescaledSurfaceStatus::Valid) {
    return true;
  }
  auto& interpolation = data->interpolation;
  while (data->next_surface < data->radius_factors.size()) {
    if (not interpolation.block_coord_holders.has_value()) {
      interpolation.strahlkorper = data->horizon;
      interpolation.strahlkorper.coefficients() *=
          data->radius_factors[data->next_surface];
      interpolation.block_coord_holders = checked_surface_coordinates(
          make_not_null(&data->status), interpolation.strahlkorper, domain,
          functions_of_time, time, blocks_for_interpolation);
      if (data->status != Storage::RescaledSurfaceStatus::Valid) {
        return true;
      }
      // Check the whole family before waiting for data. Retain only the
      // current surface's coordinates to keep the checkpointed state small.
      if (data->next_surface == 0) {
        for (size_t i = 1; i < data->radius_factors.size(); ++i) {
          auto surface = data->horizon;
          surface.coefficients() *= data->radius_factors[i];
          checked_surface_coordinates(make_not_null(&data->status), surface,
                                      domain, functions_of_time, time,
                                      blocks_for_interpolation);
          if (data->status != Storage::RescaledSurfaceStatus::Valid) {
            return true;
          }
        }
      }
      const auto number_of_points = interpolation.block_coord_holders->size();
      interpolation.indices_interpolated_to_thus_far.assign(number_of_points,
                                                            false);
      interpolation.rescaled_surface_vars.emplace(number_of_points);
    }
    for (const auto& [element_id, volume] : volume_variables) {
      interpolate_volume_data(make_not_null(&interpolation), volume,
                              element_id);
      if (interpolation.interpolation_is_complete()) {
        break;
      }
    }
    if (not interpolation.interpolation_is_complete()) {
      return false;
    }
    auto time_deriv_surface = data->time_deriv_horizon;
    time_deriv_surface.coefficients() *=
        data->radius_factors[data->next_surface];
    const auto [minimum, maximum] = rescaled_surface_char_speed_extrema(
        interpolation.strahlkorper, time_deriv_surface,
        get<gr::Tags::Lapse<DataVector>>(*interpolation.rescaled_surface_vars),
        get<gr::Tags::Shift<DataVector, 3, Frame::Distorted>>(
            *interpolation.rescaled_surface_vars),
        get<gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Distorted>>(
            interpolation.interpolated_vars));
    data->min_speeds[data->next_surface] = minimum;
    data->max_speeds[data->next_surface] = maximum;
    if (not std::isfinite(minimum) or not std::isfinite(maximum)) {
      data->status = Storage::RescaledSurfaceStatus::NonfiniteSpeed;
      return true;
    }
    ++data->next_surface;
    interpolation.reset_for_next_iteration();
  }
  return true;
}

std::vector<std::string> rescaled_surface_char_speed_legend(
    const size_t number_of_surfaces) {
  std::vector<std::string> legend{"Time", "Status"};
  legend.reserve(2 + 3 * number_of_surfaces);
  for (size_t i = 0; i < number_of_surfaces; ++i) {
    const auto suffix = "_" + std::to_string(i);
    legend.push_back("RadiusFactor" + suffix);
    legend.push_back("MinCharSpeed" + suffix);
    legend.push_back("MaxCharSpeed" + suffix);
  }
  return legend;
}

std::vector<double> rescaled_surface_char_speed_row(
    const double time, const Storage::RescaledSurfaceCharSpeeds& data) {
  ASSERT(data.radius_factors.size() == data.min_speeds.size() and
             data.radius_factors.size() == data.max_speeds.size(),
         "Rescaled-surface factors and speed extrema have different sizes.");
  std::vector<double> row{time, static_cast<double>(data.status)};
  row.reserve(2 + 3 * data.radius_factors.size());
  for (size_t i = 0; i < data.radius_factors.size(); ++i) {
    row.push_back(data.radius_factors[i]);
    row.push_back(data.min_speeds[i]);
    row.push_back(data.max_speeds[i]);
  }
  return row;
}
}  // namespace ah
