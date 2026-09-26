// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/RescaledSurfaceCharSpeeds.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace ah {
std::optional<std::vector<double>> rescaled_surface_factors(
    const DataVector& horizon_radius, const DataVector& excision_radius,
    const size_t number_of_surfaces, const double relative_margin) {
  if (number_of_surfaces < 2 or horizon_radius.size() == 0 or
      horizon_radius.size() != excision_radius.size() or
      not std::isfinite(relative_margin) or relative_margin < 0.0) {
    return std::nullopt;
  }
  double minimum_factor = 0.0;
  for (size_t p = 0; p < horizon_radius.size(); ++p) {
    if (not std::isfinite(horizon_radius[p]) or
        not std::isfinite(excision_radius[p]) or horizon_radius[p] <= 0.0 or
        excision_radius[p] <= 0.0 or excision_radius[p] >= horizon_radius[p]) {
      return std::nullopt;
    }
    minimum_factor =
        std::max(minimum_factor, excision_radius[p] / horizon_radius[p]);
  }
  minimum_factor *= 1.0 + relative_margin;
  if (minimum_factor >= 1.0) {
    return std::nullopt;
  }
  std::vector<double> factors(number_of_surfaces);
  for (size_t i = 0; i < number_of_surfaces; ++i) {
    const double fraction =
        static_cast<double>(i) / static_cast<double>(number_of_surfaces - 1);
    factors[i] = 1.0 - fraction * fraction * (1.0 - minimum_factor);
  }
  return factors;
}

std::pair<double, double> rescaled_surface_char_speed_extrema(
    const ylm::Strahlkorper<Frame::Distorted>& surface,
    const ylm::Strahlkorper<Frame::Distorted>& time_deriv_surface,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Distorted>& shift,
    const tnsr::II<DataVector, 3, Frame::Distorted>& inverse_spatial_metric) {
  [[maybe_unused]] const size_t number_of_points =
      surface.ylm_spherepack().physical_size();
  ASSERT(surface.l_max() == time_deriv_surface.l_max() and
             surface.m_max() == time_deriv_surface.m_max(),
         "The surface and its time derivative must use the same angular grid.");
  ASSERT(get(lapse).size() == number_of_points,
         "The lapse has " << get(lapse).size() << " points but the surface has "
                          << number_of_points << ".");

  const auto invalid_extrema =
      std::pair{std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
  const auto all_finite = [](const DataVector& data) {
    return std::all_of(data.begin(), data.end(),
                       [](const double value) { return std::isfinite(value); });
  };
  if (not all_finite(surface.coefficients()) or
      not all_finite(time_deriv_surface.coefficients()) or
      not all_finite(get(lapse))) {
    return invalid_extrema;
  }
  for (const auto& component : shift) {
    ASSERT(component.size() == number_of_points,
           "A shift component has " << component.size()
                                    << " points but the surface has "
                                    << number_of_points << ".");
    if (not all_finite(component)) {
      return invalid_extrema;
    }
  }
  for (const auto& component : inverse_spatial_metric) {
    ASSERT(component.size() == number_of_points,
           "An inverse spatial metric component has "
               << component.size() << " points but the surface has "
               << number_of_points << ".");
    if (not all_finite(component)) {
      return invalid_extrema;
    }
  }

  const auto radius = ylm::radius(surface);
  if (not std::all_of(get(radius).begin(), get(radius).end(),
                      [](const double value) {
                        return std::isfinite(value) and value > 0.0;
                      })) {
    return invalid_extrema;
  }
  const auto theta_phi = ylm::theta_phi(surface);
  const auto rhat = ylm::rhat(theta_phi);
  const auto normal = ylm::normal_one_form(
      ylm::cartesian_derivs_of_scalar(radius, surface, radius,
                                      ylm::inv_jacobian(theta_phi)),
      rhat);
  auto normal_magnitude = dot_product(normal, normal, inverse_spatial_metric);
  if (not std::all_of(get(normal_magnitude).begin(),
                      get(normal_magnitude).end(), [](const double value) {
                        return std::isfinite(value) and value > 0.0;
                      })) {
    return invalid_extrema;
  }
  get(normal_magnitude) = sqrt(get(normal_magnitude));
  const DataVector speeds =
      -get(lapse) +
      (get(dot_product(shift, normal)) +
       get(ylm::radius(time_deriv_surface)) * get(dot_product(rhat, normal))) /
          get(normal_magnitude);
  if (not all_finite(speeds)) {
    return invalid_extrema;
  }
  const auto extrema = std::minmax_element(speeds.begin(), speeds.end());
  return {*extrema.first, *extrema.second};
}
}  // namespace ah
