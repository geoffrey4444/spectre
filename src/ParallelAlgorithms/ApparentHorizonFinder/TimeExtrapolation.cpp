// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/TimeExtrapolation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "NumericalAlgorithms/Interpolation/LagrangePolynomial.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ah {
size_t number_of_points_for_extrapolation(
    const std::optional<size_t> configured_points, const size_t available,
    const PredictionErrors& errors) {
  if (configured_points.has_value()) {
    return std::min(*configured_points, available);
  }
  const size_t max_points =
      std::min(available, max_number_of_extrapolation_points);
  size_t best_points = max_points;
  double smallest_error = std::numeric_limits<double>::infinity();
  for (size_t points = 2; points <= max_points; ++points) {
    const auto& error = gsl::at(errors, points - 2);
    if (error.has_value() and *error < smallest_error) {
      best_points = points;
      smallest_error = *error;
    }
  }
  return best_points;
}

template <typename Fr>
DataVector extrapolate_surface_coefficients(
    const double target_time, const size_t number_of_points,
    const std::deque<Storage::PreviousSurface<Fr>>& previous_surfaces,
    const ylm::Spherepack& target_basis) {
  ASSERT(number_of_points > 0 and
             number_of_points <= max_number_of_extrapolation_points and
             number_of_points <= previous_surfaces.size(),
         "Cannot extrapolate with " << number_of_points << " points from "
                                    << previous_surfaces.size() << " surfaces");
  std::array<double, max_number_of_extrapolation_points> time_storage{};
  const auto times = gsl::make_span(time_storage.data(), number_of_points);
  for (size_t i = 0; i < number_of_points; ++i) {
    times[i] = previous_surfaces[i].time.id;
  }

  DataVector result(target_basis.spectral_size(), 0.);
  // The Lagrange weights sum to one, so infer the last weight as in the
  // original linear and quadratic predictors.
  double remaining_weight = 1.;
  for (size_t i = 0; i < number_of_points; ++i) {
    const double weight =
        i + 1 == number_of_points
            ? remaining_weight
            : lagrange_polynomial(i, target_time, times.begin(), times.end());
    remaining_weight -= weight;
    const auto& surface = previous_surfaces[i].surface;
    const auto& basis = surface.ylm_spherepack();
    if (basis == target_basis) {
      result += weight * surface.coefficients();
    } else {
      result += weight *
                basis.prolong_or_restrict(surface.coefficients(), target_basis);
    }
  }
  return result;
}

template <typename Fr>
PredictionErrors prediction_errors(
    const double target_time, const ylm::Strahlkorper<Fr>& found,
    const std::deque<Storage::PreviousSurface<Fr>>& previous_surfaces) {
  PredictionErrors errors{};
  if (previous_surfaces.size() < 2) {
    return errors;
  }
  const auto& basis = previous_surfaces.front().surface.ylm_spherepack();
  const auto found_coefficients =
      found.ylm_spherepack().prolong_or_restrict(found.coefficients(), basis);
  const size_t max_points =
      std::min(previous_surfaces.size(), max_number_of_extrapolation_points);
  ylm::SpherepackIterator mode(basis.l_max(), basis.m_max());
  for (size_t points = 2; points <= max_points; ++points) {
    const auto predicted = extrapolate_surface_coefficients(
        target_time, points, previous_surfaces, basis);
    // The SPHEREPACK buffer also contains padding, which is not part of the
    // surface and must not affect the measured errors.
    double squared_error = 0.;
    for (mode.reset(); mode; ++mode) {
      squared_error += square(predicted[mode()] - found_coefficients[mode()]);
    }
    gsl::at(errors, points - 2) =
        sqrt(squared_error / static_cast<double>(mode.current_compact_index()));
  }
  return errors;
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                    \
  template DataVector extrapolate_surface_coefficients(         \
      double, size_t,                                           \
      const std::deque<Storage::PreviousSurface<FRAME(data)>>&, \
      const ylm::Spherepack&);                                  \
  template PredictionErrors prediction_errors(                  \
      double, const ylm::Strahlkorper<FRAME(data)>&,            \
      const std::deque<Storage::PreviousSurface<FRAME(data)>>&);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Inertial, Frame::Distorted, Frame::Grid))

#undef INSTANTIATE
#undef FRAME
}  // namespace ah
