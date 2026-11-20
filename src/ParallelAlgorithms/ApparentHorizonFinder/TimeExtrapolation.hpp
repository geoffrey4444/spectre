// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <optional>

class DataVector;
namespace ylm {
class Spherepack;
template <typename Fr>
class Strahlkorper;
}  // namespace ylm
namespace ah::Storage {
template <typename Fr>
struct PreviousSurface;
}  // namespace ah::Storage

namespace ah {
inline constexpr size_t max_number_of_extrapolation_points = 4;

/// RMS coefficient errors for the two-, three-, and four-point predictors.
/// An empty entry means there were too few surfaces to score that predictor.
using PredictionErrors =
    std::array<std::optional<double>, max_number_of_extrapolation_points - 1>;

/// Select the number of stored surfaces to use at the next time. In adaptive
/// mode (no configured point count), choose the smallest measured error, with
/// ties going to fewer points. Until there is a score, use all available
/// points up to the maximum.
size_t number_of_points_for_extrapolation(
    std::optional<size_t> configured_points, size_t available,
    const PredictionErrors& errors);

/// Extrapolate the coefficients of the newest `number_of_points` surfaces to
/// `target_time`, expressed in `target_basis`. History is newest first and
/// all surfaces must have the same expansion center.
template <typename Fr>
DataVector extrapolate_surface_coefficients(
    double target_time, size_t number_of_points,
    const std::deque<Storage::PreviousSurface<Fr>>& previous_surfaces,
    const ylm::Spherepack& target_basis);

/// Score the predictions of the newly found surface using only older
/// surfaces. All predictions and the found surface are compared at the
/// resolution of the most recent older surface.
template <typename Fr>
PredictionErrors prediction_errors(
    double target_time, const ylm::Strahlkorper<Fr>& found,
    const std::deque<Storage::PreviousSurface<Fr>>& previous_surfaces);
}  // namespace ah
