// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
namespace ylm {
template <typename Fr>
class Strahlkorper;
}  // namespace ylm
/// \endcond

namespace ah {
/*!
 * \brief Factors for a family of rescaled horizons outside an excision surface.
 *
 * The input radii must be evaluated at the same angular points about a common
 * center. The smallest factor is
 * \f$q_{\min}=(1+\epsilon)\max(R_{\mathrm{ex}}/R_{\mathrm{AH}})\f$,
 * where \f$\epsilon\f$ is `relative_margin`. The returned factors are
 * \f$q_i=1-(i/(N-1))^2(1-q_{\min})\f$ for \f$i=0,\ldots,N-1\f$.
 * Returns `std::nullopt` for nonpositive or nonfinite radii, incompatible or
 * empty input sizes, fewer than two surfaces, negative or nonfinite margin,
 * or no positive gap between the smallest surface and the horizon.
 */
std::optional<std::vector<double>> rescaled_surface_factors(
    const DataVector& horizon_radius, const DataVector& excision_radius,
    size_t number_of_surfaces, double relative_margin);

/*!
 * \brief Angular extrema of the excision-sign characteristic speed on a moving
 * surface with a fixed center in the distorted frame.
 *
 * Computes \f$c=-\alpha+s_i(\beta^i+\dot{R}\hat{r}^i)\f$, where \f$s_i\f$
 * is the outward unit normal normalized with `inverse_spatial_metric`.
 * `shift` is the coordinate shift in the distorted frame, including the
 * distorted-to-inertial map velocity but not the grid-to-distorted velocity.
 * The time derivative supplies only the radial velocity. To diagnose a
 * rescaled horizon, scale both the horizon and its time derivative by the
 * same factor before calling this function; the factor is held fixed when
 * taking the time derivative.
 *
 * All tensors and both Strahlkorpers must use the same angular collocation
 * grid. Returns a pair of NaNs if any characteristic speed is nonfinite.
 */
std::pair<double, double> rescaled_surface_char_speed_extrema(
    const ylm::Strahlkorper<Frame::Distorted>& surface,
    const ylm::Strahlkorper<Frame::Distorted>& time_deriv_surface,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Distorted>& shift,
    const tnsr::II<DataVector, 3, Frame::Distorted>& inverse_spatial_metric);
}  // namespace ah
