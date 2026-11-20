// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>

#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/Gsl.hpp"

template <size_t VolumeDim>
class Domain;
class FastFlow;
namespace ylm {
template <typename Frame>
class Strahlkorper;
}  // namespace ylm

namespace ah {
/*!
 * \brief Compute the target points for the current iteration.
 *
 * \details Returns whether the computation of the target points was successful
 * or if there are some points outside the domain. If computation fails, one of
 * two attempts to recover will happen:
 *
 * - For the zeroth fast flow iteration, this will increase the $l=0,m=0$
 *   coefficient (i.e. the size) by 50%.
 * - For all other iterations, the coefficients become
 * \begin{equation}
 * S^{\mathrm{new}}_{lm} = \frac{1}{2}\left(S^{\mathrm{previous}}_{lm} +
 * S^{\mathrm{failed}}_{lm}\right)
 * \end{equation}
 *   where $S^{\mathrm{failed}}_{lm}$ are the coefficients of the failed
 *   computation and $S^{\mathrm{previous}}_{lm}$ are the coefficients from
 *   the previous successful iteration.
 *
 * This function will try recomputing the coords using the above two rules
 * \p max_compute_coords_retries times before returning false.
 *
 * \param current_iteration The returned pointer to the current Iteration object
 * \param block_order Priority order to search blocks for containing points
 * (see `::block_logical_coordinates` for details)
 * \param time The current time
 * \param fast_flow The FastFlow object for the current horizon find
 * \param initial_guess Trial surface for the first find.
 * \param previous_iteration_surface Previous trial surface at this time,
 * used for coordinate retries and reruns at higher resolution.
 * \param previous_surfaces Converged horizons at earlier times, newest first.
 * Their expansion centers are assumed to coincide in the finder frame.
 * \param max_compute_coords_retries Retry up to this many times before
 * returning false
 * \param domain The spatial domain in which the horizon is being found
 * \param functions_of_time The functions of time for the current domain
 * \param current_resolution_l Optional; if specified, current_iteration is
 * prolonged or restricted to this resolution
 * \param rerunning_with_higher_resolution Must be false unless
 * current_resolution_l is set; if true, then on iteration zero, set
 * the initial guess to the previous iteration surface
 * \param max_extrapolation_points Maximum number of previous surfaces to use
 * for the initial guess at a new time. A single surface is reused directly.
 * \returns Whether or not set_current_iteration_coords succeeded
 */
template <typename Fr>
bool set_current_iteration_coords(
    gsl::not_null<ah::Storage::Iteration<Fr>*> current_iteration,
    gsl::not_null<std::vector<size_t>*> block_order,
    const LinkedMessageId<double>& time, const FastFlow& fast_flow,
    const ylm::Strahlkorper<Fr>& initial_guess,
    const ylm::Strahlkorper<Fr>& previous_iteration_surface,
    const std::deque<ah::Storage::PreviousSurface<Fr>>& previous_surfaces,
    size_t max_compute_coords_retries, const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time,
    const std::optional<size_t>& current_resolution_l = std::nullopt,
    bool rerunning_with_higher_resolution = false,
    size_t max_extrapolation_points = 3);
}  // namespace ah
