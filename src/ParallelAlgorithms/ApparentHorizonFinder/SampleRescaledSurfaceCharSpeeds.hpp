// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Domain;
/// \endcond

namespace ah {
/*!
 * \brief Prepare characteristic-speed measurements on scaled copies of a
 * converged distorted-frame horizon.
 *
 * The horizon and excision sphere must have the same fixed expansion center.
 * The grid-to-distorted map must preserve angles about that center. Physical
 * frames coincide without explicit maps in stationary domains. In a
 * time-dependent domain, the excision surface must lie in blocks that have a
 * distorted frame.
 *
 * The derivative is taken at fixed scale factor, even though the factors are
 * chosen again at every observation. Missing derivatives and invalid geometry
 * produce a status and NaN speeds, not physical zero speeds.
 */
void initialize_rescaled_surface_char_speeds(
    gsl::not_null<Storage::RescaledSurfaceCharSpeeds*> data,
    const ylm::Strahlkorper<Frame::Distorted>& horizon,
    const ylm::Strahlkorper<Frame::Distorted>& time_deriv_horizon,
    bool time_derivative_is_available,
    const RescaledSurfaceCharSpeedOptions& options, const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time, double time);

/*!
 * \brief Sample all surfaces for which volume data have arrived.
 *
 * Returns true when the diagnostic is complete, including a diagnostic failure
 * recorded in the status. Returns false if more volume data at this same time
 * are required. The caller must retain the data and resume this function on
 * subsequent arrivals. Interpolation buffers belong to the diagnostic and do
 * not modify the converged horizon or its intersection history.
 */
bool sample_rescaled_surface_char_speeds(
    gsl::not_null<Storage::RescaledSurfaceCharSpeeds*> data,
    const std::unordered_map<ElementId<3>,
                             Storage::VolumeVariables<Frame::Distorted>>&
        volume_variables,
    const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time, double time,
    const std::unordered_set<std::string>& blocks_for_interpolation);

/// Column names for time, status, and each surface's factor and speed extrema.
std::vector<std::string> rescaled_surface_char_speed_legend(
    size_t number_of_surfaces);

/// One row matching rescaled_surface_char_speed_legend.
std::vector<double> rescaled_surface_char_speed_row(
    double time, const Storage::RescaledSurfaceCharSpeeds& data);
}  // namespace ah
