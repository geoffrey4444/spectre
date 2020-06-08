// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/Error.hpp"
#include "Time/Tags.hpp"


namespace Cce {
namespace InterfaceManagers{

enum class InterpolationStrategy { EveryStep, EverySubstep };

/*!
 * \brief Determines whether the element should interpolate for the current
 * state of its `box`, depending on the
 * `Cce::InterfaceManagers::InterpolationStrategy` used by the associated
 * `Cce::InterfaceManagers::GhInterfaceManager`.
 */
template <typename DbTagList>
bool should_interpolate_for_strategy(
    const db::DataBox<DbTagList>& box,
    const InterpolationStrategy strategy) noexcept {
  if(strategy == InterpolationStrategy::EverySubstep) {
    return true;
  }
  if(strategy == InterpolationStrategy::EveryStep) {
    return db::get<::Tags::TimeStepId>(box).substep() == 0;
  }
  ERROR("Interpolation strategy not recognized");
}
}  // namespace InterfaceManagers
}  // namespace Cce
