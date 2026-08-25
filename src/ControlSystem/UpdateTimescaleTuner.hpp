// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <string>

#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

namespace control_system {
/*!
 * \brief Updates the TimescaleTuner with information from the ControlError, if
 * possible.
 *
 * \details We check for a suggested timescale from the ControlError. If one is
 * suggested and it is smaller than the current damping timescale, we set the
 * timescale in the TimescaleTuner to this suggested value. Otherwise, we let
 * the TimescaleTuner adjust the timescale. Regardless of whether a timescale
 * was suggested or not, we always reset the control error.
 */
template <bool AllowDecrease, typename ControlError>
void update_timescale_tuner(
    const gsl::not_null<TimescaleTuner<AllowDecrease>*> tuner,
    const gsl::not_null<ControlError*> control_error, ::Verbosity verbosity,
    const double time, const std::string& function_of_time_name) {
  static_assert(
      tt::assert_conforms_to_v<ControlError, protocols::ControlError>);

  const std::optional<double>& suggested_timescale =
      control_error->get_suggested_timescale();
  const double old_timescale = min(tuner->current_timescale());

  if (suggested_timescale.value_or(std::numeric_limits<double>::infinity()) <
      old_timescale) {
    tuner->set_timescale_if_in_allowable_range(suggested_timescale.value());
  }

  if (verbosity >= ::Verbosity::Verbose) {
    using ::operator<<;
    Parallel::printf(
        "%s, time = %.16f:\n"
        " old_timescale = %.16f\n"
        " suggested_timescale = %s\n"
        " new_timescale = %.16f\n",
        function_of_time_name, time, old_timescale,
        MakeString{} << suggested_timescale, min(tuner->current_timescale()));
  }

  // The control error is reset only after the suggested timescale has been
  // consumed. Size control acknowledges discontinuous changes separately when
  // it repopulates the Averager so that the suggestion survives until here.
  control_error->reset();
}
}  // namespace control_system
