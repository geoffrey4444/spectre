// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/MathWrapper.hpp"
#include "DataStructures/TaggedVariant.hpp"
#include "DataStructures/Variables.hpp"
#include "Time/History.hpp"
#include "Time/LargestStepperError.hpp"
#include "Time/LtsMode.hpp"
#include "Time/SelfStart.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/LtsMode.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
/// Raw, complete predictor before filtering, together with its step start.
template <typename VariablesTag>
struct UnfilteredLtsPredictor : db::SimpleTag {
  using type = std::optional<std::pair<::Time, typename VariablesTag::type>>;
};
}  // namespace Tags

/// Experimental full PC defect for fixed-order conservative monotonic AMPC.
/// Run after ApplyLtsBoundaryCorrections and before filtering. Other steppers,
/// variable-order stepping and self-start retain their existing behavior.
template <typename System>
struct UpdateLtsStepperError {
  using variables_tag = typename System::variables_tag;
  using variables_type = typename variables_tag::type;
  using predictor_tag = Tags::UnfilteredLtsPredictor<variables_tag>;
  using errors_type = typename Tags::StepperErrors<variables_tag>::type;
  using simple_tags = tmpl::list<predictor_tag>;
  using return_tags =
      tmpl::list<predictor_tag, Tags::StepperErrors<variables_tag>>;
  using argument_tags =
      tmpl::list<Tags::TimeStepper<TimeStepper>, Tags::LtsMode,
                 Tags::TimeStepId, Tags::Next<Tags::TimeStepId>,
                 Tags::StepperErrorEstimatesEnabled,
                 Tags::HistoryEvolvedVariables<variables_tag>,
                 Tags::StepperErrorTolerances<variables_tag>, variables_tag>;

  static void apply(
      const gsl::not_null<typename predictor_tag::type*> predictor,
      const gsl::not_null<errors_type*> errors, const TimeStepper& stepper,
      const LtsMode lts_mode, const TimeStepId& time_id,
      const TimeStepId& next_time_id, const bool error_estimates_enabled,
      const TimeSteppers::History<variables_type>& history,
      const StepperErrorTolerances& tolerances, const variables_type& vars) {
    if (lts_mode != LtsMode::Conservative or not error_estimates_enabled or
        SelfStart::is_self_starting(time_id) or
        SelfStart::step_unused(time_id, next_time_id) or
        dynamic_cast<const TimeSteppers::AdamsMoultonPc<true>*>(&stepper) ==
            nullptr or
        not variants::holds_alternative<TimeSteppers::Tags::FixedOrder>(
            stepper.order()) or
        tolerances.estimates == StepperErrorTolerances::Estimates::None) {
      predictor->reset();
      return;
    }
    if (time_id.substep() == 0) {
      if (predictor->has_value()) {
        predictor->value().first = time_id.step_time();
        predictor->value().second = vars;
      } else {
        predictor->emplace(time_id.step_time(), vars);
      }
      return;
    }
    ASSERT(time_id.substep() == 1, "Expected AMPC corrector");
    ASSERT(predictor->has_value() and
               predictor->value().first == time_id.step_time(),
           "Full LTS error requires the matching unfiltered predictor");
    ASSERT((*errors)[1].has_value() and
               (*errors)[1]->step_time == time_id.step_time(),
           "Volume error estimate must exist for the current corrector");
    const auto current = make_math_wrapper(vars);
    const auto predicted = make_math_wrapper(predictor->value().second);
    const std::remove_const_t<typename decltype(current)::value_type> defect{
        *current - *predicted};
    const auto start = make_math_wrapper(*history.back().value);
    const double complete_error =
        largest_stepper_error(*start, defect, tolerances);
    auto& primary = gsl::at((*errors)[1]->errors, (*errors)[1]->order);
    // Retain the existing volume estimate as an additional conservative bound.
    primary = std::max(primary.value(), complete_error);
    // Keep allocated storage; the next predictor replaces its time and data.
  }
};
