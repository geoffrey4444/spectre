// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action AdvanceTime

#pragma once

#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
template <typename Tag>
struct Next;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Advance time one substep
///
/// Uses:
/// - DataBox:
///   - Tags::TimeStep
///   - Tags::TimeStepId
///   - Tags::TimeStepper<>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Tags::Next<Tags::TimeStepId>
///   - Tags::Time
///   - Tags::TimeStepId
///   - Tags::TimeStep
struct AdvanceTime {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {  // NOLINT const
    auto debug_print = [&box, &array_index,
                        &cache](const std::string& message) {
      double min_expiration_time{std::numeric_limits<double>::max()};
      const auto& functions_of_time = get<domain::Tags::FunctionsOfTime>(cache);
      if (not functions_of_time.empty()) {
        for (const auto& [name, f_of_t] : functions_of_time) {
          if (f_of_t->time_bounds()[1] < min_expiration_time) {
            min_expiration_time = f_of_t->time_bounds()[1];
          }
        }
      }
      Parallel::printf(
          "AdvanceTime: i=%s texp=%1.20f t=%1.20f tstep=%1.20f tsub=%1.20f "
          "dt=%1.20f next_trigger=%1.20f: %s\n",
          array_index, min_expiration_time, db::get<::Tags::Time>(box),
          db::get<::Tags::TimeStepId>(box).step_time().value(),
          db::get<::Tags::TimeStepId>(box).substep_time().value(),
          db::get<::Tags::TimeStep>(box).value(),
          db::get_mutable_reference<::evolution::Tags::EventsAndDenseTriggers>(
              make_not_null(&box))
              .next_trigger(box),
          message);
    };
    debug_print("Before"s);
    db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
               Tags::Time, Tags::Next<Tags::TimeStep>>(
        make_not_null(&box),
        [](const gsl::not_null<TimeStepId*> time_id,
           const gsl::not_null<TimeStepId*> next_time_id,
           const gsl::not_null<TimeDelta*> time_step,
           const gsl::not_null<double*> time,
           const gsl::not_null<TimeDelta*> next_time_step,
           const TimeStepper& time_stepper) {
          *time_id = *next_time_id;
          *time_step = next_time_step->with_slab(time_id->step_time().slab());

          *next_time_id = time_stepper.next_time_id(*next_time_id, *time_step);
          *next_time_step =
              time_step->with_slab(next_time_id->step_time().slab());
          *time = time_id->substep_time().value();
        },
        db::get<Tags::TimeStepper<>>(box));
    debug_print("After"s);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
