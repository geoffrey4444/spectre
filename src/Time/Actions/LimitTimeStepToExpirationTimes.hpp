// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action AdvanceTime

#pragma once

#include <cmath>
#include <limits>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/FractionUtilities.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Evolution/EventsAndDenseTriggers/Tags.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Tags.hpp"

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
/// \brief Make sure step size does not exceed FunctionOfTime expiration times
///
/// \details For time steppers with substeps, functions of time might expire
/// at a time earlier than one of the substeps; however, the functions of time
/// are both needed at each substep and updated only after taking a full step.
/// This can result in quiescence, i.e., in waiting to complete the next
/// substep until the functions of time are updated, but the functions of time
/// will only be updated after the next step is taken. To avoid this situation,
/// at the start of each full step, this action checks whether the current
/// step size exceeds the time remaining before the first function of time
/// expires. If it does, then the step size is adjusted to be earlier than
/// the expiration time of the next function of time to expire. Note that
/// if the time stepper does not use substeps, this action does nothing.
///
/// Uses:
/// - DataBox:
///   - Tags::Time
///   - Tags::TimeStep
///   - Tags::TimeStepId
///   - Tags::TimeStepper<>
///   - domain::Tags::FunctionsOfTime
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Tags::TimeStep
struct LimitTimeStepToExpirationTimes {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {  // NOLINT const
    auto debug_print = [&box, &array_index](const std::string& message) {
      Parallel::printf(
          "LimitTimeStepToExpirationTimes.hpp: i=%s t=%1.20f tstep=%1.20f "
          "tsub=%1.20f "
          "dt=%1.20f next_trigger=%1.20f: %s\n",
          array_index, db::get<::Tags::Time>(box),
          db::get<::Tags::TimeStepId>(box).step_time().value(),
          db::get<::Tags::TimeStepId>(box).substep_time().value(),
          db::get<::Tags::TimeStep>(box).value(),
          db::get_mutable_reference<::evolution::Tags::EventsAndDenseTriggers>(
              make_not_null(&box))
              .next_trigger(box),
          message);
    };
    debug_print("start action"s);

    // First, check whether the time stepper uses substeps
    const TimeStepper& time_stepper = db::get<Tags::TimeStepper<>>(box);
    if (time_stepper.number_of_substeps() > 0) {
      // Next, check if the current time step is at the beginning of a full step
      const auto& time_step_id = db::get<Tags::TimeStepId>(box);
      if (time_step_id.substep() == 0) {
        // Get the functions of time and, if not empty, loop over them,
        // finding the minimum expiration time
        const auto& functions_of_time =
            get<domain::Tags::FunctionsOfTime>(cache);
        if (not functions_of_time.empty()) {
          double min_expiration_time{std::numeric_limits<double>::max()};
          for (const auto& [name, f_of_t] : functions_of_time) {
            if (f_of_t->time_bounds()[1] < min_expiration_time) {
              min_expiration_time = f_of_t->time_bounds()[1];
            }
          }

          // Is the minimum expiration time less than t + dt?
          const auto& initial_time_step{db::get<Tags::TimeStep>(box)};
          if (((db::get<Tags::TimeStepId>(box).step_time().value() +
                initial_time_step.value() - min_expiration_time) > 0) and
              (fabs(db::get<Tags::TimeStepId>(box).step_time().value() -
                    min_expiration_time) > 1.e-8)) {
            // Estimate fraction of slab that is min expiration time
            const double start{initial_time_step.slab().start().value()};
            // To avoid roundoff errors when checking if functions of time are
            // valid (i.e., if current time is greater than expiration time),
            // set the time step to by slightly less than the expiration time.
            // Here I choose 1e-8, a value much smaller than typical step sizes
            // but larger than any conceivable roundoff error.
            Slab new_slab{start, min_expiration_time};
            TimeDelta new_time_step{new_slab, Rational{1, 1}};

            TimeStepId new_time_step_id{
                time_step_id.time_runs_forward(), time_step_id.slab_number(),
                time_step_id.step_time().with_slab(new_slab),
                time_step_id.substep(),
                time_step_id.substep_time().with_slab(new_slab)};
            TimeStepId new_next_time_step_id =
                time_stepper.next_time_id(new_time_step_id, new_time_step);
            TimeDelta new_next_time_step{
                new_next_time_step_id.step_time().slab(),
                initial_time_step.fraction()};

            db::mutate<::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>,
                       ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>>(
                make_not_null(&box),
                [&new_time_step, &new_next_time_step, &new_time_step_id,
                 &new_next_time_step_id](
                    const gsl::not_null<TimeDelta*> time_step,
                    const gsl::not_null<TimeDelta*> next_time_step,
                    const gsl::not_null<TimeStepId*> time_step_id,
                    const gsl::not_null<TimeStepId*> next_time_step_id) {
                  *time_step = new_time_step;
                  *next_time_step = new_next_time_step;
                  *time_step_id = new_time_step_id;
                  *next_time_step_id = new_next_time_step_id;
                });
            debug_print("adjusted slab to match expiration time"s);
          }
        }
      }
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
