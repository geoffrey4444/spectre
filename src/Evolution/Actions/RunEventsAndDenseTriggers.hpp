// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "Evolution/EventsAndDenseTriggers/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct TimeStep;
}  // namespace Tags
/// \endcond

namespace evolution::Actions {
/// \ingroup ActionsGroup
/// \ingroup EventsAndTriggersGroup
/// \brief Run the events and dense triggers
///
/// Uses:
/// - DataBox: EventsAndDenseTriggers, as required by events and triggers
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
template <typename PrimFromCon = void>
struct RunEventsAndDenseTriggers {
 private:
  // RAII object to restore the time and variables changed by dense
  // output.
  template <typename DbTags, typename Tag>
  class StateRestorer {
   public:
    StateRestorer(const gsl::not_null<db::DataBox<DbTags>*> box) : box_(box) {}

    void save() {
      // Only store the value the first time, because after that we
      // are seeing the value after the previous change instead of the
      // original.
      if (not value_.has_value()) {
        value_ = db::get<Tag>(*box_);
      }
    }

    ~StateRestorer() {
      if (value_.has_value()) {
        db::mutate<Tag>(box_,
                        [this](const gsl::not_null<typename Tag::type*> value) {
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 11
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif  // defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 11
                          *value = *value_;
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 11
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 11
                        });
      }
    }

   private:
    gsl::not_null<db::DataBox<DbTags>*> box_ = nullptr;
    std::optional<typename Tag::type> value_{};
  };

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const component) {
    auto debug_print = [&box, &array_index](const std::string& message) {
      Parallel::printf(
          "RunEventsAndTriggers: i=%s t=%1.20f tstep=%1.20f tsub=%1.20f "
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
    debug_print("start of action"s);

    using system = typename Metavariables::system;
    using variables_tag = typename system::variables_tag;

    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    if (time_step_id.slab_number() < 0) {
      // Skip dense output during self-start
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    auto& events_and_dense_triggers =
        db::get_mutable_reference<::evolution::Tags::EventsAndDenseTriggers>(
            make_not_null(&box));

    const auto step_end =
        time_step_id.step_time() + db::get<::Tags::TimeStep>(box);
    const evolution_less<double> before{time_step_id.time_runs_forward()};

    StateRestorer<DbTags, ::Tags::Time> time_restorer(make_not_null(&box));
    StateRestorer<DbTags, variables_tag> variables_restorer(
        make_not_null(&box));
    auto primitives_restorer = [&box]() {
      if constexpr (system::has_primitive_and_conservative_vars) {
        return StateRestorer<DbTags, typename system::primitive_variables_tag>(
            make_not_null(&box));
      } else {
        (void)box;
        return 0;
      }
    }();
    (void)primitives_restorer;
    for (;;) {
      const double next_trigger = events_and_dense_triggers.next_trigger(box);
      debug_print("start of for loop"s);
      if (before(step_end.value(), next_trigger)) {
        debug_print("continue"s);
        return {Parallel::AlgorithmExecution::Continue, std::nullopt};
      }

      // This can only be true the first time through the loop,
      // because triggers are not allowed to reschedule for the time
      // they just triggered at.  This check is primarily to avoid
      // special-case bookkeeping for the initial simulation time.
      const bool already_at_correct_time =
          db::get<::Tags::Time>(box) == next_trigger;
      if (not already_at_correct_time) {
        time_restorer.save();
        db::mutate<::Tags::Time>(
            make_not_null(&box),
            [&next_trigger](const gsl::not_null<double*> time) {
              *time = next_trigger;
            });
      }

      const auto triggered = events_and_dense_triggers.is_ready(
          box, cache, array_index, component);
      using TriggeringState = std::decay_t<decltype(triggered)>;
      switch (triggered) {
        case TriggeringState::NotReady:
          debug_print("not ready, retry"s);
          return {Parallel::AlgorithmExecution::Retry, std::nullopt};
        case TriggeringState::NeedsEvolvedVariables:
          if (not already_at_correct_time) {
            if constexpr (Metavariables::local_time_stepping) {
              if (not dg::receive_boundary_data_local_time_stepping<
                      Metavariables, true>(make_not_null(&box),
                                           make_not_null(&inboxes))) {
                debug_print("local time stepping needs boundary data, retry"s);
                return {Parallel::AlgorithmExecution::Retry, std::nullopt};
              }
            }

            using history_tag = ::Tags::HistoryEvolvedVariables<variables_tag>;
            bool dense_output_succeeded = false;
            variables_restorer.save();
            db::mutate<variables_tag>(
                make_not_null(&box),
                [&dense_output_succeeded, &next_trigger](
                    gsl::not_null<typename variables_tag::type*> vars,
                    const TimeStepper& stepper,
                    const typename history_tag::type& history) {
                  dense_output_succeeded =
                      stepper.dense_update_u(vars, history, next_trigger);
                },
                db::get<::Tags::TimeStepper<>>(box), db::get<history_tag>(box));
            if (not dense_output_succeeded) {
              // Need to take another time step
              debug_print("dense output failed, continue"s);
              return {Parallel::AlgorithmExecution::Continue, std::nullopt};
            }

            if constexpr (Metavariables::local_time_stepping) {
              dg::apply_boundary_corrections<system, true, true>(
                  make_not_null(&box));
            }

            static_assert(system::has_primitive_and_conservative_vars !=
                              std::is_same_v<PrimFromCon, void>,
                          "Primitive update scheme not provided.");
            if constexpr (system::has_primitive_and_conservative_vars) {
              primitives_restorer.save();
              db::mutate_apply<PrimFromCon>(make_not_null(&box));
            }
          }
          [[fallthrough]];
        default:
          break;
      }

      debug_print("running events"s);
      events_and_dense_triggers.run_events(box, cache, array_index, component);
      if (not events_and_dense_triggers.reschedule(box, cache, array_index,
                                                   component)) {
        debug_print("could not reschedule, retry"s);
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }
    }
  }
};

struct InitializeRunEventsAndDenseTriggers {
  using initialization_tags =
      tmpl::list<evolution::Tags::EventsAndDenseTriggers>;
  using initialization_tags_to_keep = initialization_tags;
  using simple_tags = tmpl::list<Tags::PreviousTriggerTime>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*component*/) {
    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               std::nullopt);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::Actions
