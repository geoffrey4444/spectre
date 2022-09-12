// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Callback.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/StdHelpers.hpp"

#include "Evolution/EventsAndDenseTriggers/Tags.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Tags.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace domain {
/// \ingroup ComputationalDomainGroup
/// Check that functions of time are up-to-date.
///
/// Check that functions of time in \p CacheTag with names in \p
/// functions_to_check are ready at time \p time.  If no names are
/// listed in \p functions_to_check, checks all functions in \p
/// CacheTag.  If any function is not ready, schedules a
/// `Parallel::PerformAlgorithmCallback` with the GlobalCache.
template <typename CacheTag, typename Metavariables, typename ArrayIndex,
          typename Component, size_t N = 0>
bool functions_of_time_are_ready(
    Parallel::GlobalCache<Metavariables>& cache, const ArrayIndex& array_index,
    const Component* /*meta*/, const double time,
    const std::array<std::string, N>& functions_to_check =
        std::array<std::string, 0>{}) {
  const auto& proxy =
      ::Parallel::get_parallel_component<Component>(cache)[array_index];

  return Parallel::mutable_cache_item_is_ready<CacheTag>(
      cache, [&functions_to_check, &proxy,
              &time](const std::unordered_map<
                     std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                         functions_of_time) {
        using ::operator<<;
        ASSERT(alg::all_of(
                   functions_to_check,
                   [&functions_of_time](const std::string& function_to_check) {
                     return functions_of_time.count(function_to_check) == 1;
                   }),
               "Not all functions to check ("
                   << functions_to_check << ") are in the global cache ("
                   << keys_of(functions_of_time) << ")");
        for (const auto& [name, f_of_t] : functions_of_time) {
          if ((not functions_to_check.empty()) and
              alg::find(functions_to_check, name) == functions_to_check.end()) {
            continue;
          }
          const double expiration_time = f_of_t->time_bounds()[1];
          Parallel::printf(
              "functions_of_time_are_ready: t=%1.20f texp=%1.20f f_of_t %s: "
              "%s\n",
              time, expiration_time, name,
              time > expiration_time ? "is not ready"s : "is ready"s);
          if (time > expiration_time) {
            return std::unique_ptr<Parallel::Callback>(
                new Parallel::PerformAlgorithmCallback(proxy));
          }
        }
        return std::unique_ptr<Parallel::Callback>{};
      });
}

namespace Actions {
/// \ingroup ComputationalDomainGroup
/// Check that functions of time are up-to-date.
///
/// Wait for all functions of time in `domain::Tags::FunctionsOfTime`
/// to be ready at `::Tags::Time`.  This ensures that the coordinates
/// can be safely accessed in later actions without first verifying
/// the state of the time-dependent maps.
struct CheckFunctionsOfTimeAreReady {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* component) {
    const bool ready =
        functions_of_time_are_ready<domain::Tags::FunctionsOfTime>(
            cache, array_index, component, db::get<::Tags::Time>(box));
    auto debug_print = [&box, &array_index](const std::string& message) {
      Parallel::printf(
          "CheckFunctionsOfTimeAreReady: i=%s t=%1.15f tstep=%1.15f "
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
    debug_print(ready ? "Ready"s : "Not ready"s);
    return {ready ? Parallel::AlgorithmExecution::Continue
                  : Parallel::AlgorithmExecution::Retry,
            std::nullopt};
  }
};
}  // namespace Actions
}  // namespace domain
