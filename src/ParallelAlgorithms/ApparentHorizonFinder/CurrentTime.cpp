// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/CurrentTime.hpp"

#include <optional>
#include <set>
#include <sstream>

#include "DataStructures/LinkedMessageId.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Gsl.hpp"

namespace ah {
void set_current_time(
    const gsl::not_null<std::optional<LinkedMessageId<double>>*> current_time,
    const gsl::not_null<std::set<LinkedMessageId<double>>*> pending_times,
    const std::set<LinkedMessageId<double>>& completed_times,
    const ::Verbosity& verbosity, const std::string& name) {
  std::stringstream ss{};
  const bool verbose_print = verbosity >= ::Verbosity::Verbose;
  const bool debug_print = verbosity >= ::Verbosity::Debug;
  if (debug_print) {
    ss << name << ": ";
  }

  // If there's already a horizon find in progress, end here
  if (current_time->has_value() or pending_times->empty()) {
    if (debug_print) {
      if (current_time->has_value()) {
        ss << "Interpolation already in progress at time "
           << current_time->value();
      } else {
        ss << "No pending times.";
      }
      Parallel::printf("%s\n", ss.str());
    }

    return;
  }

  // To determine if we are going to use the next pending time, we need to check
  // if we have any completed times so we can check if this is the very first
  // time or not.
  const auto& next_pending_time = *pending_times->begin();
  const bool use_next_pending_time =
      UNLIKELY(completed_times.empty())
          ? not next_pending_time.previous.has_value()
          : next_pending_time.previous.value() ==
                std::prev(completed_times.end())->id;

  // If we are using the next pending time:
  // 1. Set the current time to this next pending time
  // 2. Remove this time from pending
  if (use_next_pending_time) {
    (*current_time) = next_pending_time;
    pending_times->erase(pending_times->begin());

    if (verbose_print) {
      ss << "Setting current time to " << *current_time;
    }
  } else if (verbose_print) {
    ss << "Not setting current time.";
  }

  if (verbose_print) {
    Parallel::printf("%s\n", ss.str());
  }
}
}  // namespace ah
