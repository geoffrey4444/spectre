// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::bbh::Triggers {
class CompletionCriteria : public Trigger {
 public:
  /// \cond
  CompletionCriteria() = default;
  explicit CompletionCriteria(CkMigrateMessage* /*msg*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CompletionCriteria);  // NOLINT
  /// \endcond

  static constexpr Options::String help{
      "Trigger completion for BBH inspirals based on mutable GlobalCache "
      "criteria latched by horizon and constraint-check events."};
  static std::string name() { return "BbhCompletionCriteria"; }
  using options = tmpl::list<>;

  using argument_tags = tmpl::list<::Tags::TimeStepId, ::Tags::DataBox>;

  template <typename DbTags>
  bool operator()(const TimeStepId& time_step_id,
                  const db::DataBox<DbTags>& box) const {
    using metavariables =
        std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(box))>;
    auto* cache = db::get<Parallel::Tags::GlobalCache<metavariables>>(box);
    const size_t success_count =
        Parallel::get<gh::bbh::Tags::CommonHorizonSuccessCount>(*cache);
    const size_t min_successes =
        Parallel::get<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks>(
            *cache);
    const size_t max_successes =
        Parallel::get<gh::bbh::Tags::MaxCommonHorizonSuccesses>(*cache);
    ASSERT(max_successes >= min_successes,
           "MaxCommonHorizonSuccesses ("
               << max_successes << ") must be >= "
               << "MinCommonHorizonSuccessesBeforeChecks (" << min_successes
               << ").");
    if (success_count < min_successes) {
      return false;
    }

    const bool criteria_met =
        success_count >= max_successes or
        Parallel::get<gh::bbh::Tags::GaugeConstraintExceeded>(*cache) or
        Parallel::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(*cache) or
        Parallel::get<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(
            *cache) or
        Parallel::get<gh::bbh::Tags::CompletionRequested>(*cache);
    if (criteria_met and Parallel::get<gh::bbh::Tags::StopSlabNumber>(*cache) ==
                             std::numeric_limits<size_t>::max()) {
      if (time_step_id.slab_number() >= 0) {
        // Latch two slabs ahead so all nodes have time to receive mutable
        // cache updates before the Completion trigger evaluates true.
        Parallel::mutate<gh::bbh::Tags::StopSlabNumber,
                         gh::bbh::Mutators::SetStopSlabNumberIfUnset>(
            *cache, static_cast<size_t>(time_step_id.slab_number() + 2));
        Parallel::printf(
            "BBH completion stop slab latched at slab %zu (current slab %lld)."
            "\n",
            Parallel::get<gh::bbh::Tags::StopSlabNumber>(*cache),
            static_cast<long long>(time_step_id.slab_number()));
      }
    }

    const auto stop_slab_number =
        Parallel::get<gh::bbh::Tags::StopSlabNumber>(*cache);
    return stop_slab_number != std::numeric_limits<size_t>::max() and
           static_cast<size_t>(time_step_id.slab_number()) >= stop_slab_number;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;
};
}  // namespace gh::bbh::Triggers
