// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
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

  using argument_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTags>
  bool operator()(const db::DataBox<DbTags>& box) const {
    using metavariables =
        std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(box))>;
    const auto* cache =
        db::get<Parallel::Tags::GlobalCache<metavariables>>(box);
    const size_t success_count =
        Parallel::get<gh::bbh::Tags::CommonHorizonSuccessCount>(*cache);
    const size_t min_successes =
        Parallel::get<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks>(
            *cache);
    const size_t max_successes =
        Parallel::get<gh::bbh::Tags::MaxCommonHorizonSuccesses>(*cache);
    if (success_count < min_successes) {
      return false;
    }

    if (success_count >= max_successes) {
      return true;
    }

    return Parallel::get<gh::bbh::Tags::GaugeConstraintExceeded>(*cache) or
           Parallel::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(*cache) or
           Parallel::get<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(
               *cache);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;
};
}  // namespace gh::bbh::Triggers
