// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <pup.h>
#include <string>

#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::bbh::Triggers {
class CompletionCriteria : public DenseTrigger {
 public:
  /// \cond
  CompletionCriteria() = default;
  explicit CompletionCriteria(CkMigrateMessage* msg) : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CompletionCriteria);  // NOLINT
  /// \endcond

  static constexpr Options::String help{
      "Trigger completion for BBH inspirals based on mutable GlobalCache "
      "criteria latched by horizon and constraint-check events."};
  static std::string name() { return "BbhCompletionCriteria"; }
  using options = tmpl::list<>;

  using is_triggered_return_tags = tmpl::list<>;
  using is_triggered_argument_tags = tmpl::list<::Tags::Time>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<bool> is_triggered(Parallel::GlobalCache<Metavariables>& cache,
                                   const ArrayIndex& /*array_index*/,
                                   const Component* /*component*/,
                                   const double time) const {
    const size_t success_count =
        Parallel::get<gh::bbh::Tags::CommonHorizonSuccessCount>(cache);
    const size_t min_successes =
        Parallel::get<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks>(
            cache);
    const size_t max_successes =
        Parallel::get<gh::bbh::Tags::MaxCommonHorizonSuccesses>(cache);
    if (success_count < min_successes) {
      return false;
    }

    if (success_count >= max_successes) {
      Parallel::mutate<gh::bbh::Tags::MaxCommonHorizonSuccessesReached,
                       LatchMaxCommonHorizonSuccessesReachedAndPrint>(
          cache, time, success_count, max_successes);
      return true;
    }

    return Parallel::get<gh::bbh::Tags::GaugeConstraintExceeded>(cache) or
           Parallel::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(cache) or
           Parallel::get<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(
               cache);
  }

  using next_check_time_return_tags = tmpl::list<>;
  using next_check_time_argument_tags = tmpl::list<::Tags::Time>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<double> next_check_time(
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const Component* /*component*/,
      double time) const {
    const double interval =
        Parallel::get<gh::bbh::Tags::ConstraintCheckInterval>(cache);
    ASSERT(interval > 0.0, "ConstraintCheckInterval must be positive.");
    return time + interval;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  struct LatchMaxCommonHorizonSuccessesReachedAndPrint {
    static void apply(
        const gsl::not_null<bool*> max_common_horizon_successes_reached,
        const double time, const size_t success_count,
        const size_t max_successes) {
      if (not *max_common_horizon_successes_reached) {
        *max_common_horizon_successes_reached = true;
        Parallel::printf(
            "BBH completion criterion met at t=%.16f: AhC successes=%zu >= "
            "%zu.\n",
            time, success_count, max_successes);
      }
    }
  };
};
}  // namespace gh::bbh::Triggers
