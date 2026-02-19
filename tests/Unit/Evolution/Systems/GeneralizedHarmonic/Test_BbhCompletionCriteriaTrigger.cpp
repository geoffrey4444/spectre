// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/Triggers/CompletionCriteria.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct MockMetavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks,
                 gh::bbh::Tags::MaxCommonHorizonSuccesses,
                 gh::bbh::Tags::GaugeConstraintLinfThreshold,
                 gh::bbh::Tags::ThreeIndexConstraintLinfThreshold,
                 gh::bbh::Tags::ConstraintCheckInterval>;
  using mutable_global_cache_tags =
      tmpl::list<gh::bbh::Tags::GaugeConstraintExceeded,
                 gh::bbh::Tags::ThreeIndexConstraintExceeded,
                 gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold,
                 gh::bbh::Tags::CommonHorizonSuccessCount,
                 gh::bbh::Tags::MaxCommonHorizonSuccessesReached>;
};

Parallel::GlobalCache<MockMetavariables> make_cache() {
  return {{size_t{2}, size_t{3}, 10.0, 20.0, 0.5},
          {false, false, false, size_t{0}, false}};
}

SPECTRE_TEST_CASE("Unit.GeneralizedHarmonic.BbhCompletionCriteriaTrigger",
                  "[Unit][Evolution]") {
  gh::bbh::Triggers::CompletionCriteria trigger{};
  auto cache = make_cache();
  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<MockMetavariables>,
      Parallel::Tags::GlobalCache<MockMetavariables>>>(
      MockMetavariables{}, &cache);

  // Before the minimum number of AhC successes, completion criteria are gated.
  CHECK_FALSE(trigger(box));
  CHECK_FALSE(Parallel::get<gh::bbh::Tags::GaugeConstraintExceeded>(cache));
  CHECK_FALSE(
      Parallel::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(cache));

  // Arm checks once we have enough successful AhC finds.
  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      cache);
  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      cache);

  CHECK_FALSE(trigger(box));
  Parallel::mutate<gh::bbh::Tags::GaugeConstraintExceeded,
                   gh::bbh::Mutators::SetGaugeConstraintExceeded>(cache);
  CHECK(trigger(box));

  // Max success count is also a completion criterion.
  auto count_cache = make_cache();
  auto count_box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<MockMetavariables>,
      Parallel::Tags::GlobalCache<MockMetavariables>>>(
      MockMetavariables{}, &count_cache);
  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      count_cache);
  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      count_cache);
  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      count_cache);
  CHECK(trigger(count_box));
}
}  // namespace
