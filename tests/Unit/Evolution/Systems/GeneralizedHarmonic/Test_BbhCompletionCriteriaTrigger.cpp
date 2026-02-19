// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
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

struct MockComponent {};

Parallel::GlobalCache<MockMetavariables> make_cache() {
  return {{size_t{2}, size_t{3}, 10.0, 20.0, 0.5},
          {false, false, false, size_t{0}, false}};
}

SPECTRE_TEST_CASE("Unit.GeneralizedHarmonic.BbhCompletionCriteriaTrigger",
                  "[Unit][Evolution]") {
  gh::bbh::Triggers::CompletionCriteria trigger{};
  auto cache = make_cache();
  CHECK(trigger.next_check_time(cache, 0_st,
                                static_cast<const MockComponent*>(nullptr),
                                1.25) == std::optional{1.75});

  // Before the minimum number of AhC successes, completion criteria are gated.
  CHECK(trigger
            .is_triggered(cache, 0_st,
                          static_cast<const MockComponent*>(nullptr), 1.25)
            .value() == false);
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

  CHECK(trigger
            .is_triggered(cache, 0_st,
                          static_cast<const MockComponent*>(nullptr), 1.75)
            .value() == false);
  Parallel::mutate<gh::bbh::Tags::GaugeConstraintExceeded,
                   gh::bbh::Mutators::SetGaugeConstraintExceeded>(cache);
  CHECK(trigger
            .is_triggered(cache, 0_st,
                          static_cast<const MockComponent*>(nullptr), 2.25)
            .value() == true);

  // Max success count is also a completion criterion.
  auto count_cache = make_cache();
  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      count_cache);
  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      count_cache);
  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      count_cache);
  CHECK(trigger
            .is_triggered(count_cache, 0_st,
                          static_cast<const MockComponent*>(nullptr), 3.5)
            .value() == true);
}
}  // namespace
