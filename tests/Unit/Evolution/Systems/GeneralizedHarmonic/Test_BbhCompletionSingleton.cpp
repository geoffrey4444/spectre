// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionSingleton.hpp"
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
                 gh::bbh::Tags::CommonHorizonLMaxThreshold,
                 gh::bbh::Tags::ConstraintCheckVerbose>;
  using mutable_global_cache_tags = tmpl::list<>;
};

struct MockSingletonComponent {};

auto make_cache() {
  constexpr size_t min_common_horizon_successes_before_checks = 2;
  constexpr size_t max_common_horizon_successes = 100;
  constexpr double gauge_constraint_linf_threshold = 10.0;
  constexpr double three_index_constraint_linf_threshold = 20.0;
  constexpr size_t common_horizon_lmax_threshold = 6;
  constexpr bool constraint_check_verbose = false;
  return Parallel::GlobalCache<MockMetavariables>{
      {min_common_horizon_successes_before_checks, max_common_horizon_successes,
       gauge_constraint_linf_threshold, three_index_constraint_linf_threshold,
       common_horizon_lmax_threshold, constraint_check_verbose}};
}

auto make_box() {
  return db::create<
      db::AddSimpleTags<gh::bbh::Tags::GaugeConstraintExceeded,
                        gh::bbh::Tags::ThreeIndexConstraintExceeded,
                        gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold,
                        gh::bbh::Tags::CommonHorizonSuccessCount,
                        gh::bbh::Tags::CompletionRequested,
                        gh::bbh::Tags::CommonHorizonSuccessRecords,
                        gh::bbh::Tags::ConstraintCheckRecords,
                        gh::bbh::Tags::ReportedConstraintCheckRecords>>(
      false, false, false, 0_st, false,
      gh::bbh::Tags::CommonHorizonSuccessRecords::type{},
      gh::bbh::Tags::ConstraintCheckRecords::type{},
      gh::bbh::Tags::ReportedConstraintCheckRecords::type{});
}

SPECTRE_TEST_CASE("Unit.GeneralizedHarmonic.BbhCompletionSingleton",
                  "[Unit][Evolution]") {
  {
    INFO("Constraint checks remain gated by min successes at their check time");
    auto cache = make_cache();
    auto box = make_box();

    gh::bbh::Actions::ProcessConstraintMaxima::template apply<
        MockSingletonComponent>(box, cache, 0, 1.5, 1, 11.0, 1.0);
    CHECK_FALSE(db::get<gh::bbh::Tags::GaugeConstraintExceeded>(box));
    CHECK_FALSE(db::get<gh::bbh::Tags::CompletionRequested>(box));

    gh::bbh::Actions::RecordCommonHorizonSuccess::template apply<
        MockSingletonComponent>(box, cache, 0,
                                LinkedMessageId<double>{2.0, std::nullopt}, 8);
    gh::bbh::Actions::RecordCommonHorizonSuccess::template apply<
        MockSingletonComponent>(box, cache, 0,
                                LinkedMessageId<double>{3.0, std::nullopt}, 8);
    CHECK(db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(box) == 2);
    CHECK_FALSE(db::get<gh::bbh::Tags::GaugeConstraintExceeded>(box));
    CHECK_FALSE(db::get<gh::bbh::Tags::CompletionRequested>(box));

    // Delayed older successes still don't arm the t=1.5 constraint check until
    // enough successes exist at or before that check time.
    gh::bbh::Actions::RecordCommonHorizonSuccess::template apply<
        MockSingletonComponent>(box, cache, 0,
                                LinkedMessageId<double>{1.0, std::nullopt}, 8);
    CHECK_FALSE(db::get<gh::bbh::Tags::GaugeConstraintExceeded>(box));
    CHECK_FALSE(db::get<gh::bbh::Tags::CompletionRequested>(box));

    gh::bbh::Actions::RecordCommonHorizonSuccess::template apply<
        MockSingletonComponent>(box, cache, 0,
                                LinkedMessageId<double>{1.2, std::nullopt}, 8);
    CHECK(db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(box) == 4);
    CHECK(db::get<gh::bbh::Tags::GaugeConstraintExceeded>(box));
    CHECK(db::get<gh::bbh::Tags::CompletionRequested>(box));
  }

  {
    INFO("Delayed older AhC success can retroactively satisfy the LMax path");
    auto cache = make_cache();
    auto box = make_box();

    gh::bbh::Actions::RecordCommonHorizonSuccess::template apply<
        MockSingletonComponent>(box, cache, 0,
                                LinkedMessageId<double>{2.0, std::nullopt}, 8);
    gh::bbh::Actions::RecordCommonHorizonSuccess::template apply<
        MockSingletonComponent>(box, cache, 0,
                                LinkedMessageId<double>{3.0, std::nullopt}, 8);
    CHECK(db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(box) == 2);
    CHECK_FALSE(
        db::get<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(box));
    CHECK_FALSE(db::get<gh::bbh::Tags::CompletionRequested>(box));

    gh::bbh::Actions::RecordCommonHorizonSuccess::template apply<
        MockSingletonComponent>(box, cache, 0,
                                LinkedMessageId<double>{1.0, std::nullopt}, 6);
    CHECK(db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(box) == 3);
    CHECK(db::get<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(box));
    CHECK(db::get<gh::bbh::Tags::CompletionRequested>(box));
  }

  {
    INFO("Element completion-request simple action latches true");
    auto box = db::create<
        db::AddSimpleTags<gh::bbh::Tags::ElementCompletionRequested>>(false);
    auto cache = make_cache();
    gh::bbh::Actions::SetElementCompletionRequested::template apply<
        MockSingletonComponent>(box, cache, 0);
    CHECK(db::get<gh::bbh::Tags::ElementCompletionRequested>(box));
  }
}
}  // namespace
