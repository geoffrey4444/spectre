// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/Triggers/CompletionCriteria.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeStepId.hpp"
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
                 gh::bbh::Tags::MaxCommonHorizonSuccessesReached,
                 gh::bbh::Tags::CompletionRequested,
                 gh::bbh::Tags::StopSlabNumber>;
};

Parallel::GlobalCache<MockMetavariables> make_cache() {
  return {{size_t{2}, size_t{3}, 10.0, 20.0, 0.5},
          {false, false, false, size_t{0}, false, false,
           std::numeric_limits<size_t>::max()}};
}

SPECTRE_TEST_CASE("Unit.GeneralizedHarmonic.BbhCompletionCriteriaTrigger",
                  "[Unit][Evolution]") {
  gh::bbh::Triggers::CompletionCriteria trigger{};
  auto cache = make_cache();
  const Slab slab(0., 1.);
  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<MockMetavariables>,
      Parallel::Tags::GlobalCache<MockMetavariables>, Tags::TimeStepId>>(
      MockMetavariables{}, &cache, TimeStepId{true, 0, slab.start()});

  // Before the minimum number of AhC successes, completion criteria are gated.
  CHECK_FALSE(trigger(db::get<Tags::TimeStepId>(box), box));
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

  CHECK_FALSE(trigger(db::get<Tags::TimeStepId>(box), box));
  Parallel::mutate<gh::bbh::Tags::GaugeConstraintExceeded,
                   gh::bbh::Mutators::SetGaugeConstraintExceeded>(cache);
  CHECK_FALSE(trigger(db::get<Tags::TimeStepId>(box), box));
  CHECK(Parallel::get<gh::bbh::Tags::StopSlabNumber>(cache) == 2_st);
  db::mutate<Tags::TimeStepId>(
      [&slab](const gsl::not_null<TimeStepId*> time_id) {
        *time_id = TimeStepId(true, time_id->slab_number() + 1, slab.start());
      },
      make_not_null(&box));
  CHECK_FALSE(trigger(db::get<Tags::TimeStepId>(box), box));
  db::mutate<Tags::TimeStepId>(
      [&slab](const gsl::not_null<TimeStepId*> time_id) {
        *time_id = TimeStepId(true, time_id->slab_number() + 1, slab.start());
      },
      make_not_null(&box));
  CHECK(trigger(db::get<Tags::TimeStepId>(box), box));

  // Max success count is also a completion criterion.
  auto count_cache = make_cache();
  const Slab count_slab(0., 1.);
  auto count_box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<MockMetavariables>,
      Parallel::Tags::GlobalCache<MockMetavariables>, Tags::TimeStepId>>(
      MockMetavariables{}, &count_cache,
      TimeStepId{true, 10, count_slab.start()});
  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      count_cache);
  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      count_cache);
  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      count_cache);
  CHECK_FALSE(trigger(db::get<Tags::TimeStepId>(count_box), count_box));
  CHECK(Parallel::get<gh::bbh::Tags::StopSlabNumber>(count_cache) == 12_st);
  db::mutate<Tags::TimeStepId>(
      [&count_slab](const gsl::not_null<TimeStepId*> time_id) {
        *time_id =
            TimeStepId(true, time_id->slab_number() + 1, count_slab.start());
      },
      make_not_null(&count_box));
  CHECK_FALSE(trigger(db::get<Tags::TimeStepId>(count_box), count_box));
  db::mutate<Tags::TimeStepId>(
      [&count_slab](const gsl::not_null<TimeStepId*> time_id) {
        *time_id =
            TimeStepId(true, time_id->slab_number() + 1, count_slab.start());
      },
      make_not_null(&count_box));
  CHECK(trigger(db::get<Tags::TimeStepId>(count_box), count_box));

  // Simulate one-slab propagation delay between cache branches (e.g. nodes).
  auto branch0_cache = make_cache();
  auto branch1_cache = make_cache();
  auto branch0_box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<MockMetavariables>,
      Parallel::Tags::GlobalCache<MockMetavariables>, Tags::TimeStepId>>(
      MockMetavariables{}, &branch0_cache,
      TimeStepId{true, 10, count_slab.start()});
  auto branch1_box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<MockMetavariables>,
      Parallel::Tags::GlobalCache<MockMetavariables>, Tags::TimeStepId>>(
      MockMetavariables{}, &branch1_cache,
      TimeStepId{true, 10, count_slab.start()});
  for (auto* cache_ptr : std::array{&branch0_cache, &branch1_cache}) {
    Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                     gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
        *cache_ptr);
    Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                     gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
        *cache_ptr);
  }
  Parallel::mutate<gh::bbh::Tags::GaugeConstraintExceeded,
                   gh::bbh::Mutators::SetGaugeConstraintExceeded>(
      branch0_cache);

  CHECK_FALSE(trigger(db::get<Tags::TimeStepId>(branch0_box), branch0_box));
  CHECK(Parallel::get<gh::bbh::Tags::StopSlabNumber>(branch0_cache) == 12_st);
  CHECK_FALSE(trigger(db::get<Tags::TimeStepId>(branch1_box), branch1_box));

  db::mutate<Tags::TimeStepId>(
      [&count_slab](const gsl::not_null<TimeStepId*> time_id) {
        *time_id =
            TimeStepId(true, time_id->slab_number() + 1, count_slab.start());
      },
      make_not_null(&branch0_box));
  db::mutate<Tags::TimeStepId>(
      [&count_slab](const gsl::not_null<TimeStepId*> time_id) {
        *time_id =
            TimeStepId(true, time_id->slab_number() + 1, count_slab.start());
      },
      make_not_null(&branch1_box));
  CHECK_FALSE(trigger(db::get<Tags::TimeStepId>(branch0_box), branch0_box));
  CHECK_FALSE(trigger(db::get<Tags::TimeStepId>(branch1_box), branch1_box));

  Parallel::mutate<gh::bbh::Tags::GaugeConstraintExceeded,
                   gh::bbh::Mutators::SetGaugeConstraintExceeded>(
      branch1_cache);
  Parallel::mutate<gh::bbh::Tags::StopSlabNumber,
                   gh::bbh::Mutators::SetStopSlabNumberIfUnset>(
      branch1_cache,
      Parallel::get<gh::bbh::Tags::StopSlabNumber>(branch0_cache));
  db::mutate<Tags::TimeStepId>(
      [&count_slab](const gsl::not_null<TimeStepId*> time_id) {
        *time_id =
            TimeStepId(true, time_id->slab_number() + 1, count_slab.start());
      },
      make_not_null(&branch0_box));
  db::mutate<Tags::TimeStepId>(
      [&count_slab](const gsl::not_null<TimeStepId*> time_id) {
        *time_id =
            TimeStepId(true, time_id->slab_number() + 1, count_slab.start());
      },
      make_not_null(&branch1_box));
  CHECK(trigger(db::get<Tags::TimeStepId>(branch0_box), branch0_box));
  CHECK(trigger(db::get<Tags::TimeStepId>(branch1_box), branch1_box));
}
}  // namespace
