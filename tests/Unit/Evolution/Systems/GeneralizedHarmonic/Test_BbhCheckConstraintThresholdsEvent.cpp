// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/Events/CheckConstraintThresholds.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct MockMetavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks,
                 gh::bbh::Tags::GaugeConstraintLinfThreshold,
                 gh::bbh::Tags::ThreeIndexConstraintLinfThreshold,
                 gh::bbh::Tags::ConstraintCheckVerbose>;
  using mutable_global_cache_tags =
      tmpl::list<gh::bbh::Tags::GaugeConstraintExceeded,
                 gh::bbh::Tags::ThreeIndexConstraintExceeded,
                 gh::bbh::Tags::CommonHorizonSuccessCount,
                 gh::bbh::Tags::CompletionRequested>;
};

struct MockSingletonComponent {};

Parallel::GlobalCache<MockMetavariables> make_cache() {
  constexpr size_t min_successes_before_checks = 2;
  constexpr double gauge_constraint_linf_threshold = 10.0;
  constexpr double three_index_constraint_linf_threshold = 20.0;
  constexpr bool verbose_checks = false;
  constexpr bool gauge_constraint_exceeded_initial = false;
  constexpr bool three_index_constraint_exceeded_initial = false;
  constexpr size_t common_horizon_success_count_initial = 0;
  constexpr bool completion_requested_initial = false;
  return Parallel::GlobalCache<MockMetavariables>{
      {min_successes_before_checks, gauge_constraint_linf_threshold,
       three_index_constraint_linf_threshold, verbose_checks},
      {gauge_constraint_exceeded_initial,
       three_index_constraint_exceeded_initial,
       common_horizon_success_count_initial, completion_requested_initial},
      std::vector<size_t>{1},
      0,
      0,
      0};
}

void test_check_constraint_thresholds_event() {
  auto cache = make_cache();
  auto box = db::create<db::AddSimpleTags<>>();

  // Reduction callback runs on the singleton reduction target and latches
  // based on globally reduced maxima.
  constexpr int64_t first_slab_number = 4;
  constexpr double first_time = 2.0;
  constexpr double first_max_gauge_linf = 11.0;
  constexpr double first_max_three_index_linf = 1.0;
  gh::bbh::Events::CheckConstraintThresholds::ProcessConstraintMaxima::
      template apply<MockSingletonComponent>(
          box, cache, 0, first_time, first_slab_number, first_max_gauge_linf,
          first_max_three_index_linf);
  CHECK(Parallel::get<gh::bbh::Tags::GaugeConstraintExceeded>(cache));
  CHECK_FALSE(
      Parallel::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(cache));
  CHECK(Parallel::get<gh::bbh::Tags::CompletionRequested>(cache));

  constexpr int64_t second_slab_number = 6;
  constexpr double second_time = 3.0;
  constexpr double second_max_gauge_linf = 1.0;
  constexpr double second_max_three_index_linf = 21.0;
  gh::bbh::Events::CheckConstraintThresholds::ProcessConstraintMaxima::
      template apply<MockSingletonComponent>(
          box, cache, 0, second_time, second_slab_number, second_max_gauge_linf,
          second_max_three_index_linf);
  CHECK(Parallel::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(cache));
}

SPECTRE_TEST_CASE("Unit.GeneralizedHarmonic.BbhCheckConstraintThresholdsEvent",
                  "[Unit][Evolution]") {
  // clang-analyzer reports the destruction-path diagnostic at this call site.
  // clang-analyzer follows a destructor path through Charm++ proxy internals
  // (`CProxy::~CProxy` -> `CkDelegateData::unref`) and reports a
  // `clang-analyzer-cplusplus.NewDelete` use-after-free that does not originate
  // in this test logic. Scope the suppression to this helper so the rest of the
  // file remains fully checked.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete)
  test_check_constraint_thresholds_event();
}
}  // namespace
