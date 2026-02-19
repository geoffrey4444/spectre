// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
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
                 gh::bbh::Tags::ThreeIndexConstraintLinfThreshold>;
  using mutable_global_cache_tags =
      tmpl::list<gh::bbh::Tags::GaugeConstraintExceeded,
                 gh::bbh::Tags::ThreeIndexConstraintExceeded,
                 gh::bbh::Tags::CommonHorizonSuccessCount>;
};

struct MockComponent {};

Parallel::GlobalCache<MockMetavariables> make_cache() {
  return {{size_t{2}, 10.0, 20.0}, {false, false, size_t{0}}};
}

SPECTRE_TEST_CASE("Unit.GeneralizedHarmonic.BbhCheckConstraintThresholdsEvent",
                  "[Unit][Evolution]") {
  auto cache = make_cache();
  auto box = db::create<db::AddSimpleTags<>>();

  // Reduction callback does not latch from non-designated element.
  gh::bbh::Events::CheckConstraintThresholds::ProcessConstraintMaxima::
      template apply<MockComponent>(box, cache, ElementId<3>{1}, 1.0, 100.0,
                                    100.0);
  CHECK_FALSE(Parallel::get<gh::bbh::Tags::GaugeConstraintExceeded>(cache));
  CHECK_FALSE(
      Parallel::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(cache));

  // Reduction callback latches based on globally reduced maxima.
  gh::bbh::Events::CheckConstraintThresholds::ProcessConstraintMaxima::
      template apply<MockComponent>(box, cache, ElementId<3>{0}, 2.0, 11.0,
                                    1.0);
  CHECK(Parallel::get<gh::bbh::Tags::GaugeConstraintExceeded>(cache));
  CHECK_FALSE(
      Parallel::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(cache));

  gh::bbh::Events::CheckConstraintThresholds::ProcessConstraintMaxima::
      template apply<MockComponent>(box, cache, ElementId<3>{0}, 3.0, 1.0,
                                    21.0);
  CHECK(Parallel::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(cache));
}
}  // namespace
