// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
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

tnsr::a<DataVector, 3, Frame::Inertial> make_gauge_constraint(
    const double max_abs_value) {
  tnsr::a<DataVector, 3, Frame::Inertial> gauge_constraint(4_st, 0.0);
  get<0>(gauge_constraint) = DataVector{2_st, 0.0};
  get<0>(gauge_constraint)[1] = max_abs_value;
  return gauge_constraint;
}

tnsr::iaa<DataVector, 3, Frame::Inertial> make_three_index_constraint(
    const double max_abs_value) {
  tnsr::iaa<DataVector, 3, Frame::Inertial> three_index_constraint(4_st, 0.0);
  get<0, 0, 0>(three_index_constraint) = DataVector{2_st, 0.0};
  get<0, 0, 0>(three_index_constraint)[1] = max_abs_value;
  return three_index_constraint;
}

SPECTRE_TEST_CASE("Unit.GeneralizedHarmonic.BbhCheckConstraintThresholdsEvent",
                  "[Unit][Evolution]") {
  gh::bbh::Events::CheckConstraintThresholds event{};
  auto cache = make_cache();

  // Before enough AhC successes, no latching occurs.
  event(1.0, make_gauge_constraint(1.e6), make_three_index_constraint(1.e6),
        cache, 0_st, static_cast<const MockComponent*>(nullptr),
        Event::ObservationValue{"Time", 1.0});
  CHECK_FALSE(Parallel::get<gh::bbh::Tags::GaugeConstraintExceeded>(cache));
  CHECK_FALSE(
      Parallel::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(cache));

  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      cache);
  Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                   gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
      cache);

  event(2.0, make_gauge_constraint(11.0), make_three_index_constraint(1.0),
        cache, 0_st, static_cast<const MockComponent*>(nullptr),
        Event::ObservationValue{"Time", 2.0});
  CHECK(Parallel::get<gh::bbh::Tags::GaugeConstraintExceeded>(cache));
  CHECK_FALSE(
      Parallel::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(cache));

  event(3.0, make_gauge_constraint(1.0), make_three_index_constraint(21.0),
        cache, 0_st, static_cast<const MockComponent*>(nullptr),
        Event::ObservationValue{"Time", 3.0});
  CHECK(Parallel::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(cache));
}
}  // namespace
