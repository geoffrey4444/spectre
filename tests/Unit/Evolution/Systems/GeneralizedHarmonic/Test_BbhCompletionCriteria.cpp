// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.BbhCompletionCriteria",
    "[Unit][Evolution]") {
  auto box = db::create<db::AddSimpleTags<
      gh::bbh::Tags::GaugeConstraintExceeded,
      gh::bbh::Tags::ThreeIndexConstraintExceeded,
      gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold,
      gh::bbh::Tags::CommonHorizonSuccessCount,
      gh::bbh::Tags::MaxCommonHorizonSuccessesReached,
      gh::bbh::Tags::CompletionRequested, gh::bbh::Tags::StopSlabNumber>>(
      false, false, false, 0_st, false, false,
      std::numeric_limits<size_t>::max());

  db::mutate<gh::bbh::Tags::GaugeConstraintExceeded>(
      [](const gsl::not_null<bool*> flag) {
        gh::bbh::Mutators::SetGaugeConstraintExceeded::apply(flag);
      },
      make_not_null(&box));
  db::mutate<gh::bbh::Tags::ThreeIndexConstraintExceeded>(
      [](const gsl::not_null<bool*> flag) {
        gh::bbh::Mutators::SetThreeIndexConstraintExceeded::apply(flag);
      },
      make_not_null(&box));
  db::mutate<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(
      [](const gsl::not_null<bool*> flag) {
        gh::bbh::Mutators::SetCommonHorizonLMaxBelowOrEqualThreshold::apply(
            flag);
      },
      make_not_null(&box));
  db::mutate<gh::bbh::Tags::CommonHorizonSuccessCount>(
      [](const gsl::not_null<size_t*> count) {
        gh::bbh::Mutators::IncrementCommonHorizonSuccessCount::apply(count);
      },
      make_not_null(&box));
  db::mutate<gh::bbh::Tags::MaxCommonHorizonSuccessesReached>(
      [](const gsl::not_null<bool*> flag) {
        gh::bbh::Mutators::SetMaxCommonHorizonSuccessesReached::apply(flag);
      },
      make_not_null(&box));
  db::mutate<gh::bbh::Tags::CompletionRequested>(
      [](const gsl::not_null<bool*> flag) {
        gh::bbh::Mutators::SetCompletionRequested::apply(flag);
      },
      make_not_null(&box));
  db::mutate<gh::bbh::Tags::StopSlabNumber>(
      [](const gsl::not_null<size_t*> stop_slab_number) {
        gh::bbh::Mutators::SetStopSlabNumberIfUnset::apply(stop_slab_number,
                                                           7_st);
      },
      make_not_null(&box));

  CHECK(db::get<gh::bbh::Tags::GaugeConstraintExceeded>(box));
  CHECK(db::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(box));
  CHECK(db::get<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(box));
  CHECK(db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(box) == 1_st);
  CHECK(db::get<gh::bbh::Tags::MaxCommonHorizonSuccessesReached>(box));
  CHECK(db::get<gh::bbh::Tags::CompletionRequested>(box));
  CHECK(db::get<gh::bbh::Tags::StopSlabNumber>(box) == 7_st);

  // Ensure mutators are monotonic.
  db::mutate<gh::bbh::Tags::GaugeConstraintExceeded>(
      [](const gsl::not_null<bool*> flag) {
        gh::bbh::Mutators::SetGaugeConstraintExceeded::apply(flag);
      },
      make_not_null(&box));
  db::mutate<gh::bbh::Tags::CommonHorizonSuccessCount>(
      [](const gsl::not_null<size_t*> count) {
        gh::bbh::Mutators::IncrementCommonHorizonSuccessCount::apply(count);
      },
      make_not_null(&box));
  db::mutate<gh::bbh::Tags::StopSlabNumber>(
      [](const gsl::not_null<size_t*> stop_slab_number) {
        gh::bbh::Mutators::SetStopSlabNumberIfUnset::apply(stop_slab_number,
                                                           9_st);
      },
      make_not_null(&box));
  CHECK(db::get<gh::bbh::Tags::GaugeConstraintExceeded>(box));
  CHECK(db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(box) == 2_st);
  CHECK(db::get<gh::bbh::Tags::StopSlabNumber>(box) == 7_st);
}
