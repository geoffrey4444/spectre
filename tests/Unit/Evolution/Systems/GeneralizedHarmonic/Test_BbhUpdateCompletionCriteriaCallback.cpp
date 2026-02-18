// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/Callbacks/UpdateCompletionCriteria.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct MockHorizonMetavars : tt::ConformsTo<ah::protocols::HorizonMetavars> {
  using time_tag = ::Tags::TimeAndPrevious<0>;
  using frame = ::Frame::Inertial;
  using horizon_find_callbacks = tmpl::list<>;
  using horizon_find_failure_callbacks = tmpl::list<>;
  using compute_tags_on_element = tmpl::list<>;
  static constexpr ah::Destination destination = ah::Destination::Observation;
  static std::string name() { return "MockHorizonMetavars"; }
};

struct MockMetavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks,
                 gh::bbh::Tags::CommonHorizonLMaxThreshold>;
  using mutable_global_cache_tags =
      tmpl::list<gh::bbh::Tags::CommonHorizonSuccessCount,
                 gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>;
};

auto make_cache() {
  return Parallel::GlobalCache<MockMetavariables>{{size_t{2}, size_t{6}},
                                                  {size_t{0}, false}};
}

auto make_box(const double time, const size_t l_max) {
  return db::create<db::AddSimpleTags<
      ah::Tags::CurrentTime, ylm::Tags::Strahlkorper<Frame::Inertial>>>(
      std::optional{LinkedMessageId<double>{time, std::nullopt}},
      ylm::Strahlkorper<Frame::Inertial>{l_max, 2.0,
                                         std::array{0.0, 0.0, 0.0}});
}

SPECTRE_TEST_CASE(
    "Unit.GeneralizedHarmonic.BbhUpdateCompletionCriteriaCallback",
    "[Unit][Evolution]") {
  (void)MockHorizonMetavars::destination;
  {
    auto cache = make_cache();
    auto box = make_box(1.0, 8);
    gh::bbh::callbacks::UpdateCompletionCriteria<MockHorizonMetavars>::apply(
        box, cache, FastFlow::Status::TruncationTol);
    CHECK(Parallel::get<gh::bbh::Tags::CommonHorizonSuccessCount>(cache) == 1);
    CHECK_FALSE(
        Parallel::get<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(
            cache));

    box = make_box(2.5, 6);
    gh::bbh::callbacks::UpdateCompletionCriteria<MockHorizonMetavars>::apply(
        box, cache, FastFlow::Status::TruncationTol);
    CHECK(Parallel::get<gh::bbh::Tags::CommonHorizonSuccessCount>(cache) == 2);
    CHECK(Parallel::get<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(
        cache));
  }
}
}  // namespace
