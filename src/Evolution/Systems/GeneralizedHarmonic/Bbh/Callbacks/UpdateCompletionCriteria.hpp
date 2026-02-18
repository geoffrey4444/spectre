// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::bbh::callbacks {
template <typename HorizonMetavars>
struct UpdateCompletionCriteria : tt::ConformsTo<ah::protocols::Callback> {
  using const_global_cache_tags =
      tmpl::list<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks,
                 gh::bbh::Tags::CommonHorizonLMaxThreshold>;
  using mutable_global_cache_tags =
      tmpl::list<gh::bbh::Tags::CommonHorizonSuccessCount,
                 gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>;

  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const FastFlow::Status /*status*/) {
    const double time = db::get<ah::Tags::CurrentTime>(box)->id;
    const size_t l_max =
        db::get<ylm::Tags::Strahlkorper<typename HorizonMetavars::frame>>(box)
            .l_max();
    const size_t min_successes =
        Parallel::get<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks>(
            cache);
    const size_t l_max_threshold =
        Parallel::get<gh::bbh::Tags::CommonHorizonLMaxThreshold>(cache);

    const size_t old_success_count =
        Parallel::get<gh::bbh::Tags::CommonHorizonSuccessCount>(cache);
    Parallel::mutate<gh::bbh::Tags::CommonHorizonSuccessCount,
                     gh::bbh::Mutators::IncrementCommonHorizonSuccessCount>(
        cache);
    const size_t new_success_count = old_success_count + 1;
    if (old_success_count < min_successes and
        new_success_count >= min_successes) {
      Parallel::printf(
          "BBH completion criterion armed at t=%.16f: AhC successes reached "
          "%zu (minimum required: %zu).\n",
          time, new_success_count, min_successes);
    }

    if (l_max <= l_max_threshold and
        not Parallel::get<
            gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(cache)) {
      Parallel::mutate<
          gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold,
          gh::bbh::Mutators::SetCommonHorizonLMaxBelowOrEqualThreshold>(cache);
      Parallel::printf(
          "BBH completion criterion met at t=%.16f: AhC Lmax=%zu <= %zu.\n",
          time, l_max, l_max_threshold);
    }
  }
};
}  // namespace gh::bbh::callbacks
