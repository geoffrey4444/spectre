// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/FastFlow.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/PrettyType.hpp"

namespace intrp::callbacks {

/// \brief horizon_find_failure_callback that prints a message
/// and goes on.
struct IgnoreFailedApparentHorizon {
  template <typename InterpolationTargetTag, typename DbTags,
            typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const TemporalId& /*temporal_id*/,
                    const FastFlow::Status failure_reason) noexcept {
    const auto& verbosity =
        db::get<logging::Tags::Verbosity<InterpolationTargetTag>>(box);
    if (verbosity > ::Verbosity::Quiet) {
      Parallel::printf("Remark: Horizon finder %s failed, reason = %d\n",
                       pretty_type::short_name<InterpolationTargetTag>(),
                       failure_reason);
    }
  }
};

}  // namespace intrp::callbacks
