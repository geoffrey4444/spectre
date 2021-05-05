// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/FastFlow.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"

namespace intrp::callbacks {

/// \brief horizon_find_failure_callback that simply errors.
struct ErrorOnFailedApparentHorizon {
  template <typename InterpolationTargetTag, typename DbTags,
            typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const TemporalId& /*temporal_id*/,
                    const FastFlow::Status failure_reason) noexcept {
    ERROR("Apparent horizon finder "
          << pretty_type::short_name<InterpolationTargetTag>()
          << " failed, reason = " << failure_reason);
  }
};

}  // namespace intrp::callbacks
