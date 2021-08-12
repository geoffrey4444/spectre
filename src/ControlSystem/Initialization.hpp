// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/ControlErrorTags.hpp"
#include "ControlSystem/FunctionOfTimeUpdater.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Initialization {
namespace Actions {
template <typename Metavariables>
struct ControlSystem {
  static constexpr size_t DerivOrder = Metavariables::deriv_order;

  using initialization_tags =
      tmpl::list<::Tags::Averager<DerivOrder>, ::Tags::TimescaleTuner,
                 ::Tags::ExpirationDeltaTOverDampingTimescale,
                 ::Tags::ExcisionXLocationA, ::Tags::ExcisionXLocationB>;

  using initialization_tags_to_keep = initialization_tags;

  using simple_tags =
      tmpl::list<::Tags::HorizonCenter<typename Metavariables::AhA>,
                 ::Tags::HorizonCenter<typename Metavariables::AhB>>;

  using compute_tags = tmpl::list<
      Parallel::Tags::FromGlobalCache<::domain::Tags::FunctionsOfTime>>;

  template <typename DataBox, typename... InboxTags,
            //            typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<db::tag_is_retrievable_v<
                ::Tags::HorizonCenter<typename Metavariables::AhA>, DataBox>> =
                nullptr>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::forward_as_tuple(std::move(box));
  }

  template <
      typename DataBox, typename... InboxTags,  // typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::all<initialization_tags,
                             tmpl::bind<db::tag_is_retrievable, tmpl::_1,
                                        tmpl::pin<DataBox>>>::value> = nullptr>
  static std::tuple<DataBox&&> apply(
      DataBox& /*box*/, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Dependencies not fulfilled. Did you forget to terminate the phase "
        "after removing options?");
  }
};
}  // namespace Actions
}  // namespace Initialization
