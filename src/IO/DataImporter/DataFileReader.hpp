// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmNodegroup.hpp"
#include "IO/DataImporter/Tags.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"

namespace importer {

namespace detail {
struct InitializeDataFileReader;
}  // namespace detail

/*!
 * \brief A nodegroup parallel component that reads in a volume data file and
 * distributes its data to elements of an array parallel component.
 *
 * Each element of the array parallel component must register itself before
 * data can be sent to it. To do so, invoke
 * `importer::Actions::RegisterWithImporter` on each the element. In a
 * subsequent phase you can then invoke
 * `importer::ThreadedActions::ReadElementData` on the `DataFileReader`
 * component to read in the file and distribute its data to the registered
 * elements.
 */
template <typename Metavariables>
struct DataFileReader {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using options = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<detail::InitializeDataFileReader>>>;
  using const_global_cache_tag_list = tmpl::list<>;

  static void initialize(Parallel::CProxy_ConstGlobalCache<
                         Metavariables>& /*global_cache*/) noexcept {}

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<DataFileReader>(local_cache)
        .start_phase(next_phase);
  }
};

namespace detail {
struct InitializeDataFileReader {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags = db::AddSimpleTags<Tags::RegisteredElements>;
    using compute_tags = db::AddComputeTags<>;

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeDataFileReader,
                                             simple_tags, compute_tags>(
            std::move(box), db::item_type<Tags::RegisteredElements>{}),
        true);
  }
};
}  // namespace detail

}  // namespace importer
