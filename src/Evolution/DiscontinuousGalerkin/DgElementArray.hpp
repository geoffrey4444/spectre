// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/Actions/SetData.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/ElementId.hpp"               // IWYU pragma: keep
#include "Domain/ElementIndex.hpp"
#include "Domain/InitialElementIds.hpp"
#include "ErrorHandling/Error.hpp"
#include "IO/DataImporter/DataFileReader.hpp"
#include "IO/DataImporter/DataFileReaderActions.hpp"
#include "IO/DataImporter/ElementActions.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Time/Tags.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace observers {
namespace Actions {
template <observers::TypeOfObservation TypeOfObservation>
struct RegisterWithObservers;
}  // namespace Actions
}  // namespace observers
/// \endcond

template <class Metavariables, class InitializeAction, class ActionList,
          // Since these template parameters are only needed for the
          // `ReadElementData` action, we can remove them once we have PDAL's
          class ImporterOptionsGroup, class ImportFieldsTags>
struct DgElementArray {
  static constexpr size_t volume_dim = Metavariables::system::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using action_list = ActionList;
  using array_index = ElementIndex<volume_dim>;

  using read_element_data_action = importer::ThreadedActions::ReadElementData<
      ImporterOptionsGroup, ImportFieldsTags,
      db::Actions::SetData<ImportFieldsTags>, DgElementArray>;

  using const_global_cache_tag_list = tmpl::append<
      Parallel::get_const_global_cache_tags<action_list>,
      typename read_element_data_action::const_global_cache_tag_list>;

  using initial_databox = db::compute_databox_type<
      typename InitializeAction::template return_tag_list<Metavariables>>;

  using options = tmpl::flatten<tmpl::list<
      typename Metavariables::domain_creator_tag, OptionTags::InitialTime,
      OptionTags::InitialTimeStep,
      tmpl::conditional_t<tmpl::list_contains_v<const_global_cache_tag_list,
                                                OptionTags::StepController>,
                          OptionTags::InitialSlabSize, tmpl::list<>>>>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      std::unique_ptr<DomainCreator<volume_dim, Frame::Inertial>>
          domain_creator,
      double initial_time, double initial_dt) noexcept;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      std::unique_ptr<DomainCreator<volume_dim, Frame::Inertial>>
          domain_creator,
      double initial_time, double initial_dt,
      double initial_slab_size) noexcept;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& dg_element_array =
        Parallel::get_parallel_component<DgElementArray>(local_cache);
    switch (next_phase) {
      case Metavariables::Phase::Evolve:
        dg_element_array.perform_algorithm();
        break;
      case Metavariables::Phase::RegisterElements:
        // We currently use an observation_id with a fake time when registering
        // observers but in the future when we start doing load balancing and
        // elements migrate around the system they will need to register and
        // unregister themselves at specific times.
        Parallel::simple_action<observers::Actions::RegisterWithObservers<
            observers::TypeOfObservation::ReductionAndVolume>>(
            dg_element_array,
            observers::ObservationId(
                0., typename Metavariables::element_observation_type{}));
        // Do this here for now. With PDAL's this is part of the action list.
        Parallel::simple_action<importer::Actions::RegisterWithImporter>(
            dg_element_array);
        break;
      case Metavariables::Phase::ImportData:
        Parallel::threaded_action<read_element_data_action>(
            Parallel::get_parallel_component<
                importer::DataFileReader<Metavariables>>(local_cache));
        break;
      default:
        break;
    }
    //    if (next_phase == Metavariables::Phase::Evolve) {
    //      Parallel::get_parallel_component<DgElementArray>(local_cache)
    //          .perform_algorithm();
    //    } else {
    //      try_register_with_observers(next_phase, global_cache);
    //    }
  }

 private:
  template <typename PhaseType,
            Requires<not observers::has_register_with_observer_v<PhaseType>> =
                nullptr>
  static void try_register_with_observers(
      const PhaseType /*next_phase*/,
      Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) noexcept {}

  template <
      typename PhaseType,
      Requires<observers::has_register_with_observer_v<PhaseType>> = nullptr>
  static void try_register_with_observers(
      const PhaseType next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    if (next_phase == Metavariables::Phase::RegisterWithObserver) {
      auto& local_cache = *(global_cache.ckLocalBranch());
      // We currently use an observation_id with a fake time when registering
      // observers but in the future when we start doing load balancing and
      // elements migrate around the system they will need to register and
      // unregister themselves at specific times.
      const observers::ObservationId observation_id_with_fake_time(
          0., typename Metavariables::element_observation_type{});
      Parallel::simple_action<observers::Actions::RegisterWithObservers<
          observers::TypeOfObservation::ReductionAndVolume>>(
          Parallel::get_parallel_component<DgElementArray>(local_cache),
          observation_id_with_fake_time);
    }
  }
};

template <class Metavariables, class InitializeAction, class ActionList,
          class ImporterOptionsGroup, class ImportFieldsTags>
void DgElementArray<Metavariables, InitializeAction, ActionList,
                    ImporterOptionsGroup, ImportFieldsTags>::
    initialize(Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
               std::unique_ptr<DomainCreator<volume_dim, Frame::Inertial>>
                   domain_creator,
               const double initial_time, const double initial_dt) noexcept {
  initialize(global_cache, std::move(domain_creator), initial_time, initial_dt,
             std::abs(initial_dt));
}

template <class Metavariables, class InitializeAction, class ActionList,
          class ImporterOptionsGroup, class ImportFieldsTags>
void DgElementArray<Metavariables, InitializeAction, ActionList,
                    ImporterOptionsGroup, ImportFieldsTags>::
    initialize(Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
               const std::unique_ptr<DomainCreator<volume_dim, Frame::Inertial>>
                   domain_creator,
               const double initial_time, const double initial_dt,
               const double initial_slab_size) noexcept {
  auto& cache = *global_cache.ckLocalBranch();
  auto& dg_element_array =
      Parallel::get_parallel_component<DgElementArray>(cache);

  if (not Metavariables::local_time_stepping and
      std::abs(initial_dt) != initial_slab_size) {
    ERROR("Step and slab size must agree for global time-stepping.");
  }

  auto domain = domain_creator->create_domain();
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs =
        domain_creator->initial_refinement_levels()[block.id()];
    const std::vector<ElementId<volume_dim>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    int which_proc = 0;
    const int number_of_procs = Parallel::number_of_procs();
    for (size_t i = 0; i < element_ids.size(); ++i) {
      dg_element_array(ElementIndex<volume_dim>(element_ids[i]))
          .insert(global_cache, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
  }
  dg_element_array.doneInserting();

  Parallel::simple_action<InitializeAction>(
      dg_element_array, domain_creator->initial_extents(), std::move(domain),
      initial_time, initial_dt, initial_slab_size);
}
