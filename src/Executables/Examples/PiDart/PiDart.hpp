// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/DistributedObject.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace PUP {
class er;
}  // namespace PUP

template <typename Metavars>
struct DartThrower;

template <typename Metavars>
struct PiEstimator;

namespace {
size_t throw_n_darts(const size_t n) {
  std::random_device device;
  std::mt19937 gen(device());
  UniformCustomDistribution<double> dist{0.0, 1.0};
  DataVector x{n};
  fill_with_random_values(make_not_null(&x), make_not_null(&gen),
                          make_not_null(&dist));
  DataVector y{n};
  fill_with_random_values(make_not_null(&y), make_not_null(&gen),
                          make_not_null(&dist));
  return n -
         static_cast<size_t>(sum(step_function((square(x) + square(y)) - 1.0)));
}
}  // namespace

namespace OptionTags {
struct DartsPerIteration {
  using type = size_t;
  static constexpr Options::String help = {
      "Number of darts a core throws each iteration."};
  static size_t lower_bound() { return 1; };
};

struct FractionalAccuracyGoal {
  using type = double;
  static constexpr Options::String help = {
      "Fractional accuracy goal for pi estimate."};
  static double lower_bound() { return 1.e-50; };
};
}  // namespace OptionTags

namespace Tags {
struct ThrowsAllProcs : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = false;
  static size_t create_from_options() { return 0; }
};

struct HitsAllProcs : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = false;
  static size_t create_from_options() { return 0; }
};

struct DartsPerIteration : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::DartsPerIteration>;
  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t& darts_per_iteration) {
    return darts_per_iteration;
  };
};

struct FractionalAccuracyGoal : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::FractionalAccuracyGoal>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double& fractional_accuracy_goal) {
    return fractional_accuracy_goal;
  };
};
}  // namespace Tags

namespace Actions {
struct ProcessHitsAndThrows {
  template <typename ParallelComponent, typename DbTags, typename Metavars,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavars>& cache,
                    const ArrayIndex& array_index, const size_t value) {
    const size_t num_procs = Parallel::number_of_procs<size_t>(cache);
    const size_t darts_per_iteration = db::get<Tags::DartsPerIteration>(box);
    db::mutate_apply<tmpl::list<::Tags::HitsAllProcs, ::Tags::ThrowsAllProcs>,
                     tmpl::list<>>(
        [&value, &num_procs, &darts_per_iteration](
            const gsl::not_null<size_t*> hits_all_procs,
            const gsl::not_null<size_t*> throws_all_procs) {
          *hits_all_procs += value;
          *throws_all_procs += darts_per_iteration * num_procs;
        },
        make_not_null(&box));

    const double pi_estimate =
        4.0 * static_cast<double>(db::get<Tags::HitsAllProcs>(box)) /
        static_cast<double>(db::get<Tags::ThrowsAllProcs>(box));
    const double fractional_accuracy = abs(pi_estimate - M_PI) / M_PI;
    Parallel::printf("[%d]: pi ~ %1.15f (%zu/%zu darts hit) -- error %1.15f\n",
                     static_cast<int>(array_index), pi_estimate,
                     db::get<Tags::HitsAllProcs>(box),
                     db::get<Tags::ThrowsAllProcs>(box), fractional_accuracy);

    // Restart the algorithm on all processors...assumes array size =
    // number of processors, unless desired accuracy target reached
    if (fractional_accuracy > db::get<Tags::FractionalAccuracyGoal>(box)) {
      for (size_t i = 0; i < num_procs; ++i) {
        Parallel::get_parallel_component<DartThrower<Metavars>>(cache)[i]
            .perform_algorithm(true);
      }
    }
  }
};

struct ThrowDarts {
  template <typename DbTags, typename... InboxTags, typename Metavars,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavars>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t new_throws = db::get<Tags::DartsPerIteration>(box);
    const size_t new_hits = throw_n_darts(new_throws);

    const auto& array_element_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index];
    const auto& singleton_proxy =
        Parallel::get_parallel_component<PiEstimator<Metavars>, Metavars>(
            cache);
    Parallel::ReductionData<Parallel::ReductionDatum<size_t, funcl::Plus<>>>
        send_hits{new_hits};
    Parallel::contribute_to_reduction<ProcessHitsAndThrows>(
        send_hits, array_element_proxy, singleton_proxy);

    Parallel::printf("[%d]: ThrowDarts: %zu/%zu new darts hit\n",
                     static_cast<int>(array_index), new_hits, new_throws);
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};
}  // namespace Actions

template <typename Metavars>
struct PiEstimator {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavars;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Execute, tmpl::list<>>>;
  using simple_tags_from_options =
      tmpl::list<Tags::ThrowsAllProcs, Tags::HitsAllProcs,
                 Tags::DartsPerIteration, Tags::FractionalAccuracyGoal>;
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavars>& global_cache);
};

template <typename Metavars>
void PiEstimator<Metavars>::execute_next_phase(
    const Parallel::Phase next_phase,
    const Parallel::CProxy_GlobalCache<Metavars>& global_cache) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  Parallel::get_parallel_component<PiEstimator<Metavars>>(local_cache)
      .start_phase(next_phase);
}

template <typename Metavars>
struct DartThrower {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavars;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Execute,
                                        tmpl::list<Actions::ThrowDarts>>>;
  using simple_tags_from_options = tmpl::list<Tags::DartsPerIteration>;
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;

  using array_index = int;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavars>& global_cache);
  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavars>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
          initialization_items,
      const std::unordered_set<size_t>& procs_to_ignore = {});
};

template <typename Metavars>
void DartThrower<Metavars>::execute_next_phase(
    const Parallel::Phase next_phase,
    const Parallel::CProxy_GlobalCache<Metavars>& global_cache) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  Parallel::get_parallel_component<DartThrower<Metavars>>(local_cache)
      .start_phase(next_phase);
}

template <typename Metavars>
void DartThrower<Metavars>::allocate_array(
    Parallel::CProxy_GlobalCache<Metavars>& global_cache,
    const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
        initialization_items,
    const std::unordered_set<size_t>& procs_to_ignore) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  auto& array_proxy =
      Parallel::get_parallel_component<DartThrower<Metavars>>(local_cache);

  size_t which_proc = 0;
  const size_t num_procs = Parallel::number_of_procs<size_t>(local_cache);
  const size_t number_of_elements = num_procs;
  for (size_t i = 0; i < number_of_elements; ++i) {
    // Round-robin loop over number of processors
    while (procs_to_ignore.find(which_proc) != procs_to_ignore.end()) {
      which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
    }
    array_proxy[i].insert(global_cache, initialization_items, which_proc);
    which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
  }
  array_proxy.doneInserting();
}

struct Metavariables {
  using component_list =
      tmpl::list<PiEstimator<Metavariables>, DartThrower<Metavariables>>;

  static constexpr std::array<Parallel::Phase, 3> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Execute,
       Parallel::Phase::Exit}};

  static constexpr Options::String help{
      "An executable that computes pi via monte carlo integration"};

  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};
