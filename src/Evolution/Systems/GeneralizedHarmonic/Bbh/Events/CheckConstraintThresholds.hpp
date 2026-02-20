// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::bbh::Events {
class CheckConstraintThresholds : public Event {
  using ReductionData = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<int64_t, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<double, funcl::Max<>>,
      Parallel::ReductionDatum<double, funcl::Max<>>>;

 public:
  /// \cond
  explicit CheckConstraintThresholds(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CheckConstraintThresholds);  // NOLINT
  /// \endcond

  using compute_tags_for_observation_box =
      tmpl::list<gh::Tags::GaugeConstraintCompute<3, Frame::Inertial>,
                 gh::Tags::ThreeIndexConstraintCompute<3, Frame::Inertial>>;
  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Checks local Linf constraints against BBH completion thresholds and "
      "latches global-cache booleans through a singleton reduction callback.";
  static std::string name() { return "BbhCheckConstraintThresholds"; }

  CheckConstraintThresholds() = default;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<
      ::Tags::Time, ::Tags::TimeStepId,
      gh::Tags::GaugeConstraint<DataVector, 3, Frame::Inertial>,
      gh::Tags::ThreeIndexConstraint<DataVector, 3, Frame::Inertial>>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(
      const double time, const TimeStepId& time_step_id,
      const tnsr::a<DataVector, 3, Frame::Inertial>& gauge_constraint,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& three_index_constraint,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const Component* const /*component*/,
      const ObservationValue& /*observation_value*/) const {
    const size_t success_count =
        Parallel::get<gh::bbh::Tags::CommonHorizonSuccessCount>(cache);
    const size_t min_successes =
        Parallel::get<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks>(
            cache);
    if (success_count < min_successes) {
      return;
    }

    const double local_gauge_linf = local_linf_norm(gauge_constraint);
    const double local_three_index_linf =
        local_linf_norm(three_index_constraint);
    if constexpr (Parallel::is_dg_element_collection_v<Component>) {
      ERROR(
          "BbhCheckConstraintThresholds currently requires array components "
          "(not DgElementCollection).");
    } else {
      const auto& self_proxy =
          Parallel::get_parallel_component<Component>(cache)[array_index];
      auto& reduction_target_proxy = Parallel::get_parallel_component<
          mem_monitor::MemoryMonitor<Metavariables>>(cache);
      Parallel::contribute_to_reduction<ProcessConstraintMaxima>(
          ReductionData{time, time_step_id.slab_number(), local_gauge_linf,
                        local_three_index_linf},
          self_proxy, reduction_target_proxy);
    }
  }

  struct ProcessConstraintMaxima {
    template <typename ParallelComponent, typename DbTags,
              typename Metavariables, typename ArrayIndex>
    static void apply(db::DataBox<DbTags>& /*box*/,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const ArrayIndex& /*array_index*/, const double time,
                      const int64_t slab_number, const double max_gauge_linf,
                      const double max_three_index_linf) {
      const double gauge_constraint_threshold =
          Parallel::get<gh::bbh::Tags::GaugeConstraintLinfThreshold>(cache);
      const double three_index_constraint_threshold =
          Parallel::get<gh::bbh::Tags::ThreeIndexConstraintLinfThreshold>(
              cache);
      const bool verbose =
          Parallel::get<gh::bbh::Tags::ConstraintCheckVerbose>(cache);
      if (verbose) {
        Parallel::printf(
            "BBH completion constraint check at t=%.16f: "
            "Linf(GaugeConstraint)=%.16e (threshold %.16e), "
            "Linf(ThreeIndexConstraint)=%.16e (threshold %.16e).\n",
            time, max_gauge_linf, gauge_constraint_threshold,
            max_three_index_linf, three_index_constraint_threshold);
      }

      if (max_gauge_linf >= gauge_constraint_threshold) {
        Parallel::mutate<gh::bbh::Tags::GaugeConstraintExceeded,
                         LatchGaugeConstraintExceededAndPrint>(
            cache, time, max_gauge_linf, gauge_constraint_threshold);
      }

      if (max_three_index_linf >= three_index_constraint_threshold) {
        Parallel::mutate<gh::bbh::Tags::ThreeIndexConstraintExceeded,
                         LatchThreeIndexConstraintExceededAndPrint>(
            cache, time, max_three_index_linf,
            three_index_constraint_threshold);
      }

      if (max_gauge_linf >= gauge_constraint_threshold or
          max_three_index_linf >= three_index_constraint_threshold) {
        const bool completion_requested =
            Parallel::get<gh::bbh::Tags::CompletionRequested>(cache);
        if (not completion_requested) {
          Parallel::mutate<gh::bbh::Tags::CompletionRequested,
                           gh::bbh::Mutators::SetCompletionRequested>(cache);
        }
        if (slab_number >= 0) {
          Parallel::mutate<gh::bbh::Tags::StopSlabNumber,
                           gh::bbh::Mutators::SetStopSlabNumberIfUnset>(
              cache, static_cast<size_t>(slab_number + 1));
        }
      }
    }
  };

  using is_ready_argument_tags = tmpl::list<>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

 private:
  template <typename TensorType>
  static double local_linf_norm(const TensorType& tensor) {
    double result = 0.0;
    for (size_t storage_index = 0; storage_index < tensor.size();
         ++storage_index) {
      const auto& component = tensor[storage_index];
      for (size_t i = 0; i < component.size(); ++i) {
        result = std::max(result, std::abs(component[i]));
      }
    }
    return result;
  }

  struct LatchGaugeConstraintExceededAndPrint {
    static void apply(const gsl::not_null<bool*> gauge_constraint_exceeded,
                      const double time, const double value,
                      const double threshold) {
      if (not *gauge_constraint_exceeded) {
        *gauge_constraint_exceeded = true;
        Parallel::printf(
            "BBH completion criterion met at t=%.16f: "
            "Linf(GaugeConstraint)=%.16e >= %.16e.\n",
            time, value, threshold);
      }
    }
  };

  struct LatchThreeIndexConstraintExceededAndPrint {
    static void apply(
        const gsl::not_null<bool*> three_index_constraint_exceeded,
        const double time, const double value, const double threshold) {
      if (not *three_index_constraint_exceeded) {
        *three_index_constraint_exceeded = true;
        Parallel::printf(
            "BBH completion criterion met at t=%.16f: "
            "Linf(ThreeIndexConstraint)=%.16e >= %.16e.\n",
            time, value, threshold);
      }
    }
  };
};
}  // namespace gh::bbh::Events
