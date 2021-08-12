// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file defines several actions for the control systems and two structs
// with apply functions (ProlongValidityOfFunctionOfTime and
// UpdateFunctionOfTime) which are intended to act as follows:
//
// CheckValidityOfFunctionsOfTimeAndSendControlData: Runs on each DG element.
// if FofTs are invalid, suspend the algorithm and register the callback to
// restart the algorithm on update of the FofTs. Also send data if sufficient
// time has passed since a send. This is templated on the action 'WhatToSend'
// which specifies what a send means. In the BBH case WhatToSend=SendHorizonData
//
// SendHorizonData: Largely copied and pasted from the interpolator. Sends data
// necessary for a horizon find to AhA and AhB.
//
// The horizon find then finds the horizons and calls its
// 'post_horizon_find_callback's which include the actino
// 'ForwardToControlSystem' (Currently this is hacked with
// post_horizon_find_callback2)
//
// ForwardToControlSystem: Forwards whatever Tags to the ControlComponent from
// the horizon finder
//
// ReceiveHorizonCenters: Receives the horizon centers from AhA or AhB and
// places them in the DataBox of the ControlComponent at that time. If the
// center for AhA and AhB are present at that time, then call
// UpdateControlSystems
//
// UpdateControlSystems: Calculate the control error to perform a measurement.
// If enough data has been received then update the control system at the
// measurement time.

#pragma once

#include <optional>
#include <tuple>

#include "ControlSystem/Observe.hpp"
#include "ControlSystem/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/Callback.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TmplDebugging.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples

template <typename, typename>
struct DgElementArray;

struct AggregateMaximum;

template <typename>
struct ControlComponent;

template <typename>
struct Interpolator;
template <class, typename>
struct InterpolationTarget;

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace ScalarWave {

struct Psi;
}

namespace domain {
namespace Tags {
template <size_t, typename>
struct Coordinates;

struct FunctionsOfTime;
}  // namespace Tags
}  // namespace domain

// If the control system expires but does not have enough data, this is to
// prolong how long it is valid until the next expiration time. This predates
// the decision to use extrapolation of the control signal to the update time,
// so I'm not exactly sure how it might work with that design decision.
template <size_t DerivOrder>
struct ProlongTimeOfValidity {
  static void apply(
      const gsl::not_null<std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
          functions_of_time,
      const double next_expiration_time, const std::string& name) noexcept {
    try {
      auto& f_of_t = dynamic_cast<
          domain::FunctionsOfTime::PiecewisePolynomial<DerivOrder>&>(
          *(functions_of_time->at(name)));
      f_of_t.reset_expiration_time(next_expiration_time);
    } catch (const std::bad_cast&) {
      ERROR(MakeString{} << "Failed to cast FunctionOfTime" << name
                         << " to PiecewisePolynomial");
    } catch (const std::out_of_range&) {
      ERROR(MakeString{} << name
                         << " is not contained in the std::unordered_map "
                            "FunctionsOfTime");
    }
  }
};

template <size_t DerivOrder>
struct UpdateFunctionOfTime {
  static void apply(
      const gsl::not_null<std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
          functions_of_time,
      const double last_time_updated, const double next_expiration_time,
      const DataVector& control_signal, const std::string& name) noexcept {
    try {
      const auto translation = dynamic_cast<
          domain::FunctionsOfTime::PiecewisePolynomial<DerivOrder>*>(
          functions_of_time->at(name).get());

      if (translation == nullptr) {
        ERROR(
            "Failed dynamic_cast of FunctionOfTime for translation to a "
            "PiecewisePolynomial.");
      }
      translation->update(last_time_updated, control_signal,
                          next_expiration_time);

    } catch (const std::out_of_range& oor) {
      ERROR(MakeString{} << name << " is not in the functions of time");
    }
  }
};

namespace Actions {

struct UpdateControlSystems;

// This action runs on the ControlComponent and receives the horizon centers
// from ApparentHorizon finders. Each horizon center is stored in the DataBox as
// std::map<TimeStepId, std::array<double, 3>> and when the horizon at a
// particular time is sent the map is updated. When the horizon center for both
// horizons has been found at a particular time, the control system can perform
// a measurement at that time by calling the UpdateControlSystems action.
template <typename WhichHorizon>
struct ReceiveHorizonCenters {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<Tags::HorizonCenter<WhichHorizon>,
                                              DataBox>> = nullptr>
  static void apply(DataBox& box, Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, TimeStepId time_id,
                    std::array<double, 3> horizon_center) noexcept {
    db::mutate<Tags::HorizonCenter<WhichHorizon>>(
        make_not_null(&box),
        [&time_id, &horizon_center](
            const gsl::not_null<std::map<::TimeStepId, std::array<double, 3>>*>&
                horizon_centers) {
          (*horizon_centers)[time_id] = horizon_center;
        });

    using other_horizon =
        tmpl::lookup<typename Metavariables::OtherHorizon, WhichHorizon>;

    const auto& other_horizon_centers =
        db::get<Tags::HorizonCenter<other_horizon>>(box);

    if (other_horizon_centers.find(time_id) != other_horizon_centers.end()) {
      auto& this_proxy =
          Parallel::get_parallel_component<ControlComponent<Metavariables>>(
              cache);

      Parallel::simple_action<UpdateControlSystems>(this_proxy);
    }
    return;
  }
};

template <typename Target, typename ForwardList>
struct ForwardToControlSystem;

// Simply forwards the tags ForwardTags to the control component by calling
// ReceiveHorizonCenters. (This was intended to be more generic than just
// horizon centers at some point.)
template <typename Target, typename... ForwardTags>
struct ForwardToControlSystem<Target, tmpl::list<ForwardTags...>> {
  template <typename DbTags, typename Metavariables>
  static void apply(
      const db::DataBox<DbTags>& box,
      Parallel::GlobalCache<Metavariables>& cache,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    auto& control_proxy =
        Parallel::get_parallel_component<::ControlComponent<Metavariables>>(
            cache);

    Parallel::simple_action<ReceiveHorizonCenters<Target>>(
        control_proxy, temporal_id, db::get<ForwardTags>(box)...);
  }
};

struct UpdateControlSystems {
  template <
      typename ParallelComponent, typename DataBox, typename Metavariables,
      typename ArrayIndex,
      Requires<db::tag_is_retrievable_v<
          Tags::HorizonCenter<typename Metavariables::AhA>, DataBox>> = nullptr>
  static void apply(DataBox& box, Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/
                    ) noexcept {
    // This action currently deadlocks and possibly triggers quiescence
    // detection which causes the executable to exit. There's probably a logic
    // error in the extrapolation code.

    using control_systems = typename Metavariables::control_systems_list;
    static constexpr size_t DerivOrder = Metavariables::deriv_order;
    const auto controller = Controller<DerivOrder>{};

    static_assert(tmpl::size<control_systems>::value == 1,
                  "Only a single control system is currently supported");

    const auto alpha_m =
        db::get<::Tags::ExpirationDeltaTOverDampingTimescale>(box);

    tmpl::for_each<control_systems>([&box, &alpha_m, &cache,
                                     &controller](auto type_v) {
      using ControlSystemType = typename decltype(type_v)::type;
      const auto name = ControlSystemType::name();

      auto control_errors = db::apply<ControlSystemType>(box);

      db::mutate<Tags::HorizonCenter<typename Metavariables::AhA>,
                 Tags::HorizonCenter<typename Metavariables::AhB>>(
          make_not_null(&box),
          [](const gsl::not_null<std::map<
                 ::TimeStepId, std::array<double, 3>>*>& horizon_centers_a,
             const gsl::not_null<std::map<
                 ::TimeStepId, std::array<double, 3>>*>& horizon_centers_b) {
            horizon_centers_a->clear();
            horizon_centers_b->clear();
          });

      const auto& timescale_tuner = db::get<Tags::TimescaleTuner>(box);

      for (const auto& [time_id, control_error] : control_errors) {
        const auto time = time_id.substep_time().value();
        // measure the error
        db::mutate<Tags::Averager<DerivOrder>>(
            make_not_null(&box), [time, &control_error = control_error,
                                  &timescale_tuner](const auto averager_ptr) {
              averager_ptr->update(time, control_error,
                                   timescale_tuner.current_timescale());
            });
      }

      // last time
      const auto time_id = control_errors.rbegin()->first;
      const auto time = time_id.substep_time().value();
      const auto& f_of_ts = Parallel::get<domain::Tags::FunctionsOfTime>(cache);
      const double expiry_time = f_of_ts.at(name)->time_bounds()[1];

      // const bool is_expired = (time >= expiry_time);
      const auto& averager = db::get<Tags::Averager<DerivOrder>>(box);
      if (averager(time)) {
        const std::array<DataVector, DerivOrder + 1>& q_and_derivs =
            *(averager(time));
        // With extrapolation the time offset is the expiry time
        const double t_offset_of_qdot =
            expiry_time - averager.average_time(time);
        const double t_offset_of_q = averager.using_average_0th_deriv_of_q()
                                         ? t_offset_of_qdot
                                         : expiry_time - time;

        const DataVector control_signal =
            controller(timescale_tuner.current_timescale(), q_and_derivs,
                       t_offset_of_q, t_offset_of_qdot);

        const double next_expiration_time =
            expiry_time + alpha_m * min(timescale_tuner.current_timescale());

        Parallel::mutate<domain::Tags::FunctionsOfTime,
                         UpdateFunctionOfTime<DerivOrder>>(
            cache, expiry_time, next_expiration_time, control_signal, name);

        // Update timescales on this component
        db::mutate<Tags::TimescaleTuner>(
            make_not_null(&box),
            [&q_and_derivs](const auto timescale_tuner_ptr) {
              timescale_tuner_ptr->update_timescale(
                  {{q_and_derivs[0], q_and_derivs[1]}});
            });

      } else if (false) {
        // The idea behind this branch is to prolong the validity of the
        // functions of time if there is not enough data to update. WIth
        // extrapolation, the criteria for "not enough data" is unclear to me
        // since the control system would need to know that its not going to
        // receive any more data before the update time, hence I've put false as
        // a placeholder.
        const double next_expiration_time =
            time + alpha_m * min(timescale_tuner.current_timescale());

        Parallel::mutate<domain::Tags::FunctionsOfTime,
                         ProlongTimeOfValidity<DerivOrder>>(
            cache, next_expiration_time, name);
      }
    });
  }
};

template <typename InterpVars, size_t VolumeDim, typename... Tensors>
Variables<InterpVars> interpolate_vars(const size_t number_of_grid_points,
                                       const Tensors&... tensors) {
  Variables<InterpVars> interp_vars(number_of_grid_points);
  const auto copy_to_variables = [&interp_vars](const auto tensor_tag_v,
                                                const auto& tensor) noexcept {
    using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
    get<tensor_tag>(interp_vars) = tensor;
    return 0;
  };
  (void)copy_to_variables;  // GCC warns unused variable if Tensors is empty.
  expand_pack(copy_to_variables(tmpl::type_<Tensors>{}, tensors)...);
  return interp_vars;
}

struct SendHorizonData {
  template <typename Metavariables, typename DbTags, typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box, const ArrayIndex& array_index,
                    Parallel::GlobalCache<Metavariables>& cache) noexcept {
    const auto& time_id = db::get<Tags::TimeStepId>(box);

    // Code below is largely copied and pasted from the interpolator code

    using interpolator_source_vars =
        typename Metavariables::interpolator_source_vars;

    const auto& mesh =
        db::get<domain::Tags::Mesh<Metavariables::volume_dim>>(box);

    Variables<typename Metavariables::interpolator_source_vars> interp_vars(
        mesh.number_of_grid_points());

    tmpl::for_each<interpolator_source_vars>([&](const auto tag_v) noexcept {
      using tag = tmpl::type_from<decltype(tag_v)>;
      get<tag>(interp_vars) = db::get<tag>(box);
    });

    // Send volume data to the Interpolator, to trigger interpolation.
    auto& interpolator =
        *::Parallel::get_parallel_component<intrp::Interpolator<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<intrp::Actions::InterpolatorReceiveVolumeData>(
        interpolator, time_id,
        ElementId<Metavariables::volume_dim>(array_index), mesh, interp_vars);

    // Tell the interpolation target that it should interpolate.

    tmpl::for_each<typename Metavariables::interpolation_target_tags>(
        [&](const auto tag) {
          using interpolator_target_tag = tmpl::type_from<decltype(tag)>;

          auto& target =
              Parallel::get_parallel_component<intrp::InterpolationTarget<
                  Metavariables, interpolator_target_tag>>(cache);
          Parallel::simple_action<
              intrp::Actions::AddTemporalIdsToInterpolationTarget<
                  interpolator_target_tag>>(target,
                                            std::vector<TimeStepId>{time_id});
        });
  }
};

// Update the time dependent quantities.

/// \ingroup ActionsGroup
/// \ingroup ControlSystemGroup
/// \brief Send data for control system, and if functions of time are invalid
/// then suspend the algorithm.
///
/// This action combines the checking of the functions of time with the sending
/// of the data. These actions _should_ be separate, but this design predates
/// the use of extrapolation when they validity and sending were tighter
/// coupled.
template <typename WhatToSend>
struct CheckValidityOfFunctionsOfTimeAndSendControlData {
  template <typename ParallelComponent, typename... InboxTags, typename DbTags,
            typename Metavariables, typename ActionList, typename ArrayIndex>
  static std::tuple<db::DataBox<DbTags>&&, Parallel::AlgorithmExecution> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& this_specific_element_component_proxy =
        ::Parallel::get_parallel_component<
            typename Metavariables::gh_dg_element_array>(cache)[array_index];
    const auto& time_id = db::get<Tags::TimeStepId>(box);
    // Measurement and updating the control system should only occur
    // at slab boundaries. TODO: Is this the right thing to do?
    if (not(time_id.is_at_slab_boundary())) {
      return std::forward_as_tuple(std::move(box),
                                   Parallel::AlgorithmExecution::Continue);
    }

    const auto& time = time_id.substep_time().value();
    using control_systems = typename Metavariables::control_systems_list;
    static_assert(tmpl::size<control_systems>::value == 1, "");
    const std::string cs_name = tmpl::at_c<control_systems, 0>::name();

    const auto& measurement_timescales =
        db::get<Tags::MeasurementTimescales>(box);
    auto& last_measurement_times = db::get<Tags::LastMeasuredTimescales>(box);
    double measurement_timescale, last_measured_time;
    try {
      measurement_timescale = measurement_timescales.at(cs_name);
      last_measured_time = last_measurement_times.at(cs_name);
    } catch (const std::out_of_range& oor) {
      ERROR(MakeString{} << cs_name << " is not in the functions of time");
    }

    bool is_ready =
        Parallel::mutable_cache_item_is_ready<domain::Tags::FunctionsOfTime>(
            cache,
            [&](const std::unordered_map<
                std::string,
                std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                    functions_of_time) -> std::unique_ptr<Parallel::Callback> {
              for (const auto& [name, f_of_t] : functions_of_time) {
                const double expiration_time = f_of_t->time_bounds()[1];
                if (time >= expiration_time) {
                  return std::unique_ptr<Parallel::Callback>(
                      new Parallel::PerformAlgorithmCallback(
                          this_specific_element_component_proxy));
                }
              }
              return std::unique_ptr<Parallel::Callback>{};
            });

    if (not(is_ready)) {
      return std::forward_as_tuple(std::move(box),
                                   Parallel::AlgorithmExecution::Retry);
    }

    bool measure_this_step =
        (time >= last_measured_time + measurement_timescale);
    if (measure_this_step) {
      db::mutate<Tags::LastMeasuredTimescales>(
          make_not_null(&box), [&time, &cs_name](const auto time_ptr) noexcept {
            try {
              time_ptr->at(cs_name) = time;
            } catch (const std::out_of_range& oor) {
              ERROR("Tried to access invalid name\n");
            }
          });
      WhatToSend::apply(box, array_index, cache);
    }
    return std::forward_as_tuple(std::move(box),
                                 Parallel::AlgorithmExecution::Continue);
  }
};
}  // namespace Actions
