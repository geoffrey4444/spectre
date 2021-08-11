// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iostream>

#include "ControlSystem/FunctionOfTimeUpdater.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"

namespace Tags {
struct ControlError;
struct Time;
}  // namespace Tags

template <size_t>
class Averager;
class TimescaleTuner;
template <size_t>
class Controller;

namespace OptionTags {

struct ControlSystemGroup {
  static std::string name() noexcept { return "ControlSystem"; }
  static constexpr Options::String help = "Control system";
};

template <size_t DerivOrder>
struct Averager {
  using type = ::Averager<DerivOrder>;
  static constexpr Options::String help = "Averager";
  using group = OptionTags::ControlSystemGroup;
};

struct TimescaleTuner {
  using type = ::TimescaleTuner;
  static constexpr Options::String help = {"TimescaleTuner"};
  using group = OptionTags::ControlSystemGroup;
};

// This is alpha_m in Dan's paper
struct MeasurementTimeScaleOverExpirationDeltaT {
  using type = double;
  static constexpr Options::String help = {
      "How often to measure control error as a fraction of the expiration "
      "timescale of the function of time"};
  using group = OptionTags::ControlSystemGroup;
};

// This is alpha_d in Dan's paper
struct ExpirationDeltaTOverDampingTimescale {
  using type = double;
  static constexpr Options::String help = {
      "How long function of time expiration should be as a fraction of the "
      "damping timescale"};
  using group = OptionTags::ControlSystemGroup;
};
}  // namespace OptionTags

namespace Tags {

struct MeasurementTimeScaleOverExpirationDeltaT : db::SimpleTag {
  using type = double;
  using option_tags =
      tmpl::list<OptionTags::MeasurementTimeScaleOverExpirationDeltaT>;
  static constexpr bool pass_metavariables = false;
  static auto create_from_options(const double value) { return value; }
};

struct ExpirationDeltaTOverDampingTimescale : db::SimpleTag {
  using type = double;
  using option_tags =
      tmpl::list<OptionTags::ExpirationDeltaTOverDampingTimescale>;
  static constexpr bool pass_metavariables = false;
  static auto create_from_options(const double value) { return value; }
};

template <size_t DerivOrder>
struct Averager : db::SimpleTag {
  using type = ::Averager<DerivOrder>;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::Averager<DerivOrder>>;

  static auto create_from_options(
      const ::Averager<DerivOrder>& averager) noexcept {
    return averager;
  }
};

struct TimescaleTuner : db::SimpleTag {
  using type = ::TimescaleTuner;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::TimescaleTuner>;

  static auto create_from_options(
      const ::TimescaleTuner& timescale_tuner) noexcept {
    return timescale_tuner;
  }
};

// This is tau_m in Dan's paper for each control system
struct MeasurementTimescales : db::SimpleTag {
  using type = std::unordered_map<std::string, double>;
};

struct MeasurementTimescalesCompute : db::ComputeTag, MeasurementTimescales {
  using argument_tags =
      tmpl::list<domain::Tags::FunctionsOfTime,
                 ::Tags::MeasurementTimeScaleOverExpirationDeltaT>;
  using base = MeasurementTimescales;
  using return_type = base::type;

  static void function(
      const gsl::not_null<std::unordered_map<std::string, double>*> result,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const double measurement_timescale_over_expiration_delta_t) noexcept {
    for (const auto& [k, v] : functions_of_time) {
      result->emplace(k, measurement_timescale_over_expiration_delta_t *
                            (v->time_bounds()[1]- v->time_bounds()[0]));
    }
  }
};

// This is so each element can keep track of the last time it sent data
struct LastMeasuredTimescales : db::SimpleTag {
  using type = std::unordered_map<std::string, double>;
  static constexpr bool pass_metavariables = true;

  template <typename Metavariables>
  using option_tags =
      tmpl::list<::OptionTags::InitialTime,
                 domain::OptionTags::DomainCreator<Metavariables::volume_dim>>;

  template <typename Metavariables>
  static auto create_from_options(
      const double initial_time,
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator) {
    std::unordered_map<std::string, double> last_measured{};
    for (const auto& names_and_functions :
         domain_creator->functions_of_time()) {
      last_measured[names_and_functions.first] = initial_time;
    }
    return last_measured;
  }
};

template <typename Horizon>
struct HorizonCenter : db::SimpleTag {
  // This is a map from TimeStepId's to arrays so that the control system
  // can store multiple horizon centers at multiple times (Since the control
  // system may receive multiple horizon A's before any horizon B)
  using type = std::map<::TimeStepId, std::array<double, 3>>;
};

}  // namespace Tags
