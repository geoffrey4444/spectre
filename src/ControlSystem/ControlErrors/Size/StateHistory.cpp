// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/StateHistory.hpp"

#include <deque>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace control_system::size {
void ControlErrorArgs::pup(PUP::er& p) {
  p | min_char_speed;
  p | control_error_delta_r;
  p | control_error_delta_r_outward;
  p | avg_distorted_normal_dot_unit_coord_vector;
  p | time_deriv_of_lambda_00;
}

StateHistory::StateHistory() = default;

StateHistory::StateHistory(const size_t num_times_to_store)
    : num_times_to_store_(num_times_to_store) {}

void StateHistory::store(const double time,
                         const ControlErrorArgs& control_error_args) {
  stored_control_error_args_.emplace_back(time, control_error_args);
  while (stored_control_error_args_.size() > num_times_to_store_) {
    stored_control_error_args_.pop_front();
  }
}

std::deque<std::pair<double, double>> StateHistory::state_history(
    const Info& info) const {
  std::deque<std::pair<double, double>> result{};
  for (const auto& [time, control_error_args] : stored_control_error_args_) {
    result.emplace_back(time,
                        info.state->control_error(info, control_error_args));
  }
  return result;
}

void StateHistory::pup(PUP::er& p) {
  p | num_times_to_store_;
  p | stored_control_error_args_;
}

bool operator==(const StateHistory& lhs, const StateHistory& rhs) {
  return lhs.num_times_to_store_ == rhs.num_times_to_store_ and
         lhs.stored_control_error_args_ == rhs.stored_control_error_args_;
}

bool operator!=(const StateHistory& lhs, const StateHistory& rhs) {
  return not(lhs == rhs);
}
}  // namespace control_system::size
