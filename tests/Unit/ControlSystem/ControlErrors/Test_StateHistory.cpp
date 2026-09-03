// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <deque>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRDriftInward.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRDriftOutward.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRNoDrift.hpp"
#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/Initial.hpp"
#include "ControlSystem/ControlErrors/Size/RegisterDerivedWithCharm.hpp"
#include "ControlSystem/ControlErrors/Size/StateHistory.hpp"
#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace control_system::size {
namespace {
void test_target_change() {
  constexpr double old_ah_speed_target = 0.001614715321563;
  constexpr double inward_drift_target = 0.0003;
  Info info{std::make_unique<States::AhSpeed>(),
            1.0,
            old_ah_speed_target,
            0.0,
            std::nullopt,
            false};
  const ControlErrorArgs first_args{0.2, 0.0, 0.4, -1.0, 0.5};
  const ControlErrorArgs second_args{0.3, 0.0, 0.5, -2.0, 0.6};

  StateHistory state_history{3};
  state_history.store(0.0, first_args);
  state_history.store(1.0, second_args);

  // A transition to inward drift changes the meaning and value of the shared
  // target. Historical errors must use the new inward-drift target, not the
  // AhSpeed target that was active when the measurements were stored.
  info.state = std::make_unique<States::DeltaRDriftInward>();
  info.target_char_speed = inward_drift_target;
  const auto inward_history = state_history.state_history(info);
  REQUIRE(inward_history.size() == 2);
  CHECK(inward_history[0].second == approx(inward_drift_target));
  CHECK(inward_history[1].second == approx(inward_drift_target));

  // Replaying a constant destination-state error must not create the large,
  // artificial derivative seen in the production transition.
  Averager<1> averager{0.25, true};
  const DataVector timescale{1, 1.0};
  for (const auto& [time, control_error] : inward_history) {
    averager.update(time, DataVector{1, control_error}, timescale);
  }
  const ControlErrorArgs current_args{0.4, 0.0, 0.6, -1.5, 0.7};
  averager.update(2.0,
                  DataVector{1, info.state->control_error(info, current_args)},
                  timescale);
  const auto averaged_values = averager(2.0);
  REQUIRE(averaged_values.has_value());
  CHECK(averaged_values->at(0)[0] == approx(inward_drift_target));
  CHECK(averaged_values->at(1)[0] == approx(0.0).margin(1.0e-14));

  // Check the reverse transition as well. Reconstructing the AhSpeed errors
  // must combine the new target with each historical measurement's own speed
  // and surface normal.
  StateHistory reverse_state_history{3};
  reverse_state_history.store(0.0, first_args);
  reverse_state_history.store(1.0, second_args);
  info.state = std::make_unique<States::AhSpeed>();
  info.target_char_speed = 0.4;
  const auto ah_speed_history = reverse_state_history.state_history(info);
  REQUIRE(ah_speed_history.size() == 2);
  const double y00 = sqrt(0.25 / M_PI);
  CHECK(ah_speed_history[0].second == approx((0.4 - 0.2) / (-y00)));
  CHECK(ah_speed_history[1].second == approx((0.4 - 0.3) / (-2.0 * y00)));
}

void test_state_history(const size_t num_times_to_store) {
  CAPTURE(num_times_to_store);
  Info info{
      std::make_unique<States::Initial>(), 1.0, 1.0, 1.0, std::nullopt, false};
  ControlErrorArgs control_error_args{1.0, 1.0, 1.0, 1.0, 1.0};

  StateHistory state_history{num_times_to_store};
  std::vector<std::unique_ptr<State>> states{};
  states.emplace_back(std::make_unique<States::Initial>());
  states.emplace_back(std::make_unique<States::AhSpeed>());
  states.emplace_back(std::make_unique<States::DeltaR>());
  states.emplace_back(std::make_unique<States::DeltaRDriftInward>());
  states.emplace_back(std::make_unique<States::DeltaRNoDrift>());
  states.emplace_back(std::make_unique<States::DeltaRDriftOutward>());

  // Test that as we fill up the history, we have the expected number of stored
  // entries and that they are the correct values, for each state
  for (size_t i = 0; i < num_times_to_store; i++) {
    const auto time = static_cast<double>(i);
    state_history.store(time, control_error_args);

    for (const auto& state : states) {
      CAPTURE(state->number());
      info.state = state->get_clone();
      const auto history = state_history.state_history(info);
      CHECK(history.size() == i + 1);
      for (size_t j = 0; j < history.size(); j++) {
        const auto& [stored_time, control_error] = history[j];
        CHECK(static_cast<double>(j) == stored_time);
        // These are hand calculated from the above parameters Info and
        // ControlErrorArgs.
        switch (state->number()) {
          case 0:
            CHECK(control_error == 0.0);
            break;
          case 1:
            CHECK(control_error == 0.0);
            break;
          case 2:
            CHECK(control_error == 1.0);
            break;
          case 3:
            CHECK(control_error == 2.0);
            break;
          case 4:
            CHECK(control_error == 1.0);
            break;
          case 5:
            CHECK(control_error == 1.0);
            break;
          default:
            ERROR("Unknown state: " << state->number());
        }
      }
    }
  }

  // Test that trying to store one more value causes us to pop off the initial
  // (first) value and add this new one to the end (last), while keeping the
  // total number of entries at num_times_to_store
  state_history.store(static_cast<double>(num_times_to_store),
                      control_error_args);
  for (const auto& state : states) {
    info.state = state->get_clone();
    const auto history = state_history.state_history(info);
    CHECK(history.size() == num_times_to_store);
    CHECK(history.front().first == 1.0);
    CHECK(history.back().first == static_cast<double>(num_times_to_store));
  }
  CHECK(serialize_and_deserialize(state_history) == state_history);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.ControlErrors.StateHistory",
                  "[Domain][Unit]") {
  control_system::size::register_derived_with_charm();
  for (size_t num_times = 1; num_times < 5; num_times++) {
    test_state_history(num_times);
  }
  test_target_change();
}
}  // namespace control_system::size
