// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <pup.h>
#include <utility>

#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"

namespace control_system::size {
/*!
 * \brief A struct for holding the measurements needed to reconstruct control
 * errors for the `control_system::Systems::Size` control system.
 */
struct StateHistory {
  StateHistory();

  /// \brief Only keep `num_times_to_store` entries in the state_history
  StateHistory(size_t num_times_to_store);

  /*!
   * \brief Store the inputs used to compute each
   * `control_system::size::State`'s control error.
   *
   * \param time Time of the stored inputs
   * \param control_error_args `control_system::size::ControlErrorArgs`
   */
  void store(double time, const ControlErrorArgs& control_error_args);

  /*!
   * \brief Reconstruct the stored control errors for the state and target in
   * the current `info`.
   *
   * Reconstructing the errors when they are requested ensures target-dependent
   * states use the target associated with the current state transition, rather
   * than a target that belonged to a different state when the measurement was
   * stored.
   *
   * \param info Current `control_system::size::Info`
   * \return std::deque<std::pair<double, double>> The `std::pair` holds
   * the time and control error, respectively. The `std::deque` is ordered with
   * earlier times at the "front" and later times at the "back". This is to make
   * iteration over the deque easier as we typically want to start with earlier
   * times.
   */
  std::deque<std::pair<double, double>> state_history(const Info& info) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  friend bool operator==(const StateHistory& lhs, const StateHistory& rhs);
  friend bool operator!=(const StateHistory& lhs, const StateHistory& rhs);

 private:
  size_t num_times_to_store_{};
  std::deque<std::pair<double, ControlErrorArgs>> stored_control_error_args_{};
};
}  // namespace control_system::size
