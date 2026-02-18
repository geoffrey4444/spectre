// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <pup.h>
#include <string>

#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::bbh::Triggers {
/*!
 * \brief Dense trigger that schedules periodic checks for whether to terminate
 * a binary-black-hole simulation from constraint-threshold criteria.
 *
 * \details The next check time is advanced by the option
 * `gh::bbh::Tags::ConstraintCheckInterval`.
 */
class ConstraintCheck : public DenseTrigger {
 public:
  /// \cond
  ConstraintCheck() = default;
  explicit ConstraintCheck(CkMigrateMessage* msg) : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ConstraintCheck);  // NOLINT
  /// \endcond

  static constexpr Options::String help{
      "Trigger for periodic BBH constraint-threshold checks."};
  static std::string name() { return "BbhConstraintCheck"; }
  using options = tmpl::list<>;

  using is_triggered_return_tags = tmpl::list<>;
  using is_triggered_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<bool> is_triggered(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const Component* /*component*/) const {
    return true;
  }

  using next_check_time_return_tags = tmpl::list<>;
  using next_check_time_argument_tags = tmpl::list<::Tags::Time>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<double> next_check_time(
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const Component* /*component*/,
      double time) const {
    const double interval =
        Parallel::get<gh::bbh::Tags::ConstraintCheckInterval>(cache);
    ASSERT(interval > 0.0, "ConstraintCheckInterval must be positive.");
    return time + interval;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;
};
}  // namespace gh::bbh::Triggers
