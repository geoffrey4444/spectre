// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::bbh {
namespace OptionTags {
/// BBH inspiral-completion controls that are read from input options.
struct CompletionCriteria {
  static std::string name() { return "BbhCompletionCriteria"; }
  static constexpr Options::String help =
      "Options controlling inspiral termination based on global BBH criteria.";
};

struct MinCommonHorizonSuccessesBeforeChecks {
  using type = size_t;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "Do not evaluate binary-black-hole completion checks before this many "
      "successful common-horizon finds.";
};

struct MaxCommonHorizonSuccesses {
  using type = size_t;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "When successful common-horizon finds reach this count in a "
      "binary-black-hole simulation, request completion.";
};

struct GaugeConstraintLinfThreshold {
  using type = double;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "When the reduced Linf norm of the gauge constraint is greater than or "
      "equal to this threshold in a binary-black-hole simulation, request "
      "completion.";
};

struct ThreeIndexConstraintLinfThreshold {
  using type = double;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "When the reduced Linf norm of the three-index constraint is greater "
      "than or equal to this threshold in a binary-black-hole simulation, "
      "request completion.";
};

struct CommonHorizonLMaxThreshold {
  using type = size_t;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "When the common-horizon LMax is less than or equal to this value in a "
      "binary-black-hole simulation, request completion (after the minimum "
      "common-horizon success count is reached).";
};

struct ConstraintCheckInterval {
  using type = double;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "Dense-trigger interval for checking reduced global constraint criteria "
      "that can request binary-black-hole completion.";
};

struct ConstraintCheckVerbose {
  using type = bool;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "Whether to print reduced binary-black-hole constraint norms at each "
      "constraint-threshold check.";
};
}  // namespace OptionTags

namespace Tags {
/// Minimum number of successful common-horizon finds required before
/// completion checks are evaluated.
struct MinCommonHorizonSuccessesBeforeChecks : db::SimpleTag {
  using type = size_t;
  using option_tags =
      tmpl::list<OptionTags::MinCommonHorizonSuccessesBeforeChecks>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

/// Number of successful common-horizon finds that triggers completion.
/// Controlled by `gh::bbh::OptionTags::MaxCommonHorizonSuccesses`.
struct MaxCommonHorizonSuccesses : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::MaxCommonHorizonSuccesses>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

/// Threshold for requesting completion from the reduced gauge-constraint Linf
/// criterion:
/// when Linf(gauge constraint) >= this threshold in a binary-black-hole
/// simulation, request completion.
/// Controlled by `gh::bbh::OptionTags::GaugeConstraintLinfThreshold`.
struct GaugeConstraintLinfThreshold : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::GaugeConstraintLinfThreshold>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

/// Threshold for requesting completion from the reduced three-index-constraint
/// Linf criterion:
/// when Linf(three-index constraint) >= this threshold in a binary-black-hole
/// simulation, request completion.
/// Controlled by `gh::bbh::OptionTags::ThreeIndexConstraintLinfThreshold`.
struct ThreeIndexConstraintLinfThreshold : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::ThreeIndexConstraintLinfThreshold>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

/// Common-horizon `LMax` threshold criterion for requesting completion
/// (after the minimum common-horizon success count is reached):
/// when common-horizon `LMax` <= this threshold in a binary-black-hole
/// simulation, request completion.
/// Controlled by `gh::bbh::OptionTags::CommonHorizonLMaxThreshold`.
struct CommonHorizonLMaxThreshold : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::CommonHorizonLMaxThreshold>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

/// Dense-trigger interval used to check reduced constraint criteria for
/// binary-black-hole termination.
/// Controlled by `gh::bbh::OptionTags::ConstraintCheckInterval`.
struct ConstraintCheckInterval : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::ConstraintCheckInterval>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

/// Verbosity control for reduced constraint checks.
/// Controlled by `gh::bbh::OptionTags::ConstraintCheckVerbose`.
struct ConstraintCheckVerbose : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::ConstraintCheckVerbose>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

/// Latch indicating that the reduced gauge-constraint criterion was met.
struct GaugeConstraintExceeded : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return false; }
};

/// Latch indicating that the reduced three-index-constraint criterion was met.
struct ThreeIndexConstraintExceeded : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return false; }
};

/// Latch indicating that a successful common-horizon find satisfied the `LMax`
/// criterion.
struct CommonHorizonLMaxBelowOrEqualThreshold : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return false; }
};

/// Count of successful common-horizon finds.
struct CommonHorizonSuccessCount : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return 0; }
};

/// Latch used by phase control to checkpoint-and-exit the binary-black-hole
/// simulation when completion is requested by any criterion.
struct CompletionRequested : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return false; }
};

/// Element-local latch mirrored from the BBH completion singleton and used by
/// phase control to request checkpoint-and-exit.
struct ElementCompletionRequested : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return false; }
};
}  // namespace Tags

namespace Mutators {
/// Sets `gh::bbh::Tags::GaugeConstraintExceeded` to true.
struct SetGaugeConstraintExceeded {
  static void apply(const gsl::not_null<bool*> gauge_constraint_exceeded) {
    *gauge_constraint_exceeded = true;
  }
};

/// Sets `gh::bbh::Tags::ThreeIndexConstraintExceeded` to true.
struct SetThreeIndexConstraintExceeded {
  static void apply(
      const gsl::not_null<bool*> three_index_constraint_exceeded) {
    *three_index_constraint_exceeded = true;
  }
};

/// Sets `gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold` to true.
struct SetCommonHorizonLMaxBelowOrEqualThreshold {
  static void apply(const gsl::not_null<bool*> lmax_below_or_equal_threshold) {
    *lmax_below_or_equal_threshold = true;
  }
};

/// Increments `gh::bbh::Tags::CommonHorizonSuccessCount`.
struct IncrementCommonHorizonSuccessCount {
  static void apply(const gsl::not_null<size_t*> common_horizon_success_count) {
    ++(*common_horizon_success_count);
  }
};

/// Sets `gh::bbh::Tags::CompletionRequested` to true.
struct SetCompletionRequested {
  static void apply(const gsl::not_null<bool*> completion_requested) {
    *completion_requested = true;
  }
};
}  // namespace Mutators
}  // namespace gh::bbh
