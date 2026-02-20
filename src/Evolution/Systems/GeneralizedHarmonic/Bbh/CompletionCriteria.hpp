// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::bbh {
namespace OptionTags {
struct CompletionCriteria {
  static std::string name() { return "BbhCompletionCriteria"; }
  static constexpr Options::String help =
      "Options controlling inspiral termination based on global BBH criteria.";
};

struct MinCommonHorizonSuccessesBeforeChecks {
  using type = size_t;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "Do not trigger completion checks before this many successful AhC finds.";
};

struct MaxCommonHorizonSuccesses {
  using type = size_t;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "Trigger completion once successful AhC finds reach this count.";
};

struct GaugeConstraintLinfThreshold {
  using type = double;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "Threshold for Linf(GaugeConstraint) completion criterion.";
};

struct ThreeIndexConstraintLinfThreshold {
  using type = double;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "Threshold for Linf(ThreeIndexConstraint) completion criterion.";
};

struct CommonHorizonLMaxThreshold {
  using type = size_t;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "Trigger completion if AhC Lmax is less than or equal to this value.";
};

struct ConstraintCheckInterval {
  using type = double;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "Interval for evaluating global constraint completion criteria.";
};

struct ConstraintCheckVerbose {
  using type = bool;
  using group = CompletionCriteria;
  static constexpr Options::String help =
      "Whether to print reduced BBH constraint norms at each check.";
};
}  // namespace OptionTags

namespace Tags {
struct MinCommonHorizonSuccessesBeforeChecks : db::SimpleTag {
  using type = size_t;
  using option_tags =
      tmpl::list<OptionTags::MinCommonHorizonSuccessesBeforeChecks>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

struct MaxCommonHorizonSuccesses : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::MaxCommonHorizonSuccesses>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

struct GaugeConstraintLinfThreshold : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::GaugeConstraintLinfThreshold>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

struct ThreeIndexConstraintLinfThreshold : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::ThreeIndexConstraintLinfThreshold>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

struct CommonHorizonLMaxThreshold : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::CommonHorizonLMaxThreshold>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

struct ConstraintCheckInterval : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::ConstraintCheckInterval>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

struct ConstraintCheckVerbose : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::ConstraintCheckVerbose>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type value) { return value; }
};

struct GaugeConstraintExceeded : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return false; }
};

struct ThreeIndexConstraintExceeded : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return false; }
};

struct CommonHorizonLMaxBelowOrEqualThreshold : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return false; }
};

struct CommonHorizonSuccessCount : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return 0; }
};

struct MaxCommonHorizonSuccessesReached : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return false; }
};

struct CompletionRequested : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return false; }
};

struct StopSlabNumber : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() {
    return std::numeric_limits<size_t>::max();
  }
};
}  // namespace Tags

namespace Mutators {
struct SetGaugeConstraintExceeded {
  static void apply(const gsl::not_null<bool*> gauge_constraint_exceeded) {
    *gauge_constraint_exceeded = true;
  }
};

struct SetThreeIndexConstraintExceeded {
  static void apply(
      const gsl::not_null<bool*> three_index_constraint_exceeded) {
    *three_index_constraint_exceeded = true;
  }
};

struct SetCommonHorizonLMaxBelowOrEqualThreshold {
  static void apply(const gsl::not_null<bool*> lmax_below_or_equal_threshold) {
    *lmax_below_or_equal_threshold = true;
  }
};

struct IncrementCommonHorizonSuccessCount {
  static void apply(const gsl::not_null<size_t*> common_horizon_success_count) {
    ++(*common_horizon_success_count);
  }
};

struct SetMaxCommonHorizonSuccessesReached {
  static void apply(
      const gsl::not_null<bool*> max_common_horizon_successes_reached) {
    *max_common_horizon_successes_reached = true;
  }
};

struct SetCompletionRequested {
  static void apply(const gsl::not_null<bool*> completion_requested) {
    *completion_requested = true;
  }
};

struct SetStopSlabNumberIfUnset {
  static void apply(const gsl::not_null<size_t*> stop_slab_number,
                    const size_t new_stop_slab_number) {
    if (*stop_slab_number == std::numeric_limits<size_t>::max()) {
      *stop_slab_number = new_stop_slab_number;
    }
  }
};
}  // namespace Mutators
}  // namespace gh::bbh
