// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace gh {
/// Options for an explicitly experimental source-only numerical method.
struct RadialSourceProjectionOptions {
  struct EvaluationPoints {
    using type = size_t;
    static constexpr Options::String help = "Temporary radial points:32 or40.";
  };
  struct BlockIds {
    using type = std::vector<size_t>;
    static constexpr Options::String help = "Shell blocks on which to apply.";
  };
  struct PreserveInnerSource {
    using type = bool;
    static constexpr Options::String help =
        "Preserve source increment value and derivative at a lower external "
        "radial endpoint.";
  };
  using options = tmpl::list<EvaluationPoints, BlockIds, PreserveInnerSource>;
  static constexpr Options::String help =
      "Experimental radial local DH source projection. Original principal, "
      "damping and DG boundary operators are retained. Requires gamma1=-1.";

  RadialSourceProjectionOptions() = default;
  RadialSourceProjectionOptions(size_t points, std::vector<size_t> blocks,
                                bool preserve_inner,
                                const Options::Context& context = {});
  void pup(PUP::er& p);
  size_t evaluation_points = 32;
  std::vector<size_t> block_ids{};
  bool preserve_inner_source = true;
};

namespace OptionTags {
struct RadialSourceProjection {
  using type =
      Options::Auto<RadialSourceProjectionOptions, Options::AutoLabel::None>;
  using group = gh::OptionTags::Group;
  static constexpr Options::String help =
      "None disables the experimental radial local-source projection.";
};
}  // namespace OptionTags

namespace Tags {
struct RadialSourceProjection : db::SimpleTag {
  using type = std::optional<RadialSourceProjectionOptions>;
  using option_tags = tmpl::list<gh::OptionTags::RadialSourceProjection>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& settings) { return settings; }
};
}  // namespace Tags
}  // namespace gh
