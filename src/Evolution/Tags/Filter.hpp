// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "Options/String.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/StdHelpers.hpp"

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups the filtering configurations in the input file.
 */
struct FilteringGroup {
  static std::string name() { return "Filtering"; }
  static constexpr Options::String help = "Options for filtering";
};

/*!
 * \ingroup OptionTagsGroup
 * \brief The option tag that retrieves the parameters for the filter
 * from the input file
 */
template <typename FilterType>
struct Filter {
  static std::string name() { return pretty_type::name<FilterType>(); }
  static constexpr Options::String help = "Options for the filter";
  using type = FilterType;
  using group = FilteringGroup;
};
}  // namespace OptionTags

namespace Filters {
namespace Tags {
/*!
 * \brief The global cache tag for the filter
 *
 * Also checks if the specified blocks are actually in the domain.
 */
template <typename FilterType>
struct Filter : db::SimpleTag {
  using type = FilterType;
  template <typename Metavariables>
  using option_tags =
      tmpl::list<::OptionTags::Filter<FilterType>,
                 domain::OptionTags::DomainCreator<Metavariables::volume_dim>>;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables>
  static FilterType create_from_options(
      FilterType filter,
      const std::unique_ptr<DomainCreator<Metavariables::volume_dim>>&
          domain_creator) {
    const auto& block_names = domain_creator->block_names();
    const auto& block_groups = domain_creator->block_groups();
    // Filters derived from Filters::Filter<Dim, TagList> use
    // set_blocks_to_filter to resolve string names to block IDs. Legacy filters
    // (e.g. Filters::Exponential) still expose blocks_to_filter() as strings
    // and are validated directly here.
    if constexpr (requires {
                    filter.set_blocks_to_filter(
                        std::declval<const std::vector<std::string>&>(),
                        std::declval<const std::unordered_map<
                            std::string, std::unordered_set<std::string>>&>());
                  }) {
      filter.set_blocks_to_filter(block_names, block_groups);
    } else {
      const auto& blocks_to_filter = filter.blocks_to_filter();
      if (blocks_to_filter.has_value()) {
        if (block_names.empty()) {
          ERROR(
              "The domain chosen doesn't use block names, but the Filter tag "
              "has specified block names to use.");
        }
        for (const auto& block_to_filter : blocks_to_filter.value()) {
          const auto block_name_iter = alg::find(block_names, block_to_filter);
          if (block_name_iter == block_names.end() and
              block_groups.count(block_to_filter) == 0) {
            ERROR("Specified block (group) name '"
                  << block_to_filter
                  << "' is not a block name or a block "
                     "group. Existing blocks are:\n"
                  << block_names << "\nExisting block groups are:\n"
                  << keys_of(block_groups));
          }
        }
      }
    }
    return filter;
  }
};
}  // namespace Tags
}  // namespace Filters
