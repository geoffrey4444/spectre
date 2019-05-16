// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"

namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Mutate the DataBox tags in `TagsList` according to the `data`.
 *
 * DataBox changes:
 * - Modifies:
 *   - All tags in `TagsList`
 */
template <typename TagsList>
struct SetData;

template <typename... Tags>
struct SetData<tmpl::list<Tags...>> {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex,
      Requires<cpp17::conjunction_v<tmpl::list_contains<DbTagsList, Tags>...>> =
          nullptr>
  static void apply(
      db::DataBox<DbTagsList>& box,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      tuples::tagged_tuple_from_typelist<tmpl::list<Tags...>> data) noexcept {
    tmpl::for_each<tmpl::list<Tags...>>([&box, &data ](auto tag_v) noexcept {
      using tag = tmpl::type_from<decltype(tag_v)>;
      db::mutate<tag>(
          make_not_null(&box), [&data](const gsl::not_null<db::item_type<tag>*>
                                           value) noexcept {
            *value = std::move(get<tag>(data));
          });
    });
  }
};

}  // namespace Actions
