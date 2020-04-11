// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/optional.hpp>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace evolution {
namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \brief Compute and add the source term modification for moving meshes
 *
 * Adds to the time derivative *not* the source terms because some systems do
 * not have source terms and so we optimize for that. The term being added to
 * the time derivative is:
 *
 * \f{align}{
 *  -u_\alpha \partial_i v^i_g,
 * \f}
 *
 * where \f$u_\alpha\f$ are the evolved variables and \f$v^i_g\f$ is the
 * velocity of the mesh.
 *
 * Uses:
 * - DataBox:
 *   - `System::variables_tags`
 *   - `domain::Tags::DivMeshVelocity`
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: `Tags::dt<system::variable_tags>`
 */
struct AddMeshVelocityNonconservative {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {  // NOLINT const
    using gradients_tags = typename Metavariables::system::gradients_tags;
    const auto& mesh_velocity =
        db::get<::domain::Tags::MeshVelocity<Metavariables::volume_dim>>(box);

    if (static_cast<bool>(mesh_velocity)) {
      tmpl::for_each<gradients_tags>([&box, &mesh_velocity](
                                         auto gradient_tag_v) noexcept {
        using gradient_tag = typename decltype(gradient_tag_v)::type;
        using dt_gradient_tag = db::add_tag_prefix<::Tags::dt, gradient_tag>;
        using deriv_tag =
            db::add_tag_prefix<::Tags::deriv, gradient_tag,
                               tmpl::size_t<Metavariables::volume_dim>,
                               Frame::Inertial>;

        db::mutate<dt_gradient_tag>(
            make_not_null(&box),
            [](const auto dt_var_ptr, const auto& deriv_tensor,
               const boost::optional<tnsr::I<
                   DataVector, Metavariables::volume_dim, Frame::Inertial>>&
                   grid_velocity) noexcept {
              for (size_t storage_index = 0;
                   storage_index < deriv_tensor.size(); ++storage_index) {
                const auto deriv_tensor_index =
                    deriv_tensor.get_tensor_index(storage_index);
                const auto tensor_index =
                    all_but_specified_element_of(deriv_tensor_index, 0);
                const size_t deriv_index = gsl::at(deriv_tensor_index, 0);
                dt_var_ptr->get(tensor_index) +=
                    grid_velocity->get(deriv_index) *
                    deriv_tensor[storage_index];
              }
            },
            db::get<deriv_tag>(box), mesh_velocity);
      });
    }
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace evolution
