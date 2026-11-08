// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "Time/SelfStart.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace tuples {
template <typename... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace dg::Actions {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Applies the element's spectral volume filter to the evolved variables.
 *
 * Retrieves the per-element filter from `Filters::Tags::SpectralFilter` from
 * the DataBox and applies the volume filter.
 *
 * The filter is skipped when:
 * - the filter is `Filters::None` (explicit no-op), or
 * - `apply_volume_filter_on_substep()` returns `false` and the updated
 *   variables are at an intermediate substep, or
 * - both `apply_volume_filter_on_substep()` and
 *   `apply_volume_filter_on_this_step(step_number)` return `false` at a
 *   completed step.
 *
 * The `TimeStepIdTag` template parameter identifies the time of the variables
 * being filtered. Its default, `Tags::Next<Tags::TimeStepId>`, is appropriate
 * after `UpdateU` and before `AdvanceTime`. When filtering before `UpdateU`,
 * set it to `Tags::TimeStepId` instead. In the default mode, unused self-start
 * steps are skipped because `UpdateU` did not update the variables.
 *
 * The grid-to-inertial Jacobian and its inverse are supplied only when
 * `Filters::Filter::need_jacobians()` returns `true`; otherwise `std::nullopt`
 * is passed for both arguments. The cached DataBox Jacobians are used when
 * `TimeStepIdTag` has the same physical time as `Tags::Time`, or when the
 * map's Jacobians are time independent. Otherwise, the Jacobians are evaluated
 * at `TimeStepIdTag::substep_time()`. The action waits for the map's functions
 * of time to be ready before evaluating new Jacobians and filtering with them.
 *
 * \note Currently the check for whether to filter on every `N` steps is done
 * relative to the start of the current Slab. This means that for GTS,
 * independent of the value of `N` for every `N` steps, every step has a
 * filter applied since GTS has one step per slab.
 *
 * Uses:
 * - DataBox:
 *   - `Filters::Tags::SpectralFilter<Dim, TagList>`
 *   - `Tags::StepNumberWithinSlab`
 *   - `TimeStepIdTag`
 *   - `Tags::TimeStepId` (in the default post-`UpdateU` mode)
 *   - `Tags::Time` (only when `need_jacobians()` is `true`)
 *   - `domain::Tags::Mesh<Dim>`
 *   - `domain::Tags::InverseJacobian<Dim, Frame::Grid, Frame::Inertial>`
 *     (only when `need_jacobians()` is `true`)
 *   - `domain::Tags::Jacobian<Dim, Frame::Grid, Frame::Inertial>`
 *     (only when `need_jacobians()` is `true`)
 *   - `domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
 *     Frame::Inertial>` (when Jacobians are needed at a different time)
 *   - `domain::Tags::Coordinates<Dim, Frame::Grid>` and
 *     `domain::Tags::FunctionsOfTime` (when evaluating new Jacobians)
 * - GlobalCache:
 *   - `domain::Tags::FunctionsOfTime` (when evaluating new Jacobians)
 * - DataBox changes:
 *   - Modifies: `Metavariables::system::variables_tag`
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 */
template <size_t Dim, typename TagList,
          typename TimeStepIdTag = ::Tags::Next<::Tags::TimeStepId>>
struct SpectralFilter {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const component) {
    const auto& filter =
        db::get<Filters::Tags::SpectralFilter<Dim, TagList>>(box);

    const Mesh<Dim>& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    if (not filter.supports_mesh(mesh)) {
      ERROR("The filter '"
            << filter.name() << "' on element "
            << db::get<domain::Tags::Element<Dim>>(box).id() << " with mesh "
            << mesh
            << " does not support that mesh. This is a bug in initialization "
               "or if AMR projection during h-refinement incorrectly adjusted "
               "the filter.");
    }

    // None is explicitly a no-op; skip immediately.
    if (dynamic_cast<const Filters::None<Dim, TagList>*>(&filter) != nullptr) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    if constexpr (std::is_same_v<TimeStepIdTag,
                                 ::Tags::Next<::Tags::TimeStepId>>) {
      if (SelfStart::step_unused(db::get<::Tags::TimeStepId>(box),
                                 db::get<TimeStepIdTag>(box))) {
        return {Parallel::AlgorithmExecution::Continue, std::nullopt};
      }
    }

    const auto step_number =
        static_cast<size_t>(db::get<::Tags::StepNumberWithinSlab>(box));
    if (not(filter.apply_volume_filter_on_substep() or
            (db::get<TimeStepIdTag>(box).substep() == 0 and
             filter.apply_volume_filter_on_this_step(step_number)))) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const size_t num_pts = mesh.number_of_grid_points();

    std::optional<
        InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>
        inv_jac{std::nullopt};
    std::optional<Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>> jac{
        std::nullopt};
    if (filter.need_jacobians()) {
      bool use_cached_jacobians = true;
      const double filter_time = db::get<TimeStepIdTag>(box).substep_time();
      if (filter_time != db::get<::Tags::Time>(box)) {
        using MapTag =
            domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                        Frame::Inertial>;
        if constexpr (db::tag_is_retrievable_v<MapTag, db::DataBox<DbTags>>) {
          const auto& map = db::get<MapTag>(box);
          if (map.jacobian_is_time_dependent() or
              map.inv_jacobian_is_time_dependent()) {
            if (not domain::functions_of_time_are_ready_algorithm_callback<
                    domain::Tags::FunctionsOfTime, Dim>(
                    cache, array_index, component, filter_time,
                    map.function_of_time_names())) {
              return {Parallel::AlgorithmExecution::Retry, std::nullopt};
            }
            const auto& grid_coords =
                db::get<domain::Tags::Coordinates<Dim, Frame::Grid>>(box);
            const auto& functions_of_time =
                db::get<domain::Tags::FunctionsOfTime>(box);
            jac = map.jacobian(grid_coords, filter_time, functions_of_time);
            // Shape maps compute their inverse by evaluating the Jacobian
            // again, so reuse the Jacobian already computed here.
            inv_jac = determinant_and_inverse(*jac).second;
            use_cached_jacobians = false;
          }
        } else {
          ERROR(
              "SpectralFilter: The variables and cached Jacobians have "
              "different times, but the DataBox has no grid-to-inertial "
              "coordinate map to evaluate the required Jacobians.");
        }
      }

      if (use_cached_jacobians) {
        if constexpr (db::tag_is_retrievable_v<
                          domain::Tags::InverseJacobian<Dim, Frame::Grid,
                                                        Frame::Inertial>,
                          db::DataBox<DbTags>>) {
          const auto& src = db::get<
              domain::Tags::InverseJacobian<Dim, Frame::Grid, Frame::Inertial>>(
              box);
          inv_jac =
              InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>{};
          for (size_t i = 0; i < src.size(); ++i) {
            make_const_view(make_not_null(&std::as_const((*inv_jac)[i])),
                            src[i], 0, num_pts);
          }
        } else {
          ERROR(
              "SpectralFilter: Missing "
              "domain::Tags::InverseJacobian<Dim, Grid, Inertial> in the "
              "DataBox. This should be present for both time-dependent and "
              "time-independent evolutions.");
        }
        if constexpr (db::tag_is_retrievable_v<
                          domain::Tags::Jacobian<Dim, Frame::Grid,
                                                 Frame::Inertial>,
                          db::DataBox<DbTags>>) {
          const auto& src = db::get<
              domain::Tags::Jacobian<Dim, Frame::Grid, Frame::Inertial>>(box);
          jac = Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>{};
          for (size_t i = 0; i < src.size(); ++i) {
            make_const_view(make_not_null(&std::as_const((*jac)[i])), src[i], 0,
                            num_pts);
          }
        } else {
          ERROR(
              "SpectralFilter: Missing "
              "domain::Tags::Jacobian<Dim, Grid, Inertial> in the "
              "DataBox. This should be present for both time-dependent and "
              "time-independent evolutions.");
        }
      }
    }

    db::mutate<::Tags::Variables<TagList>>(
        [&filter, &mesh, &inv_jac,
         &jac](const gsl::not_null<Variables<TagList>*> vars) {
          filter.apply_in_volume(vars, mesh, inv_jac, jac);
        },
        make_not_null(&box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace dg::Actions
