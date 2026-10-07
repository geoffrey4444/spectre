// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <map>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/RadialProjectionSourceIncrement.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/RadialSourceProjectionOptions.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace gh::Actions {
/// Optional research action immediately after the original DG volume RHS.
/// Adds only a local-source increment before boundary application/history
/// recording. The disabled setting returns before accessing element data.
struct ProjectRadialLocalSource {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& settings =
        Parallel::get<gh::Tags::RadialSourceProjection>(cache);
    if (not settings.has_value()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const auto& element = db::get<domain::Tags::Element<3>>(box);
    if (not std::binary_search(settings->block_ids.begin(),
                               settings->block_ids.end(),
                               element.id().block_id())) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const auto& gamma1 = get(db::get<gh::Tags::ConstraintGamma1>(box));
    if (not std::all_of(gamma1.begin(), gamma1.end(),
                        [](const double x) { return x == -1.0; })) {
      ERROR(
          "Experimental radial local-source projection is qualified only for "
          "gamma1=-1; element "
          << element.id());
    }
    const auto& mesh = db::get<domain::Tags::Mesh<3>>(box);
    // Only immutable polynomial operators are cached. Moving maps/functions
    // of time and all GH sources are evaluated afresh at the current time.
    static thread_local std::map<std::pair<size_t, size_t>,
                                 gh::RadialSourceProjection>
        operators{};
    const auto key = std::pair(mesh.extents(0), settings->evaluation_points);
    const auto& projection =
        operators.try_emplace(key, key.first, key.second).first->second;
    const bool preserve_inner =
        settings->preserve_inner_source and
        element.external_boundaries().contains(Direction<3>::lower_xi());
    using VariablesTag = typename gh::System<3>::variables_tag;
    const auto increment = gh::radial_projection_source_increment(
        projection, db::get<VariablesTag>(box), mesh,
        db::get<::Tags::Time>(box),
        db::get<domain::Tags::Coordinates<3, Frame::ElementLogical>>(box),
        db::get<domain::Tags::ElementMap<3, Frame::Grid>>(box),
        db::get<domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                            Frame::Inertial>>(
            box),
        db::get<domain::Tags::FunctionsOfTime>(box),
        db::get<domain::Tags::Coordinates<3, Frame::Inertial>>(box),
        db::get<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                              Frame::Inertial>>(box),
        db::get<domain::Tags::MeshVelocity<3, Frame::Inertial>>(box),
        Parallel::get<gh::gauges::Tags::GaugeCondition>(cache), preserve_inner);
    using DtTag = db::add_tag_prefix<::Tags::dt, VariablesTag>;
    db::mutate<DtTag>(
        [&increment](const gsl::not_null<typename DtTag::type*> dt) {
          for (size_t i = 0; i < increment.size(); ++i) {
            dt->data()[i] += increment[i];
          }
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace gh::Actions
