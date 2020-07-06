// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>  // IWYU pragma: keep
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace GeneralizedHarmonic {
namespace Actions {
template <size_t Dim>
struct InitializeConstraints {
  using frame = Frame::Inertial;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = tmpl::flatten<db::AddComputeTags<
        GeneralizedHarmonic::Tags::GaugeConstraintCompute<Dim, frame>,
        // following tags added to observe constraints
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::GaugeConstraint<Dim, frame>>,
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::ThreeIndexConstraint<Dim, frame>>,
        // The 4-index constraint is only implemented in 3d
        tmpl::conditional_t<
            Dim == 3,
            tmpl::list<GeneralizedHarmonic::Tags::FourIndexConstraintCompute<
                           Dim, frame>,
                       ::Tags::PointwiseL2NormCompute<
                           GeneralizedHarmonic::Tags::FourIndexConstraint<
                               Dim, frame>>>,
            tmpl::list<>>>>;

    return std::make_tuple(
        Initialization::merge_into_databox<InitializeConstraints,
                                           db::AddSimpleTags<>, compute_tags>(
            std::move(box)));
  }
};

template <size_t Dim>
struct InitializeGhAnd3Plus1Variables {
  using frame = Frame::Inertial;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = db::AddComputeTags<
        gr::Tags::SpatialMetricCompute<Dim, frame, DataVector>,
        gr::Tags::DetAndInverseSpatialMetricCompute<Dim, frame, DataVector>,
        gr::Tags::ShiftCompute<Dim, frame, DataVector>,
        gr::Tags::LapseCompute<Dim, frame, DataVector>,
        gr::Tags::SqrtDetSpatialMetricCompute<Dim, frame, DataVector>,
        gr::Tags::SpacetimeNormalOneFormCompute<Dim, frame, DataVector>,
        gr::Tags::SpacetimeNormalVectorCompute<Dim, frame, DataVector>,
        gr::Tags::InverseSpacetimeMetricCompute<Dim, frame, DataVector>,
        GeneralizedHarmonic::Tags::ThreeIndexConstraintCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::ConstraintGamma0BBHCompute<Dim, Frame::Grid>,
        GeneralizedHarmonic::Tags::ConstraintGamma1BBHCompute<Dim, Frame::Grid>,
        GeneralizedHarmonic::Tags::ConstraintGamma2BBHCompute<Dim,
                                                              Frame::Grid>>;

    return std::make_tuple(
        Initialization::merge_into_databox<InitializeGhAnd3Plus1Variables,
                                           db::AddSimpleTags<>, compute_tags>(
            std::move(box)));
  }
};

template <size_t Dim>
struct SetPhiFromDerivSpacetimeMetric {
  using frame = Frame::Inertial;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // Reset phi to the spaital derivative of the spacetime metric, to
    // satisfy the three-index constraint
    const auto& deriv_spacetime_metric =
        db::get<::Tags::deriv<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial>,
                              tmpl::size_t<Dim>, Frame::Inertial>>(box);

    // In the damped harmonic gauge initialization, Pi is reset in terms
    // of lapse, shift, dt_lapse, dt_shift, and Phi. So here correct Pi
    // so it's the same as if we had corrected Phi during damped harmonic
    // gauge initialization.
    const Scalar<DataVector>& lapse = db::get<gr::Tags::Lapse<DataVector>>(box);
    const tnsr::I<DataVector, Dim, frame>& shift =
        db::get<gr::Tags::Shift<Dim, frame, DataVector>>(box);
    const tnsr::iaa<DataVector, Dim, frame>& old_phi =
        db::get<GeneralizedHarmonic::Tags::Phi<Dim, frame>>(box);
    const tnsr::iaa<DataVector, Dim, frame>& three_index_constraint =
        GeneralizedHarmonic::three_index_constraint(deriv_spacetime_metric,
                                                    old_phi);
    auto pi_correction =
        make_with_value<tnsr::aa<DataVector, Dim, frame>>(get(lapse), 0.0);
    for (size_t a = 0; a < Dim + 1; ++a) {
      for (size_t b = a; b < Dim + 1; ++b) {
        for (size_t i = 0; i < Dim; ++i) {
          pi_correction.get(a, b) +=
              shift.get(i) * three_index_constraint.get(i, a, b) / get(lapse);
        }
      }
    }

    db::mutate<GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>(
        make_not_null(&box),
        [&deriv_spacetime_metric](const auto phi_ptr) noexcept {
          *phi_ptr = deriv_spacetime_metric;
        });

    db::mutate<GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>>(
        make_not_null(&box), [&pi_correction](const auto pi_ptr) noexcept {
          for (size_t a = 0; a < Dim + 1; ++a) {
            for (size_t b = a; b < Dim + 1; ++b) {
              (*pi_ptr).get(a, b) += pi_correction.get(a, b);
            }
          }
        });

    return std::make_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace GeneralizedHarmonic
