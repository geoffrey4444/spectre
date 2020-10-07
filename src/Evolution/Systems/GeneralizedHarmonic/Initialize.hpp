// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>  // IWYU pragma: keep
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace GeneralizedHarmonic {
namespace Actions {
template <size_t Dim>
struct InitializeConstraints {
  using frame = Frame::Inertial;

  using compute_tags = tmpl::flatten<db::AddComputeTags<
      GeneralizedHarmonic::Tags::GaugeConstraintCompute<Dim, frame>,
      GeneralizedHarmonic::Tags::FConstraintCompute<Dim, frame>,
      GeneralizedHarmonic::Tags::TwoIndexConstraintCompute<Dim, frame>,
      // following tags added to observe constraints
      ::Tags::PointwiseL2NormCompute<
          GeneralizedHarmonic::Tags::GaugeConstraint<Dim, frame>>,
      ::Tags::PointwiseL2NormCompute<
          GeneralizedHarmonic::Tags::FConstraint<Dim, frame>>,
      ::Tags::PointwiseL2NormCompute<
          GeneralizedHarmonic::Tags::TwoIndexConstraint<Dim, frame>>,
      ::Tags::PointwiseL2NormCompute<
          GeneralizedHarmonic::Tags::ThreeIndexConstraint<Dim, frame>>,
      // The 4-index constraint is only implemented in 3d
      tmpl::conditional_t<
          Dim == 3,
          tmpl::list<
              GeneralizedHarmonic::Tags::FourIndexConstraintCompute<Dim, frame>,
              GeneralizedHarmonic::Tags::ConstraintEnergyCompute<Dim, frame>,
              ::Tags::PointwiseL2NormCompute<
                  GeneralizedHarmonic::Tags::FourIndexConstraint<Dim, frame>>,
              ::Tags::PointwiseL2NormCompute<
                  GeneralizedHarmonic::Tags::ConstraintEnergy<Dim, frame>>>,
          tmpl::list<>>>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(std::move(box));
  }
};

template <size_t Dim>
struct InitializeGhAnd3Plus1Variables {
  using frame = Frame::Inertial;
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
      ConstraintDamping::Tags::ConstraintGamma0Compute<Dim, frame>,
      ConstraintDamping::Tags::ConstraintGamma1Compute<Dim, frame>,
      ConstraintDamping::Tags::ConstraintGamma2Compute<Dim, frame>>;

  using const_global_cache_tags = tmpl::list<
      GeneralizedHarmonic::Tags::ExtrinsicCurvatureCompute<Dim, frame>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma0<
          Dim, frame>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma1<
          Dim, frame>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma2<
          Dim, frame>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace GeneralizedHarmonic
