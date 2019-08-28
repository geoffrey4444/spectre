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
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace GeneralizedHarmonic {
namespace Actions {
template <size_t Dim>
struct InitializeConstraintsTags {
  using initialization_option_tags = tmpl::list<>;

  using Inertial = Frame::Inertial;
  using simple_tags = db::AddSimpleTags<>;
  using compute_tags = db::AddComputeTags<
      GeneralizedHarmonic::Tags::GaugeConstraintCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::FConstraintCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::TwoIndexConstraintCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::ThreeIndexConstraintCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::FourIndexConstraintCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::ConstraintEnergyCompute<Dim, Inertial>,
      // following tags added to observe constraints
      ::Tags::PointwiseL2NormCompute<
          GeneralizedHarmonic::Tags::GaugeConstraint<Dim, Inertial>>,
      ::Tags::PointwiseL2NormCompute<
          GeneralizedHarmonic::Tags::FConstraint<Dim, Inertial>>,
      ::Tags::PointwiseL2NormCompute<
          GeneralizedHarmonic::Tags::TwoIndexConstraint<Dim, Inertial>>,
      ::Tags::PointwiseL2NormCompute<
          GeneralizedHarmonic::Tags::ThreeIndexConstraint<Dim, Inertial>>,
      ::Tags::PointwiseL2NormCompute<
          GeneralizedHarmonic::Tags::FourIndexConstraint<Dim, Inertial>>,
      ::Tags::PointwiseL2NormCompute<
          GeneralizedHarmonic::Tags::ConstraintEnergy<Dim, Inertial>>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        Initialization::merge_into_databox<InitializeConstraintsTags,
                                           simple_tags, compute_tags>(
            std::move(box)));
  }
};

template <size_t Dim>
struct InitializeGHAnd3Plus1VariablesTags {
  using initialization_option_tags = tmpl::list<>;

  using Inertial = Frame::Inertial;
  using system = GeneralizedHarmonic::System<Dim>;
  using variables_tag = typename system::variables_tag;

  using simple_tags = db::AddSimpleTags<>;
  using compute_tags = db::AddComputeTags<
      gr::Tags::SpatialMetricCompute<Dim, Inertial, DataVector>,
      gr::Tags::DetAndInverseSpatialMetricCompute<Dim, Inertial, DataVector>,
      gr::Tags::ShiftCompute<Dim, Inertial, DataVector>,
      gr::Tags::LapseCompute<Dim, Inertial, DataVector>,
      gr::Tags::SqrtDetSpatialMetricCompute<Dim, Inertial, DataVector>,
      gr::Tags::SpacetimeNormalOneFormCompute<Dim, Inertial, DataVector>,
      gr::Tags::SpacetimeNormalVectorCompute<Dim, Inertial, DataVector>,
      gr::Tags::InverseSpacetimeMetricCompute<Dim, Inertial, DataVector>,
      GeneralizedHarmonic::Tags::DerivSpatialMetricCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::DerivLapseCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::DerivShiftCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::TimeDerivSpatialMetricCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::TimeDerivLapseCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::TimeDerivShiftCompute<Dim, Inertial>,
      gr::Tags::DerivativesOfSpacetimeMetricCompute<Dim, Inertial>,
      gr::Tags::DerivSpacetimeMetricCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::ThreeIndexConstraintCompute<Dim, Inertial>,
      gr::Tags::SpacetimeChristoffelFirstKindCompute<Dim, Inertial, DataVector>,
      gr::Tags::SpacetimeChristoffelSecondKindCompute<Dim, Inertial,
                                                      DataVector>,
      gr::Tags::TraceSpacetimeChristoffelFirstKindCompute<Dim, Inertial,
                                                          DataVector>,
      gr::Tags::SpatialChristoffelFirstKindCompute<Dim, Inertial, DataVector>,
      gr::Tags::SpatialChristoffelSecondKindCompute<Dim, Inertial, DataVector>,
      gr::Tags::TraceSpatialChristoffelFirstKindCompute<Dim, Inertial,
                                                        DataVector>,
      GeneralizedHarmonic::Tags::ExtrinsicCurvatureCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::TraceExtrinsicCurvatureCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::ConstraintGamma0Compute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::ConstraintGamma1Compute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::ConstraintGamma2Compute<Dim, Inertial>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        Initialization::merge_into_databox<InitializeGHAnd3Plus1VariablesTags,
                                           simple_tags, compute_tags>(
            std::move(box)));
  }
};

template <size_t Dim>
struct InitializeGaugeTags {
  using initialization_option_tags = tmpl::list<>;

  using Inertial = Frame::Inertial;
  using system = GeneralizedHarmonic::System<Dim>;
  using variables_tag = typename system::variables_tag;

  using simple_tags = db::AddSimpleTags<
      GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Inertial>,
      GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<Dim, Inertial>>;
  using compute_tags = db::AddComputeTags<
      GeneralizedHarmonic::DampedHarmonicHCompute<Dim, Inertial>,
      GeneralizedHarmonic::SpacetimeDerivDampedHarmonicHCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::SpatialDerivGaugeHCompute<Dim, Inertial>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // compute initial-gauge related quantities
    const auto mesh = db::get<::Tags::Mesh<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(box);
    const auto& dt_lapse = get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(box);
    const auto& deriv_lapse =
        get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                          Inertial>>(box);
    const auto& shift = get<gr::Tags::Shift<Dim, Inertial, DataVector>>(box);
    const auto& dt_shift =
        get<::Tags::dt<gr::Tags::Shift<Dim, Inertial, DataVector>>>(box);
    const auto& deriv_shift =
        get<::Tags::deriv<gr::Tags::Shift<Dim, Inertial, DataVector>,
                          tmpl::size_t<Dim>, Inertial>>(box);
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<Dim, Inertial, DataVector>>(box);
    const auto& trace_extrinsic_curvature =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(box);
    const auto& trace_christoffel_last_indices = get<
        gr::Tags::TraceSpatialChristoffelFirstKind<Dim, Inertial, DataVector>>(
        box);

    // call compute item for the initial gauge source function
    const auto initial_gauge_h =
        GeneralizedHarmonic::Tags::GaugeHImplicitFrom3p1QuantitiesCompute<
            Dim, Inertial>::function(lapse, dt_lapse, deriv_lapse, shift,
                                     dt_shift, deriv_shift, spatial_metric,
                                     trace_extrinsic_curvature,
                                     trace_christoffel_last_indices);
    // set time derivatives of InitialGaugeH = 0
    const auto dt_initial_gauge_source =
        make_with_value<tnsr::a<DataVector, Dim, Inertial>>(lapse, 0.);

    // compute spatial derivatives of InitialGaugeH
    using extras_tag = ::Tags::Variables<tmpl::list<
        GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Frame::Inertial>>>;
    using ExtraVars = db::item_type<extras_tag>;
    ExtraVars extra_vars{num_grid_points};
    get<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Inertial>>(extra_vars) =
        initial_gauge_h;
    const auto inverse_jacobian = db::get<
        ::Tags::InverseJacobian<::Tags::ElementMap<Dim, Inertial>,
                                ::Tags::Coordinates<Dim, Frame::Logical>>>(box);
    const auto d_initial_gauge_source = get<
        ::Tags::deriv<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Inertial>,
                      tmpl::size_t<Dim>, Inertial>>(
        partial_derivatives<typename db::item_type<extras_tag>::tags_list>(
            extra_vars, mesh, inverse_jacobian));

    // compute spacetime derivatives of InitialGaugeH
    const auto initial_d4_gauge_h =
        GeneralizedHarmonic::Tags::SpacetimeDerivGaugeHCompute<
            Dim, Inertial>::function(dt_initial_gauge_source,
                                     d_initial_gauge_source);

    // Finally, insert gauge related quantities to the box
    return std::make_tuple(
        Initialization::merge_into_databox<InitializeGaugeTags, simple_tags,
                                           compute_tags>(
            std::move(box), std::move(initial_gauge_h),
            std::move(initial_d4_gauge_h)));
  }
};

}  // namespace Actions
}  // namespace GeneralizedHarmonic
