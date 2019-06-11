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
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Domain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/GeneralizedHarmonicInterface.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/ConstGlobalCache.hpp"
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

namespace GeneralizedHarmonic {
namespace Actions {
namespace detail {
template <class T, class = cpp17::void_t<>>
struct has_analytic_solution_alias : std::false_type {};
template <class T>
struct has_analytic_solution_alias<
    T, cpp17::void_t<typename T::analytic_solution_tag>> : std::true_type {};
}  // namespace detail

// Note:  I've left the Dim and System template parameters until it is clear
// whether or not what remains is specific to this system, and what might be
// applicable to more than one system
template <size_t Dim>
struct Initialize {
  template <typename Metavariables>
  struct ConstraintsAndGaugeTags {
    using Inertial = Frame::Inertial;
    using simple_tags = db::AddSimpleTags<>;
    using compute_tags = db::AddComputeTags<
        GeneralizedHarmonic::DampedHarmonicHCompute<Dim, Inertial>,
        GeneralizedHarmonic::SpacetimeDerivDampedHarmonicHCompute<Dim,
                                                                  Inertial>,
        GeneralizedHarmonic::Tags::SpatialDerivGaugeHCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::GaugeConstraintCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::FConstraintCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::TwoIndexConstraintCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::FourIndexConstraintCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::ConstraintEnergyCompute<Dim, Inertial>>;
    template <typename TagsList>
    static auto initialize(db::DataBox<TagsList>&& box) noexcept {
      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box));
    }
  };

  template <typename Metavariables>
  struct VariablesTags {
    using Inertial = Frame::Inertial;
    using system = typename Metavariables::system;
    using variables_tag = typename system::variables_tag;

    using simple_tags = db::AddSimpleTags<
        variables_tag, GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Inertial>,
        GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<Dim, Inertial>,
        GeneralizedHarmonic::OptionTags::GaugeHRollOnStartTime,
        GeneralizedHarmonic::OptionTags::GaugeHRollOnTimeWindow,
        GeneralizedHarmonic::OptionTags::GaugeHSpatialWeightDecayWidth<
            Inertial>>;
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
        gr::Tags::SpacetimeChristoffelFirstKindCompute<Dim, Inertial,
                                                       DataVector>,
        gr::Tags::SpacetimeChristoffelSecondKindCompute<Dim, Inertial,
                                                        DataVector>,
        gr::Tags::TraceSpacetimeChristoffelFirstKindCompute<Dim, Inertial,
                                                            DataVector>,
        gr::Tags::SpatialChristoffelFirstKindCompute<Dim, Inertial, DataVector>,
        gr::Tags::SpatialChristoffelSecondKindCompute<Dim, Inertial,
                                                      DataVector>,
        gr::Tags::TraceSpatialChristoffelFirstKindCompute<Dim, Inertial,
                                                          DataVector>,
        GeneralizedHarmonic::Tags::ExtrinsicCurvatureCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::TraceExtrinsicCurvatureCompute<Dim,
                                                                  Inertial>,
        GeneralizedHarmonic::Tags::ConstraintGamma0Compute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::ConstraintGamma1Compute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::ConstraintGamma2Compute<Dim, Inertial>>;

    template <typename TagsList>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
        const double /*initial_time*/) noexcept {
      const size_t num_grid_points =
          db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points();
      const auto& inertial_coords =
          db::get<::Tags::Coordinates<Dim, Inertial>>(box);

      // The evolution variables, gauge source function, and spacetime
      // derivative of the gauge source function will be overwritten by
      // numerical initial data. So here just initialize them to signaling nans.
      const auto& spacetime_metric =
          make_with_value<tnsr::aa<DataVector, Dim, Inertial>>(
              inertial_coords, std::numeric_limits<double>::signaling_NaN());
      const auto& phi = make_with_value<tnsr::iaa<DataVector, Dim, Inertial>>(
          inertial_coords, std::numeric_limits<double>::signaling_NaN());
      const auto& pi = make_with_value<tnsr::aa<DataVector, Dim, Inertial>>(
          inertial_coords, std::numeric_limits<double>::signaling_NaN());

      using Vars = db::item_type<variables_tag>;
      Vars vars{num_grid_points};
      const tuples::TaggedTuple<gr::Tags::SpacetimeMetric<Dim>,
                                GeneralizedHarmonic::Tags::Phi<Dim>,
                                GeneralizedHarmonic::Tags::Pi<Dim>>
          solution_tuple(spacetime_metric, phi, pi);

      vars.assign_subset(solution_tuple);

      const auto& initial_gauge_source =
          make_with_value<tnsr::a<DataVector, Dim, Inertial>>(
              inertial_coords, std::numeric_limits<double>::signaling_NaN());
      const auto& spacetime_deriv_initial_gauge_source =
          make_with_value<tnsr::ab<DataVector, Dim, Inertial>>(
              inertial_coords, std::numeric_limits<double>::signaling_NaN());

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(vars), std::move(initial_gauge_source),
          std::move(spacetime_deriv_initial_gauge_source), 0.0, 1.0e30, 5.0e30);
    }
  };

  template <class Metavariables>
  using return_tag_list = tmpl::append<
      typename Initialization::Domain<Dim>::simple_tags,
      typename VariablesTags<Metavariables>::simple_tags,
      typename Initialization::GeneralizedHarmonicInterface<
          typename Metavariables::system>::simple_tags,
      typename Initialization::Evolution<
          typename Metavariables::system>::simple_tags,
      typename ConstraintsAndGaugeTags<Metavariables>::simple_tags,
      typename Initialization::DiscontinuousGalerkin<
          Metavariables>::simple_tags,
      typename Initialization::MinMod<Dim>::simple_tags,
      typename Initialization::Domain<Dim>::compute_tags,
      typename VariablesTags<Metavariables>::compute_tags,
      typename Initialization::GeneralizedHarmonicInterface<
          typename Metavariables::system>::compute_tags,
      typename Initialization::Evolution<
          typename Metavariables::system>::compute_tags,
      typename ConstraintsAndGaugeTags<Metavariables>::compute_tags,
      typename Initialization::DiscontinuousGalerkin<
          Metavariables>::compute_tags,
      typename Initialization::MinMod<Dim>::compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    std::vector<std::array<size_t, Dim>> initial_extents,
                    Domain<Dim, Frame::Inertial> domain,
                    const double initial_time, const double initial_dt,
                    const double initial_slab_size) noexcept {
    using system = typename Metavariables::system;
    auto domain_box = Initialization::Domain<Dim>::initialize(
        db::DataBox<tmpl::list<>>{}, array_index, initial_extents, domain);
    auto variables_box = VariablesTags<Metavariables>::initialize(
        std::move(domain_box), cache, initial_time);
    auto domain_interface_box =
        Initialization::GeneralizedHarmonicInterface<system>::initialize(
            std::move(variables_box));
    auto evolution_box = Initialization::Evolution<system>::initialize(
        std::move(domain_interface_box), cache, initial_time, initial_dt,
        initial_slab_size);
    auto constraints_and_gauge_box =
        ConstraintsAndGaugeTags<Metavariables>::initialize(
            std::move(evolution_box));
    auto dg_box =
        Initialization::DiscontinuousGalerkin<Metavariables>::initialize(
            std::move(constraints_and_gauge_box), initial_extents);
    auto limiter_box =
        Initialization::MinMod<Dim>::initialize(std::move(dg_box));
    return std::make_tuple(std::move(limiter_box));
  }
};
}  // namespace Actions
}  // namespace GeneralizedHarmonic
