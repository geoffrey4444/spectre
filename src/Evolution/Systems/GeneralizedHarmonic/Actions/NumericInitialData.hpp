// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpatialMetric.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace GeneralizedHarmonic {

/// Sets of initial data variables from which the generalized harmonic system
/// variables can be computed.
enum class NumericInitialDataVariables {
  /*!
   * Generalized harmonic variables:
   *
   * - gr::Tags::SpacetimeMetric
   * - GeneralizedHarmonic::Tags::Pi
   * - GeneralizedHarmonic::Tags::Phi
   */
  GeneralizedHarmonic,
  /*!
   * ADM variables:
   *
   * - gr::Tags::SpatialMetric
   * - gr::Tags::Lapse
   * - gr::Tags::Shift
   * - gr::Tags::ExtrinsicCurvature
   */
  Adm
};

/// List of tags corresponding to the selected
/// GeneralizedHarmonic::NumericInitialDataVariables
template <NumericInitialDataVariables SelectedVars>
using numeric_vars = tmpl::conditional_t<
    SelectedVars == NumericInitialDataVariables::GeneralizedHarmonic,
    // Generalized harmonic system variables
    tmpl::list<gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>,
               Tags::Pi<3, Frame::Inertial>, Tags::Phi<3, Frame::Inertial>>,
    // ADM variables
    tmpl::list<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
               gr::Tags::Lapse<DataVector>,
               gr::Tags::Shift<3, Frame::Inertial, DataVector>,
               gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>>;

/// This is the set of all fields that we support loading from volume data
/// files. The GeneralizedHarmonic::NumericInitialDataVariables enum defines
/// subsets of these fields that are loaded and distributed to elements.
using all_numeric_vars = tmpl::remove_duplicates<
    tmpl::append<numeric_vars<NumericInitialDataVariables::GeneralizedHarmonic>,
                 numeric_vars<NumericInitialDataVariables::Adm>>>;

namespace OptionTags {
template <typename ImporterOptionsGroup>
struct NumericInitialDataVariables {
  static std::string name() { return "Variables"; }
  using type = GeneralizedHarmonic::NumericInitialDataVariables;
  static constexpr Options::String help =
      "Set of initial data variables from which the generalized harmonic \n"
      "system variables are computed. Possible values are:\n"
      "  - GeneralizedHarmonic: Read the GH variables 'SpacetimeMetric', \n"
      "    'Pi' and 'Phi' from the initial data directly.\n"
      "  - Adm: Read the ADM variables 'SpatialMetric', 'Lapse', 'Shift' \n"
      "    and 'ExtrinsicCurvature' from the initial data and compute the \n"
      "    GH variables from them.";
  using group = ImporterOptionsGroup;
};
}  // namespace OptionTags

namespace Tags {
/// Selection of GeneralizedHarmonic::NumericInitialDataVariables to load from
/// from the initial data.
template <typename ImporterOptionsGroup>
struct NumericInitialDataVariables : db::SimpleTag {
  using type = GeneralizedHarmonic::NumericInitialDataVariables;
  using option_tags =
      tmpl::list<OptionTags::NumericInitialDataVariables<ImporterOptionsGroup>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) { return value; }
};
}  // namespace Tags

std::ostream& operator<<(std::ostream& os,
                         const NumericInitialDataVariables& value);

}  // namespace GeneralizedHarmonic

/// \cond
template <>
struct Options::create_from_yaml<
    GeneralizedHarmonic::NumericInitialDataVariables> {
  template <typename Metavariables>
  static GeneralizedHarmonic::NumericInitialDataVariables create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
GeneralizedHarmonic::NumericInitialDataVariables
Options::create_from_yaml<GeneralizedHarmonic::NumericInitialDataVariables>::
    create<void>(const Options::Option& options);
/// \endcond

namespace GeneralizedHarmonic::Actions {

/*!
 * \brief Dispatch loading numeric initial data from files.
 *
 * Place this action before GeneralizedHarmonic::Actions::SetNumericInitialData
 * in the action list. See importers::Actions::ReadAllVolumeDataAndDistribute
 * for details, which is invoked by this action.
 *
 * \tparam ImporterOptionsGroup Option group in which options are placed.
 */
template <typename ImporterOptionsGroup>
struct ReadNumericInitialData {
  using const_global_cache_tags =
      tmpl::list<importers::Tags::FileName<ImporterOptionsGroup>,
                 importers::Tags::Subgroup<ImporterOptionsGroup>,
                 importers::Tags::ObservationValue<ImporterOptionsGroup>,
                 Tags::NumericInitialDataVariables<ImporterOptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Select the subset of the available variables that we want to read from
    // the volume data file
    tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<importers::Tags::Selected, all_numeric_vars>>
        selected_fields{};
    const auto select_field = [&selected_fields](auto field_tag_v) {
      using field_tag = tmpl::type_from<decltype(field_tag_v)>;
      get<importers::Tags::Selected<field_tag>>(selected_fields) = true;
    };
    const auto selected_initial_data_vars =
        get<Tags::NumericInitialDataVariables<ImporterOptionsGroup>>(box);
    if (selected_initial_data_vars ==
        NumericInitialDataVariables::GeneralizedHarmonic) {
      tmpl::for_each<
          numeric_vars<NumericInitialDataVariables::GeneralizedHarmonic>>(
          select_field);
    } else if (selected_initial_data_vars == NumericInitialDataVariables::Adm) {
      tmpl::for_each<numeric_vars<NumericInitialDataVariables::Adm>>(
          select_field);
    } else {
      ERROR("Invalid initial data variables: " << selected_initial_data_vars);
    }
    // Dispatch loading the variables from the volume data file
    // - Not using `ckLocalBranch` here to make sure the simple action
    //   invocation is asynchronous.
    auto& reader_component = Parallel::get_parallel_component<
        importers::ElementDataReader<Metavariables>>(cache);
    Parallel::simple_action<importers::Actions::ReadAllVolumeDataAndDistribute<
        ImporterOptionsGroup, all_numeric_vars, ParallelComponent>>(
        reader_component, std::move(selected_fields));
    return {std::move(box)};
  }
};

/*!
 * \brief Receive numeric initial data loaded by
 * GeneralizedHarmonic::Actions::ReadNumericInitialData.
 *
 * Place this action in the action list after
 * GeneralizedHarmonic::Actions::ReadNumericInitialData to wait until the data
 * for this element has arrived, and then transform the data to GH variables and
 * store it in the DataBox to be used as initial data.
 *
 * This action modifies the following tags in the DataBox:
 * - gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>
 * - GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>
 * - GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>
 *
 * \tparam ImporterOptionsGroup Option group in which options are placed.
 */
template <typename ImporterOptionsGroup>
struct SetNumericInitialData {
  static constexpr size_t Dim = 3;
  using inbox_tags = tmpl::list<
      importers::Tags::VolumeData<ImporterOptionsGroup, all_numeric_vars>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    auto& inbox = tuples::get<
        importers::Tags::VolumeData<ImporterOptionsGroup, all_numeric_vars>>(
        inboxes);
    // Using 0 for the temporal ID since we only read the volume data once, so
    // there's no need to keep track of the temporal ID.
    if (inbox.find(0_st) == inbox.end()) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }
    auto numeric_initial_data = std::move(inbox.extract(0_st).mapped());

    const auto selected_initial_data_vars =
        get<Tags::NumericInitialDataVariables<ImporterOptionsGroup>>(box);
    if (selected_initial_data_vars ==
        NumericInitialDataVariables::GeneralizedHarmonic) {
      // We have loaded the GH system variables from the file, so just move the
      // data into the DataBox directly. No conversion needed.
      db::mutate<gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>,
                 Tags::Pi<3, Frame::Inertial>, Tags::Phi<3, Frame::Inertial>>(
          make_not_null(&box),
          [&numeric_initial_data](
              const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
              const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
              const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi) {
            *spacetime_metric = std::move(
                get<gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>(
                    numeric_initial_data));
            *pi = std::move(
                get<Tags::Pi<3, Frame::Inertial>>(numeric_initial_data));
            *phi = std::move(
                get<Tags::Phi<3, Frame::Inertial>>(numeric_initial_data));
          });
    } else if (selected_initial_data_vars == NumericInitialDataVariables::Adm) {
      // We have loaded ADM variables from the file. Convert to GH variables.
      const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
      const auto& inv_jacobian =
          db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                Frame::Inertial>>(box);
      db::mutate<gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>,
                 Tags::Pi<3, Frame::Inertial>, Tags::Phi<3, Frame::Inertial>>(
          make_not_null(&box),
          [](const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
             const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
             const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi,
             const tnsr::ii<DataVector, 3>& spatial_metric,
             const tnsr::ijj<DataVector, 3>& deriv_spatial_metric,
             const Scalar<DataVector>& lapse,
             const tnsr::i<DataVector, 3>& deriv_lapse,
             const tnsr::I<DataVector, 3>& shift,
             const tnsr::iJ<DataVector, 3>& deriv_shift,
             const tnsr::ii<DataVector, 3>& extrinsic_curvature) {
            // Choose dt_lapse = 0 and dt_shift = 0 (for now)
            const auto dt_lapse =
                make_with_value<Scalar<DataVector>>(lapse, 0.);
            const auto dt_shift =
                make_with_value<tnsr::I<DataVector, 3>>(shift, 0.);
            const auto dt_spatial_metric =
                gr::time_derivative_of_spatial_metric(
                    lapse, shift, deriv_shift, spatial_metric,
                    deriv_spatial_metric, extrinsic_curvature);
            gr::spacetime_metric(spacetime_metric, lapse, shift,
                                 spatial_metric);
            GeneralizedHarmonic::phi(phi, lapse, deriv_lapse, shift,
                                     deriv_shift, spatial_metric,
                                     deriv_spatial_metric);
            GeneralizedHarmonic::pi(pi, lapse, dt_lapse, shift, dt_shift,
                                    spatial_metric, dt_spatial_metric, *phi);
          },
          get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
              numeric_initial_data),
          partial_derivative(
              get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
                  numeric_initial_data),
              mesh, inv_jacobian),
          get<gr::Tags::Lapse<DataVector>>(numeric_initial_data),
          partial_derivative(
              get<gr::Tags::Lapse<DataVector>>(numeric_initial_data), mesh,
              inv_jacobian),
          get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(
              numeric_initial_data),
          partial_derivative(
              get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(
                  numeric_initial_data),
              mesh, inv_jacobian),
          get<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>(
              numeric_initial_data));
    } else {
      ERROR("Invalid initial data variables: " << selected_initial_data_vars);
    }
    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};

}  // namespace GeneralizedHarmonic::Actions
