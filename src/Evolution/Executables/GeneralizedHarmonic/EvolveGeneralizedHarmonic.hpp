// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "AlgorithmSingleton.hpp"
#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsCharacteresticSpeeds.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/NumericalInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/SemiAnalyticBoundaryConditions.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Importers/ElementActions.hpp"
#include "IO/Importers/ReadSpecThirdOrderPiecewisePolynomial.hpp"
#include "IO/Importers/VolumeDataReader.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/RegisterObservers.hpp"
#include "IO/Observer/Tags.hpp"
#include "Informer/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderSchemeLts.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolate.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/Events/ObserveTensorNorms.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepChoosers/StepToTimes.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

#include "Evolution/Actions/AddMeshVelocityNonconservative.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/UpwindPenaltyCorrection.hpp"

/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_ConstGlobalCache;
}  // namespace Parallel
/// \endcond

struct EvolutionMetavars {
  static constexpr int volume_dim = 3;
  using frame = Frame::Inertial;
  using system = GeneralizedHarmonic::System<volume_dim>;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;

  using analytic_solution =
      GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>;
  using analytic_solution_tag = Tags::AnalyticSolution<analytic_solution>;
  using initial_data_tag = Tags::AnalyticSolution<analytic_solution>;
  using boundary_condition_tag = initial_data_tag;
  using normal_dot_numerical_flux = Tags::NumericalFlux<
      GeneralizedHarmonic::UpwindPenaltyCorrection<volume_dim>>;

  // The type of initial data for the evolution. Set to `analytic_solution` for
  // starting from an analytic solution, or `NumericalInitialData` to read
  // data from the disk.
  using initial_data = NumericalInitialData<system>;

  using step_choosers_common =
      tmpl::list<StepChoosers::Registrars::Cfl<volume_dim, Frame::Inertial>,
                 StepChoosers::Registrars::Constant,
                 StepChoosers::Registrars::Increase>;
  using step_choosers_for_step_only =
      tmpl::list<StepChoosers::Registrars::PreventRapidIncrease>;
  using step_choosers_for_slab_only =
      tmpl::list<StepChoosers::Registrars::StepToTimes>;
  using step_choosers = tmpl::conditional_t<
      local_time_stepping,
      tmpl::append<step_choosers_common, step_choosers_for_step_only>,
      tmpl::list<>>;
  using slab_choosers = tmpl::conditional_t<
      local_time_stepping,
      tmpl::append<step_choosers_common, step_choosers_for_slab_only>,
      tmpl::append<step_choosers_common, step_choosers_for_step_only,
                   step_choosers_for_slab_only>>;

  using time_stepper_tag = Tags::TimeStepper<
      tmpl::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>>;
  using boundary_scheme = tmpl::conditional_t<
      local_time_stepping,
      dg::FirstOrderScheme::FirstOrderSchemeLts<
          volume_dim, typename system::variables_tag, normal_dot_numerical_flux,
          Tags::TimeStepId, time_stepper_tag>,
      dg::FirstOrderScheme::FirstOrderScheme<
          volume_dim, typename system::variables_tag, normal_dot_numerical_flux,
          Tags::TimeStepId>>;

  using analytic_solution_fields =
      db::get_variables_tags_list<typename system::variables_tag>;
  using observe_fields = tmpl::append<
      tmpl::list<
          gr::Tags::Lapse<DataVector>,
          ::Tags::PointwiseL2Norm<
              GeneralizedHarmonic::Tags::GaugeConstraint<volume_dim, frame>>,
          ::Tags::PointwiseL2Norm<GeneralizedHarmonic::Tags::
                                      ThreeIndexConstraint<volume_dim, frame>>,
          ::Tags::PointwiseL2Norm<GeneralizedHarmonic::Tags::
                                      FourIndexConstraint<volume_dim, frame>>>>;

  // HACK until we merge in a compute tag StrahlkorperGr::AreaCompute.
  // For now, simply do a surface integral of unity on the horizon to get the
  // horizon area.
  struct Unity : db::ComputeTag {
    static std::string name() noexcept { return "Unity"; }
    static Scalar<DataVector> function(
        const Scalar<DataVector>& used_for_size) noexcept {
      return make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
    }
    using argument_tags =
        tmpl::list<StrahlkorperGr::Tags::AreaElement<Frame::Inertial>>;
  };

  struct AhA {
    using tags_to_observe =
        tmpl::list<StrahlkorperGr::Tags::SurfaceIntegral<Unity, frame>>;
    using compute_items_on_source = tmpl::list<
        gr::Tags::SpatialMetricCompute<volume_dim, frame, DataVector>,
        ah::Tags::InverseSpatialMetricCompute<volume_dim, frame>,
        ah::Tags::ExtrinsicCurvatureCompute<volume_dim, frame>,
        ah::Tags::SpatialChristoffelSecondKindCompute<volume_dim, frame>>;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
                   gr::Tags::InverseSpatialMetric<volume_dim, frame>,
                   gr::Tags::ExtrinsicCurvature<volume_dim, frame>,
                   gr::Tags::SpatialChristoffelSecondKind<volume_dim, frame>>;
    using compute_items_on_target = tmpl::append<
        tmpl::list<StrahlkorperGr::Tags::AreaElement<frame>, Unity>,
        tags_to_observe>;
    using compute_target_points =
        intrp::TargetPoints::ApparentHorizon<AhA, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<AhA>;
    using post_horizon_find_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, AhA,
                                                     AhA>;
  };
  using interpolation_target_tags = tmpl::list<AhA>;
  using interpolator_source_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Pi<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Phi<volume_dim, frame>>;

  using observation_events = tmpl::list<
      dg::Events::Registrars::ObserveTensorNorms<Tags::Time, observe_fields>,
      dg::Events::Registrars::ObserveFields<volume_dim, Tags::Time,
                                            observe_fields>,
      Events::Registrars::ChangeSlabSize<slab_choosers>>;
  using triggers = Triggers::time_triggers;

  // Events include the observation events and finding the horizon
  using events = tmpl::push_back<observation_events,
                                 intrp::Events::Registrars::Interpolate<
                                     3, AhA, interpolator_source_vars>>;

  // A tmpl::list of tags to be added to the ConstGlobalCache by the
  // metavariables
  using const_global_cache_tags = tmpl::list<
      initial_data_tag, normal_dot_numerical_flux, time_stepper_tag,
      GeneralizedHarmonic::Tags::GaugeHRollOnStartTime,
      GeneralizedHarmonic::Tags::GaugeHRollOnTimeWindow,
      GeneralizedHarmonic::Tags::GaugeHSpatialWeightDecayWidth<frame>,
      Tags::EventsAndTriggers<events, triggers>>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::push_back<Event<observation_events>::creatable_classes,
                      typename AhA::post_horizon_find_callback>>;

  using step_actions = tmpl::flatten<tmpl::list<
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          domain::Tags::InternalDirections<volume_dim>>,
      dg::Actions::CollectDataForFluxes<
          boundary_scheme, domain::Tags::InternalDirections<volume_dim>>,
      dg::Actions::SendDataForFluxes<boundary_scheme>,
      Actions::ComputeTimeDerivative<
          GeneralizedHarmonic::ComputeDuDt<volume_dim>>,
      evolution::Actions::AddMeshVelocityNonconservative,
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
      GeneralizedHarmonic::Actions::
          ImposeDirichletBoundaryConditionsUnlessOnlyOutgoingCharSpeeds<
              EvolutionMetavars>,
      dg::Actions::CollectDataForFluxes<
          boundary_scheme,
          domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
      dg::Actions::ReceiveDataForFluxes<boundary_scheme>,
      tmpl::conditional_t<local_time_stepping,
                          tmpl::list<Actions::RecordTimeStepperData<>,
                                     Actions::MutateApply<boundary_scheme>>,
                          tmpl::list<Actions::MutateApply<boundary_scheme>,
                                     Actions::RecordTimeStepperData<>>>,
      Actions::UpdateU<>>>;

  enum class Phase {
    Initialization,
    InitializeTimeStepperHistory,
    Register,
    ImportData,
    Evolve,
    Exit
  };

  using initialization_actions = tmpl::list<
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<volume_dim>,
      Initialization::Actions::NonconservativeSystem,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,
      GeneralizedHarmonic::Actions::InitializeGhAnd3Plus1Variables<volume_dim>,
      dg::Actions::InitializeInterfaces<
          system,
          dg::Initialization::slice_tags_to_face<
              typename system::variables_tag,
              domain::Tags::Coordinates<volume_dim, Frame::Grid>,
              gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
              gr::Tags::DetAndInverseSpatialMetricCompute<volume_dim, frame,
                                                          DataVector>,
              gr::Tags::Shift<volume_dim, frame, DataVector>,
              gr::Tags::Lapse<DataVector>>,
          dg::Initialization::slice_tags_to_exterior<
              domain::Tags::Coordinates<volume_dim, Frame::Grid>,
              gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
              gr::Tags::DetAndInverseSpatialMetricCompute<volume_dim, frame,
                                                          DataVector>,
              gr::Tags::Shift<volume_dim, frame, DataVector>,
              gr::Tags::Lapse<DataVector>>,
          dg::Initialization::face_compute_tags<
              domain::Tags::BoundaryCoordinates<volume_dim, true>,
              GeneralizedHarmonic::Tags::ConstraintGamma0BBHCompute<
                  volume_dim, Frame::Grid>,
              GeneralizedHarmonic::Tags::ConstraintGamma1BBHCompute<
                  volume_dim, Frame::Grid>,
              GeneralizedHarmonic::Tags::ConstraintGamma2BBHCompute<
                  volume_dim, Frame::Grid>,
              GeneralizedHarmonic::CharacteristicFieldsCompute<volume_dim,
                                                               frame>>,
          dg::Initialization::exterior_compute_tags<
              GeneralizedHarmonic::Tags::ConstraintGamma0BBHCompute<
                  volume_dim, Frame::Grid>,
              GeneralizedHarmonic::Tags::ConstraintGamma1BBHCompute<
                  volume_dim, Frame::Grid>,
              GeneralizedHarmonic::Tags::ConstraintGamma2BBHCompute<
                  volume_dim, Frame::Grid>,
              GeneralizedHarmonic::CharacteristicFieldsCompute<volume_dim,
                                                               frame>>,
          true, true>,
      Initialization::Actions::AddComputeTags<
          tmpl::list<evolution::Tags::AnalyticCompute<
              volume_dim, initial_data_tag, analytic_solution_fields>>>,
      GeneralizedHarmonic::Actions::InitializeGauge<volume_dim>,
      GeneralizedHarmonic::Actions::InitializeConstraints<volume_dim>,
      dg::Actions::InitializeMortars<boundary_scheme, true>,
      Initialization::Actions::DiscontinuousGalerkin<EvolutionMetavars>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using component_list = tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      intrp::Interpolator<EvolutionMetavars>,
      intrp::InterpolationTarget<EvolutionMetavars, AhA>,
      tmpl::conditional_t<is_numerical_initial_data_v<initial_data>,
                          importers::VolumeDataReader<EvolutionMetavars>,
                          tmpl::list<>>,
      DgElementArray<
          EvolutionMetavars,
          tmpl::list<
              Parallel::PhaseActions<Phase, Phase::Initialization,
                                     initialization_actions>,

              Parallel::PhaseActions<
                  Phase, Phase::InitializeTimeStepperHistory,
                  SelfStart::self_start_procedure<step_actions>>,

              Parallel::PhaseActions<
                  Phase, Phase::Register,
                  tmpl::list<
                      importers::Actions::ReadSpecThirdOrderPiecewisePolynomial,
                      intrp::Actions::RegisterElementWithInterpolator,
                      observers::Actions::RegisterWithObservers<
                          observers::RegisterObservers<
                              Tags::Time, element_observation_type>>,
                      tmpl::conditional_t<
                          is_numerical_initial_data_v<initial_data>,
                          importers::Actions::RegisterWithVolumeDataReader,
                          tmpl::list<>>,
                      Parallel::Actions::TerminatePhase>>,

              Parallel::PhaseActions<
                  Phase, Phase::Evolve,
                  tmpl::list<Actions::RunEventsAndTriggers,
                             Actions::ChangeSlabSize, step_actions,
                             Actions::AdvanceTime>>>>>;

  static constexpr OptionString help{
      "Evolve a generalized harmonic analytic solution.\n\n"
      "The analytic solution is: KerrSchild\n"
      "The numerical flux is:    UpwindFlux\n"};

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::InitializeTimeStepperHistory;
      case Phase::InitializeTimeStepperHistory:
        return Phase::Register;
      case Phase::Register:
        return is_numerical_initial_data_v<initial_data> ? Phase::ImportData
                                                         : Phase::Evolve;
      case Phase::ImportData:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::slab_choosers>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
