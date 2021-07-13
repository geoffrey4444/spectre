// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Protocols/Metavariables.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "NumericalAlgorithms/LinearOperators/FilterAction.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "Time/StepControllers/Factory.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

// First template parameter specifies the source of the initial data, which
// could be an analytic solution, analytic data, or imported numerical data.
// Second template parameter specifies the analytic solution used when imposing
// dirichlet boundary conditions or against which to compute error norms.
template <typename InitialData, typename BoundaryConditions>
struct EvolutionMetavars
    : public virtual GeneralizedHarmonicDefaults,
      public GeneralizedHarmonicTemplateBase<
          EvolutionMetavars<InitialData, BoundaryConditions>> {
  using const_global_cache_tags =
      typename GeneralizedHarmonicTemplateBase<EvolutionMetavars<
          InitialData, BoundaryConditions>>::const_global_cache_tags;
  using observed_reduction_data_tags =
      typename GeneralizedHarmonicTemplateBase<EvolutionMetavars<
          InitialData, BoundaryConditions>>::observed_reduction_data_tags;
  using component_list = typename GeneralizedHarmonicTemplateBase<
      EvolutionMetavars<InitialData, BoundaryConditions>>::component_list;
  template <typename ParallelComponent>
  using registration_list = typename GeneralizedHarmonicTemplateBase<
      EvolutionMetavars<InitialData, BoundaryConditions>>::
      template registration_list<ParallelComponent>;

  static constexpr bool use_damped_harmonic_rollon = false;
  static constexpr bool override_functions_of_time = true;
  struct domain : tt::ConformsTo<::domain::protocols::Metavariables> {
    static constexpr bool enable_time_dependent_maps = true;
  };

  using initialization_actions = tmpl::list<
      Actions::SetupDataBox,
      Initialization::Actions::TimeAndTimeStep<
          EvolutionMetavars<InitialData, BoundaryConditions>>,
      evolution::dg::Initialization::Domain<volume_dim,
                                            override_functions_of_time>,
      Initialization::Actions::NonconservativeSystem<system>,
      std::conditional_t<
          evolution::is_numeric_initial_data_v<InitialData>, tmpl::list<>,
          evolution::Initialization::Actions::SetVariables<
              ::domain::Tags::Coordinates<volume_dim, Frame::Logical>>>,
      Initialization::Actions::TimeStepperHistory<
          EvolutionMetavars<InitialData, BoundaryConditions>>,
      GeneralizedHarmonic::Actions::InitializeGhAnd3Plus1Variables<volume_dim>,
      Initialization::Actions::AddComputeTags<tmpl::push_back<
          StepChoosers::step_chooser_compute_tags<
              EvolutionMetavars<InitialData, BoundaryConditions>>,
          evolution::Tags::AnalyticCompute<
              volume_dim, Tags::AnalyticSolution<BoundaryConditions>,
              system::variables_tag::tags_list>>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Parallel::Actions::TerminatePhase>;

  using step_actions = tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<
          EvolutionMetavars<InitialData, BoundaryConditions>>,
      evolution::dg::Actions::ApplyBoundaryCorrections<
          EvolutionMetavars<InitialData, BoundaryConditions>>,
      tmpl::conditional_t<
          local_time_stepping, tmpl::list<>,
          tmpl::list<
              Actions::RecordTimeStepperData<>,
              evolution::Actions::RunEventsAndDenseTriggers<>,
              Actions::UpdateU<>,
              dg::Actions::Filter<
                  Filters::Exponential<0>,
                  tmpl::list<gr::Tags::SpacetimeMetric<
                                 volume_dim, Frame::Inertial, DataVector>,
                             GeneralizedHarmonic::Tags::Pi<volume_dim,
                                                           Frame::Inertial>,
                             GeneralizedHarmonic::Tags::Phi<
                                 volume_dim, Frame::Inertial>>>>>>;

  static constexpr Options::String help{
      "Evolve a binary black hole using the Generalized Harmonic "
      "formulation\n"};
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &GeneralizedHarmonic::BoundaryConditions::register_derived_with_charm,
    &GeneralizedHarmonic::BoundaryCorrections::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &GeneralizedHarmonic::ConstraintDamping::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        PhaseChange<metavariables::phase_changes>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
