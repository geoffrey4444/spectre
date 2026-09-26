// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ControlSystem/UpdateFunctionOfTime.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeRescaledSurfaceCharSpeedVars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Events/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct MockFindApparentHorizon {
  using frame = ::Frame::Grid;

  struct Results {
    LinkedMessageId<double> time{};
    ElementId<3> element_id{};
    Mesh<3> mesh;
    Variables<ah::vars_to_interpolate_to_target<3, frame>> vars;
    std::optional<std::string> dependency;
  };
  static Results results;  // NOLINT

  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const LinkedMessageId<double>& incoming_time,
      const ElementId<3>& incoming_element_id, const ::Mesh<3>& incoming_mesh,
      Variables<ah::vars_to_interpolate_to_target<3, frame>>&&
          incoming_vars_to_interpolate,
      const std::optional<std::string>& dependency,
      const bool /*source_vars_have_already_been_received*/ = false) {
    results.time = incoming_time;
    results.element_id = incoming_element_id;
    results.mesh = incoming_mesh;
    results.vars = std::move(incoming_vars_to_interpolate);
    results.dependency = dependency;
  }
};

MockFindApparentHorizon::Results MockFindApparentHorizon::results{};  // NOLINT

struct MockHorizonMetavars : tt::ConformsTo<ah::protocols::HorizonMetavars> {
  using time_tag = ::Tags::TimeAndPrevious<0>;

  using frame = ::Frame::Grid;

  // Don't need callbacks
  using horizon_find_callbacks = tmpl::list<>;
  using horizon_find_failure_callbacks = tmpl::list<>;

  using compute_tags_on_element = tmpl::list<>;

  static constexpr ah::Destination destination = ah::Destination::ControlSystem;

  static std::string name() { return "MockHorizonMetavars"; }
};

template <typename Metavariables>
struct MockComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      ah::Component<Metavariables, MockHorizonMetavars>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<3>, ah::Tags::BlocksForHorizonFind>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize,
                 ah::Tags::PreviousSurface<MockHorizonMetavars>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;

  using replace_these_simple_actions =
      tmpl::list<ah::FindApparentHorizon<MockHorizonMetavars>>;
  using with_these_simple_actions = tmpl::list<MockFindApparentHorizon>;
};

template <typename Metavariables>
struct MockElement {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using initial_databox = db::compute_databox_type<db::AddSimpleTags<>>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;
};

struct MockMetavariables {
  using component_list = tmpl::list<MockComponent<MockMetavariables>,
                                    MockElement<MockMetavariables>>;

  using event = ah::Events::FindApparentHorizon<MockHorizonMetavars>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<Event, tmpl::list<event>>>;
  };
};

struct DiagnosticHorizonMetavars : MockHorizonMetavars {
  using frame = Frame::Distorted;
  static constexpr ah::Destination destination = ah::Destination::Observation;
  static std::string name() { return "DiagnosticHorizon"; }
};

struct MockFindDiagnosticHorizon {
  struct Results {
    LinkedMessageId<double> time{};
    ElementId<3> element_id{};
    Mesh<3> mesh;
    std::optional<std::string> dependency;
    bool source_vars_have_already_been_received{false};
    std::optional<Variables<ah::rescaled_surface_char_speed_vars>> diagnostic;
  };
  static Results results;  // NOLINT

  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const LinkedMessageId<double>& incoming_time,
      const ElementId<3>& incoming_element_id, const Mesh<3>& incoming_mesh,
      Variables<ah::vars_to_interpolate_to_target<3, Frame::Distorted>>&&
      /*incoming_vars_to_interpolate*/,
      const std::optional<std::string>& dependency,
      const bool source_vars_have_already_been_received = false,
      std::optional<Variables<ah::rescaled_surface_char_speed_vars>>
          diagnostic = std::nullopt) {
    results = {incoming_time,
               incoming_element_id,
               incoming_mesh,
               dependency,
               source_vars_have_already_been_received,
               std::move(diagnostic)};
  }
};

MockFindDiagnosticHorizon::Results
    MockFindDiagnosticHorizon::results{};  // NOLINT

template <typename Metavariables>
struct MockDiagnosticComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      ah::Component<Metavariables, DiagnosticHorizonMetavars>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<3>, ah::Tags::BlocksForHorizonFind,
                 ah::Tags::ApparentHorizonOptions<DiagnosticHorizonMetavars>>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize,
                 ah::Tags::PreviousSurface<DiagnosticHorizonMetavars>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using replace_these_simple_actions =
      tmpl::list<ah::FindApparentHorizon<DiagnosticHorizonMetavars>>;
  using with_these_simple_actions = tmpl::list<MockFindDiagnosticHorizon>;
};

struct DiagnosticMetavariables {
  using component_list =
      tmpl::list<MockDiagnosticComponent<DiagnosticMetavariables>,
                 MockElement<DiagnosticMetavariables>>;
  using event = ah::Events::FindApparentHorizon<DiagnosticHorizonMetavars>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<Event, tmpl::list<event>>>;
  };
};

void test_rescaled_surface_event(
    const bool diagnostic_enabled, const bool moving,
    const std::optional<double> previous_surface_time,
    const bool intersects_previous_surface, const bool selected_block) {
  (void)DiagnosticHorizonMetavars::destination;
  CAPTURE(diagnostic_enabled, moving, previous_surface_time,
          intersects_previous_surface, selected_block);
  using metavars = DiagnosticMetavariables;
  using component = MockDiagnosticComponent<metavars>;
  using elem_component = MockElement<metavars>;
  const ElementId<3> element_id{0};
  const LinkedMessageId<double> observation_time{2.0, {1.0}};
  const Mesh<3> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  // The inner shell supplies diagnostic surfaces even if the previous horizon
  // intersected only the outer shell, and no neighbor of this element.
  const domain::creators::Sphere domain_creator{
      1.8,          3.0,  domain::creators::Sphere::Excision{},
      0_st,         3_st, false,
      std::nullopt, {2.2}};
  auto domain = domain_creator.create_domain();
  const std::string block_name = domain.blocks()[element_id.block_id()].name();
  domain::FunctionsOfTimeMap functions_of_time{};
  const DataVector distorted_to_inertial_velocity{0.01, 0.02, 0.03};
  if (moving) {
    using Translation = domain::CoordinateMaps::TimeDependent::Translation<3>;
    const Translation grid_to_distorted{"GridTranslation"};
    const Translation distorted_to_inertial{"InertialTranslation"};
    domain.inject_time_dependent_map_for_block(
        0,
        domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            grid_to_distorted, distorted_to_inertial),
        domain::make_coordinate_map_base<Frame::Grid, Frame::Distorted>(
            grid_to_distorted),
        domain::make_coordinate_map_base<Frame::Distorted, Frame::Inertial>(
            distorted_to_inertial));
    using FunctionOfTime = domain::FunctionsOfTime::PiecewisePolynomial<2>;
    functions_of_time["GridTranslation"] = std::make_unique<FunctionOfTime>(
        0.0,
        std::array{DataVector(3, 0.0), DataVector{0.4, -0.5, 0.6},
                   DataVector(3, 0.0)},
        3.0);
    functions_of_time["InertialTranslation"] = std::make_unique<FunctionOfTime>(
        0.0,
        std::array{DataVector(3, 0.0), distorted_to_inertial_velocity,
                   DataVector(3, 0.0)},
        3.0);
  }
  ah::HorizonOptions<Frame::Distorted> options{};
  if (diagnostic_enabled) {
    options.rescaled_surface_char_speeds.emplace("ExcisionSphere", 3, 1.e-7);
  }
  ah::Storage::LockedPreviousSurface<Frame::Distorted> previous_surface{};
  if (previous_surface_time.has_value()) {
    previous_surface.surface.emplace(
        LinkedMessageId<double>{*previous_surface_time, std::nullopt},
        ylm::Strahlkorper<Frame::Distorted>{3, 2.6, {0.0, 0.0, 0.0}},
        std::unordered_set<ElementId<3>>{
            intersects_previous_surface ? element_id : ElementId<3>{6}});
  }
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {std::move(domain),
       std::unordered_map<std::string, std::unordered_set<std::string>>{
           {DiagnosticHorizonMetavars::name(),
            selected_block ? std::unordered_set<std::string>{block_name}
                           : std::unordered_set<std::string>{}}},
       std::move(options)},
      {std::move(functions_of_time), std::move(previous_surface)}};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_array_component<component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0);
  ActionTesting::emplace_array_component<elem_component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
      element_id);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  auto& cache = ActionTesting::cache<elem_component>(runner, element_id);

  // Minkowski data makes the diagnostic lapse one. The distorted shift is
  // precisely the distorted-to-inertial velocity, excluding the grid velocity.
  Variables<ah::source_vars<3>> vars{mesh.number_of_grid_points(), 0.0};
  auto& metric = get<gr::Tags::SpacetimeMetric<DataVector, 3>>(vars);
  get<0, 0>(metric) = -1.0;
  get<1, 1>(metric) = 1.0;
  get<2, 2>(metric) = 1.0;
  get<3, 3>(metric) = 1.0;
  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<metavars>,
      DiagnosticHorizonMetavars::time_tag, ::Events::Tags::ObserverMesh<3>,
      domain::Tags::Element<3>, ::Tags::Variables<ah::source_vars<3>>>>(
      metavars{}, observation_time, mesh, Element<3>{element_id, {}}, vars);
  auto obs_box = make_observation_box<
      typename metavars::event::compute_tags_for_observation_box>(
      make_not_null(&box));
  const std::optional<std::string> dependency{"DiagnosticDependency"};
  const metavars::event event{dependency};
  event.run(make_not_null(&obs_box), cache, element_id,
            std::add_pointer_t<elem_component>{}, {});

  const bool should_send =
      selected_block and
      (diagnostic_enabled or not previous_surface_time.has_value() or
       (*previous_surface_time < observation_time.id and
        intersects_previous_surface));
  CHECK(runner.is_simple_action_queue_empty<component>(0) == not should_send);
  if (not should_send) {
    return;
  }
  REQUIRE_FALSE(runner.is_simple_action_queue_empty<component>(0));
  runner.invoke_queued_simple_action<component>(0);
  CHECK(runner.is_simple_action_queue_empty<component>(0));
  CHECK(runner.is_simple_action_queue_empty<elem_component>(element_id));
  const auto& results = MockFindDiagnosticHorizon::results;
  CHECK(results.time == observation_time);
  CHECK(results.element_id == element_id);
  CHECK(results.mesh == mesh);
  CHECK(results.dependency == dependency);
  CHECK_FALSE(results.source_vars_have_already_been_received);
  CHECK(results.diagnostic.has_value() == diagnostic_enabled);
  if (diagnostic_enabled) {
    REQUIRE(results.diagnostic.has_value());
    CHECK_ITERABLE_APPROX(
        get(get<gr::Tags::Lapse<DataVector>>(results.diagnostic.value())),
        DataVector(mesh.number_of_grid_points(), 1.0));
    const auto& shift = get<gr::Tags::Shift<DataVector, 3, Frame::Distorted>>(
        results.diagnostic.value());
    for (size_t i = 0; i < 3; ++i) {
      CHECK_ITERABLE_APPROX(
          shift.get(i),
          DataVector(mesh.number_of_grid_points(),
                     moving ? distorted_to_inertial_velocity[i] : 0.0));
    }
  }
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.FindApparentHorizonEvent",
                  "[ApparentHorizonFinder][Unit]") {
  (void)MockHorizonMetavars::destination;
  ::domain::FunctionsOfTime::register_derived_with_charm();
  ::domain::creators::register_derived_with_charm();
  ::domain::creators::time_dependence::register_derived_with_charm();
  using metavars = MockMetavariables;
  const ElementId<3> element_id(2);

  using component = MockComponent<metavars>;
  using elem_component = MockElement<metavars>;
  const double initial_time = 0.0;
  const std::array<double, 3> translation_velocity{{0.01, 0.02, 0.03}};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence = std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3>>(
          initial_time, translation_velocity);
  const auto domain_creator = domain::creators::Sphere(
      1.8, 2.2, domain::creators::Sphere::Excision{}, 1_st, 5_st, false,
      std::nullopt, std::vector<double>{},
      domain::CoordinateMaps::Distribution::Linear, ShellWedges::All,
      std::optional{domain::creators::Sphere::TimeDepOptionType{
          std::move(time_dependence)}});
  const auto block_names = domain_creator.block_names();
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain(),
       std::unordered_map<std::string, std::unordered_set<std::string>>{
           {"MockHorizonMetavars", {block_names.begin(), block_names.end()}}}},
      {domain_creator.functions_of_time(),
       ah::Storage::LockedPreviousSurface<Frame::Grid>{}}};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_array_component<component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0);
  ActionTesting::emplace_array_component<elem_component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
      element_id);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const Mesh<3> mesh(5, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const LinkedMessageId<double> observation_time{2.0, {1.0}};
  auto& cache = ActionTesting::cache<elem_component>(runner, element_id);

  // Fill source vars with an analytic solution so that
  // ah::vars_to_interpolate_to_target can be computed from
  // the ah::source_vars. Previously, the source vars were all just set
  // to a constant value in this test. But in that case, the call to
  // FindApparentHorizon::apply() doesn't receive well-defined values, now
  // that it receives vars_to_interpolate_to_target instead of source_vars.
  Variables<ah::source_vars<3>> vars{};
  {
    const double mass = 1.0;
    const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
    const gr::Solutions::KerrSchild solution(mass, spin, {0.0, 0.0, 0.0});

    const auto& domain = Parallel::get<domain::Tags::Domain<3>>(cache);
    const auto& block = domain.blocks()[element_id.block_id()];

    const auto logical_coords = logical_coordinates(mesh);
    InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
        inv_jacobian_logical_to_inertial{mesh.number_of_grid_points(), 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{};
    if (block.is_time_dependent()) {
      const ElementMap<3, Frame::Grid> map_logical_to_grid{
          element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
      const auto& functions_of_time = domain_creator.functions_of_time();
      inertial_coords = block.moving_mesh_grid_to_inertial_map()(
          map_logical_to_grid(logical_coords), observation_time.id,
          functions_of_time);

      const auto inv_jacobian_logical_to_grid =
          map_logical_to_grid.inv_jacobian(logical_coords);
      const auto inv_jacobian_grid_to_inertial =
          block.moving_mesh_grid_to_inertial_map().inv_jacobian(
              map_logical_to_grid(logical_coords), observation_time.id,
              functions_of_time);
      inv_jacobian_logical_to_inertial = tenex::evaluate<ti::I, ti::j>(
          inv_jacobian_logical_to_grid(ti::I, ti::k) *
          inv_jacobian_grid_to_inertial(ti::K, ti::j));
    } else {
      const ElementMap<3, Frame::Inertial> map_logical_to_inertial{
          element_id, block.stationary_map().get_clone()};
      inertial_coords = map_logical_to_inertial(logical_coords);
      inv_jacobian_logical_to_inertial =
          map_logical_to_inertial.inv_jacobian(logical_coords);
    }

    const auto solution_vars = solution.variables(
        inertial_coords, observation_time.id,
        typename gr::Solutions::KerrSchild::tags<DataVector,
                                                 Frame::Inertial>{});

    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
    const auto& dt_lapse =
        get<Tags::dt<gr::Tags::Lapse<DataVector>>>(solution_vars);
    const auto& d_lapse = get<typename gr::Solutions::KerrSchild::DerivLapse<
        DataVector, Frame::Inertial>>(solution_vars);
    const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(solution_vars);
    const auto& dt_shift =
        get<Tags::dt<gr::Tags::Shift<DataVector, 3>>>(solution_vars);
    const auto& d_shift = get<typename gr::Solutions::KerrSchild::DerivShift<
        DataVector, Frame::Inertial>>(solution_vars);
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<DataVector, 3>>(solution_vars);
    const auto& dt_spatial_metric =
        get<Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(solution_vars);
    const auto& d_spatial_metric =
        get<typename gr::Solutions::KerrSchild::DerivSpatialMetric<
            DataVector, Frame::Inertial>>(solution_vars);

    vars.initialize(get(lapse).size());
    get<gr::Tags::SpacetimeMetric<DataVector, 3>>(vars) =
        gr::spacetime_metric(lapse, shift, spatial_metric);
    get<gh::Tags::Phi<DataVector, 3>>(vars) = gh::phi(
        lapse, d_lapse, shift, d_shift, spatial_metric, d_spatial_metric);
    get<gh::Tags::Pi<DataVector, 3>>(vars) =
        gh::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
               dt_spatial_metric, get<gh::Tags::Phi<DataVector, 3>>(vars));
    get<Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                    Frame::Inertial>>(vars) =
        partial_derivative(get<gh::Tags::Phi<DataVector, 3>>(vars), mesh,
                           inv_jacobian_logical_to_inertial);
  }
  std::optional<std::string> dependency{"FakeDependency"};

  // Test the event version
  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<metavars>,
      typename MockHorizonMetavars::time_tag, ::Events::Tags::ObserverMesh<3>,
      domain::Tags::Element<3>, ::Tags::Variables<ah::source_vars<3>>>>(
      metavars{}, observation_time, mesh, Element<3>{element_id, {}}, vars);

  const metavars::event event{dependency};
  const metavars::event serialized_event = serialize_and_deserialize(event);

  CHECK(serialized_event.needs_evolved_variables());

  auto obs_box = make_observation_box<
      typename metavars::event::compute_tags_for_observation_box>(
      make_not_null(&box));
  serialized_event.run(make_not_null(&obs_box), cache, element_id,
                       std::add_pointer_t<elem_component>{}, {});

  const auto check_results = [&]() {
    // Invoke all actions
    runner.invoke_queued_simple_action<component>(0);

    // No more queued simple actions.
    CHECK(runner.is_simple_action_queue_empty<component>(0));
    CHECK(runner.is_simple_action_queue_empty<elem_component>(element_id));

    const auto& results = MockFindApparentHorizon::results;
    CHECK(results.time == observation_time);
    CHECK(results.element_id == element_id);
    CHECK(results.mesh == mesh);

    Variables<ah::vars_to_interpolate_to_target<3, ::Frame::Grid>> target_vars{
        vars.number_of_grid_points()};
    const auto functions_of_time = domain_creator.functions_of_time();
    ah::compute_vars_to_interpolate_to_target(
        make_not_null(&target_vars),
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(vars),
        get<gh::Tags::Pi<DataVector, 3>>(vars),
        get<gh::Tags::Phi<DataVector, 3>>(vars),
        get<Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(vars),
        observation_time, domain_creator.create_domain(), mesh, element_id,
        functions_of_time);
    CHECK(results.vars == target_vars);

    CHECK(results.dependency == dependency);
  };

  check_results();

  MockFindApparentHorizon::results = MockFindApparentHorizon::Results{};
  dependency.reset();

  const auto option_event =
      TestHelpers::test_creation<typename metavars::event>("");
  const metavars::event serialized_option_event =
      serialize_and_deserialize(option_event);

  serialized_option_event.run(make_not_null(&obs_box), cache, element_id,
                              std::add_pointer_t<elem_component>{}, {});

  check_results();

  for (const bool enabled : {false, true}) {
    for (const bool moving : {false, true}) {
      test_rescaled_surface_event(enabled, moving, std::nullopt, false, true);
      test_rescaled_surface_event(enabled, moving, 1.0, false, true);
      test_rescaled_surface_event(enabled, moving, 1.0, true, true);
      // Publishing the horizon must not suppress a late diagnostic sender.
      test_rescaled_surface_event(enabled, moving, 2.0, true, true);
      test_rescaled_surface_event(enabled, moving, 3.0, true, true);
      test_rescaled_surface_event(enabled, moving, std::nullopt, false, false);
    }
  }
}
}  // namespace
