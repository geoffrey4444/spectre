// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <optional>

#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Creators/SphericalShells.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/ProjectRadialLocalSource.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/Phase.hpp"

namespace {
using VarsTag = typename gh::System<3>::variables_tag;
using Vars = typename VarsTag::type;
using DtTag = db::add_tag_prefix<::Tags::dt, VarsTag>;
using DtVars = typename DtTag::type;
using GridMapTag = domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                               Frame::Inertial>;
using SimpleTags = tmpl::list<
    VarsTag, DtTag, domain::Tags::Element<3>, gh::Tags::ConstraintGamma1,
    domain::Tags::Mesh<3>, ::Tags::Time,
    domain::Tags::Coordinates<3, Frame::ElementLogical>,
    domain::Tags::ElementMap<3, Frame::Grid>, GridMapTag,
    domain::Tags::FunctionsOfTime,
    domain::Tags::Coordinates<3, Frame::Inertial>,
    domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>,
    domain::Tags::MeshVelocity<3, Frame::Inertial>>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<gh::Tags::RadialSourceProjection,
                                             gh::gauges::Tags::GaugeCondition>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<
                     Parallel::Phase::Initialization,
                     tmpl::list<ActionTesting::InitializeDataBox<SimpleTags>>>,
                 Parallel::PhaseActions<
                     Parallel::Phase::Testing,
                     tmpl::list<gh::Actions::ProjectRadialLocalSource>>>;
};
struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
};

// The deliberately invalid state/maps/gamma1 are never needed by either
// disabled or unselected-block paths. The entire existing RHS must stay exact.
void test_skip(
    const std::optional<gh::RadialSourceProjectionOptions>& settings) {
  using Component = ElementArray<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {settings, std::unique_ptr<gh::gauges::GaugeCondition>{}}};
  const DtVars original_rhs(1, 3.125);
  ActionTesting::emplace_component_and_initialize<Component>(
      &runner, 0,
      {Vars(1, 9.0), original_rhs, Element<3>{ElementId<3>{0}, {}},
       Scalar<DataVector>(size_t{1}, 9.0), Mesh<3>{}, 0.0,
       tnsr::I<DataVector, 3, Frame::ElementLogical>{},
       ElementMap<3, Frame::Grid>{}, typename GridMapTag::type{},
       domain::FunctionsOfTimeMap{}, tnsr::I<DataVector, 3, Frame::Inertial>{},
       InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>{},
       std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>{}});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<Component>(make_not_null(&runner), 0);
  const auto& after =
      ActionTesting::get_databox_tag<Component, DtTag>(runner, 0);
  REQUIRE(after.size() == original_rhs.size());
  for (size_t i = 0; i < after.size(); ++i) {
    CHECK(after.data()[i] == original_rhs.data()[i]);
  }
}

void test_enabled_flat_source() {
  using Component = ElementArray<Metavariables>;
  const std::optional<gh::RadialSourceProjectionOptions> settings{
      gh::RadialSourceProjectionOptions{32, {0}, true}};
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {settings, std::unique_ptr<gh::gauges::GaugeCondition>{
                     std::make_unique<gh::gauges::DampedHarmonic>(
                         100.0, std::array<double, 3>{1., 1., 1.},
                         std::array<int, 3>{2, 2, 2})}}};
  const Mesh<3> mesh{
      {{4, 3, 5}},
      {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
        Spectral::Quadrature::Equiangular}}};
  const domain::creators::SphericalShells creator{
      2.,
      3.,
      0,
      size_t{4},
      2,
      {},
      domain::CoordinateMaps::Distribution::Linear};
  const auto domain = creator.create_domain();
  const ElementId<3> id{0};
  const ElementMap<3, Frame::Inertial> to_inertial{id, domain.blocks()[0]};
  const auto logical = logical_coordinates(mesh);
  Vars state(mesh.number_of_grid_points(), 0.0);
  auto& metric = get<gr::Tags::SpacetimeMetric<DataVector, 3>>(state);
  metric.get(0, 0) = -1.;
  for (size_t i = 1; i < 4; ++i) {
    metric.get(i, i) = 1.;
  }
  const DtVars original_rhs(mesh.number_of_grid_points(), 3.125);
  ActionTesting::emplace_component_and_initialize<Component>(
      &runner, 0,
      {state, original_rhs, Element<3>{id, {}},
       Scalar<DataVector>(mesh.number_of_grid_points(), -1.), mesh, 0.0,
       logical, ElementMap<3, Frame::Grid>{id, domain.blocks()[0]},
       domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
           domain::CoordinateMaps::Identity<3>{}),
       domain::FunctionsOfTimeMap{}, to_inertial(logical),
       to_inertial.inv_jacobian(logical),
       std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>{}});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<Component>(make_not_null(&runner), 0);
  const auto& after =
      ActionTesting::get_databox_tag<Component, DtTag>(runner, 0);
  const auto& after_state =
      ActionTesting::get_databox_tag<Component, VarsTag>(runner, 0);
  for (size_t i = 0; i < after.size(); ++i) {
    CHECK(after.data()[i] == approx(original_rhs.data()[i]));
    CHECK(after_state.data()[i] == state.data()[i]);
  }
}

void test_options() {
  using Tag = gh::OptionTags::RadialSourceProjection;
  Options::Parser<tmpl::list<Tag>> parser;
  parser.parse(
      "EvolutionSystem:\n  GeneralizedHarmonic:\n"
      "    RadialSourceProjection: None\n");
  const std::optional<gh::RadialSourceProjectionOptions> disabled =
      parser.get<Tag>();
  CHECK_FALSE(disabled.has_value());
  const gh::RadialSourceProjectionOptions options{40, {2, 0}, true};
  const auto restored = serialize_and_deserialize(options);
  CHECK(restored.evaluation_points == 40);
  CHECK(restored.block_ids == std::vector<size_t>{0, 2});
  CHECK(restored.preserve_inner_source);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Gh.ProjectRadialLocalSource",
                  "[Unit][Evolution]") {
  test_options();
  test_skip(std::nullopt);
  test_skip(gh::RadialSourceProjectionOptions{32, {1}, true});
  test_enabled_flat_source();
}
