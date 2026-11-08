// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/SpectralFilter.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using Vars = Variables<tmpl::list<ScalarFieldTag>>;
using FilterBase = Filters::Filter<1, tmpl::list<ScalarFieldTag>>;

// Minimal concrete filter for testing the action.
//
// apply_in_volume multiplies each component of the Variables by 2 and records
// whether the optional Jacobians were populated.
class MockFilter : public FilterBase {
 public:
  MockFilter() = default;
  MockFilter(bool apply_substep, bool apply_this_step, bool need_jacs,
             bool supports_mesh = true)
      : apply_substep_(apply_substep),
        apply_this_step_(apply_this_step),
        need_jacs_(need_jacs),
        supports_mesh_(supports_mesh) {}

  explicit MockFilter(CkMigrateMessage* m) : FilterBase(m) {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  // NOLINTNEXTLINE
  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(FilterBase), MockFilter);
#pragma GCC diagnostic pop

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    FilterBase::pup(p);
    p | apply_substep_;
    p | apply_this_step_;
    p | need_jacs_;
    p | supports_mesh_;
    p | saw_inv_jac_;
    p | saw_jac_;
    p | inv_jac_values_;
    p | jac_values_;
  }

  std::unique_ptr<FilterBase> get_clone() const override {
    return std::make_unique<MockFilter>(*this);
  }

  bool apply_volume_filter_on_substep() const override {
    return apply_substep_;
  }
  bool apply_volume_filter_on_this_step(size_t /*step_number*/) const override {
    return apply_this_step_;
  }
  bool apply_boundary_filter_on_substep() const override { return false; }
  bool apply_boundary_filter_on_this_step(
      size_t /*step_number*/) const override {
    return false;
  }
  bool need_jacobians() const override { return need_jacs_; }
  bool supports_mesh(const Mesh<1>& /*mesh*/) const override {
    return supports_mesh_;
  }
  std::string name() const override { return "MockFilter"; }
  bool is_equal(const FilterBase& other) const override {
    const auto* rhs = dynamic_cast<const MockFilter*>(&other);
    return rhs != nullptr and apply_substep_ == rhs->apply_substep_ and
           apply_this_step_ == rhs->apply_this_step_ and
           need_jacs_ == rhs->need_jacs_;
  }

  const std::optional<std::vector<size_t>>& blocks_to_filter() const override {
    return blocks_to_filter_;
  }
  void set_blocks_to_filter(
      const std::vector<std::string>& /*all_block_names*/,
      const std::unordered_map<std::string, std::unordered_set<std::string>>&
      /*block_groups*/) override {}

  void apply_in_volume(
      gsl::not_null<Vars*> vars, const Mesh<1>& /*mesh*/,
      const std::optional<InverseJacobian<DataVector, 1, Frame::Grid,
                                          Frame::Inertial>>& inv_jac,
      const std::optional<Jacobian<DataVector, 1, Frame::Grid,
                                   Frame::Inertial>>& jac) const override {
    saw_inv_jac_ = inv_jac.has_value();
    saw_jac_ = jac.has_value();
    if (inv_jac.has_value()) {
      inv_jac_values_ = get<0, 0>(*inv_jac);
    }
    if (jac.has_value()) {
      jac_values_ = get<0, 0>(*jac);
    }
    for (auto& component : get<ScalarFieldTag>(*vars)) {
      component *= 2.0;
    }
  }

  void apply_on_boundary(
      gsl::not_null<Vars*> /*vars*/, const Mesh<0>& /*mesh*/,
      const std::optional<
          InverseJacobian<DataVector, 1, Frame::Grid, Frame::Inertial>>&
      /*inv_jac*/,
      const std::optional<
          Jacobian<DataVector, 1, Frame::Grid, Frame::Inertial>>&
      /*jac*/) const override {}

  bool saw_inv_jac() const { return saw_inv_jac_; }
  bool saw_jac() const { return saw_jac_; }
  const DataVector& inv_jac_values() const { return inv_jac_values_; }
  const DataVector& jac_values() const { return jac_values_; }

 private:
  bool apply_substep_{false};
  bool apply_this_step_{false};
  bool need_jacs_{false};
  bool supports_mesh_{true};
  std::optional<std::vector<size_t>> blocks_to_filter_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable bool saw_inv_jac_{false};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable bool saw_jac_{false};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable DataVector inv_jac_values_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable DataVector jac_values_{};
};

// NOLINTNEXTLINE
PUP::able::PUP_ID MockFilter::my_PUP_ID = 0;

using VarsTag = ::Tags::Variables<tmpl::list<ScalarFieldTag>>;
using InvJacTag =
    domain::Tags::InverseJacobian<1, Frame::Grid, Frame::Inertial>;
using JacTag = domain::Tags::Jacobian<1, Frame::Grid, Frame::Inertial>;
using FilterTag = Filters::Tags::SpectralFilter<1, tmpl::list<ScalarFieldTag>>;
using StepTag = ::Tags::StepNumberWithinSlab;
using NextTimeStepIdTag = ::Tags::Next<::Tags::TimeStepId>;
using MeshTag = domain::Tags::Mesh<1>;
using ElementTag = domain::Tags::Element<1>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<
                     Parallel::Phase::Initialization,
                     tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<
                         VarsTag, MeshTag, ElementTag, StepTag, ::Tags::Time,
                         ::Tags::TimeStepId, NextTimeStepIdTag, FilterTag,
                         InvJacTag, JacTag>>>>,
                 Parallel::PhaseActions<
                     Parallel::Phase::Testing,
                     tmpl::list<typename Metavariables::filter_action>>>;
};

template <bool BeforeUpdate>
struct Metavariables {
  using filter_action = tmpl::conditional_t<
      BeforeUpdate,
      dg::Actions::SpectralFilter<1, tmpl::list<ScalarFieldTag>,
                                  ::Tags::TimeStepId>,
      dg::Actions::SpectralFilter<1, tmpl::list<ScalarFieldTag>>>;
  using component_list = tmpl::list<ElementArray<Metavariables>>;
};

InverseJacobian<DataVector, 1, Frame::Grid, Frame::Inertial>
make_identity_inv_jac() {
  return InverseJacobian<DataVector, 1, Frame::Grid, Frame::Inertial>{5_st,
                                                                      1.0};
}

Jacobian<DataVector, 1, Frame::Grid, Frame::Inertial> make_identity_jac() {
  return Jacobian<DataVector, 1, Frame::Grid, Frame::Inertial>{5_st, 1.0};
}

template <bool BeforeUpdate = false, typename Filter, typename Callback>
void run_action(std::unique_ptr<Filter> filter_ptr, const double initial_value,
                const uint64_t step_number, Callback&& callback,
                const TimeStepId& next_time_step_id =
                    TimeStepId{true, 0, Slab{0., 1.}.start()},
                const TimeStepId& time_step_id = TimeStepId{
                    true, 0, Slab{0., 1.}.start()}) {
  using element_array = ElementArray<Metavariables<BeforeUpdate>>;
  ActionTesting::MockRuntimeSystem<Metavariables<BeforeUpdate>> runner{{}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, 0,
      {Vars(5, initial_value),
       Mesh<1>{5, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto},
       Element<1>{ElementId<1>{0}, {}, domain::topologies::hypercube<1>},
       step_number, time_step_id.substep_time(), time_step_id,
       next_time_step_id, std::unique_ptr<FilterBase>(std::move(filter_ptr)),
       make_identity_inv_jac(), make_identity_jac()});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), 0);
  std::forward<Callback>(callback)(runner);
}

template <typename Metavariables>
DataVector get_scalar_field(
    const ActionTesting::MockRuntimeSystem<Metavariables>& runner) {
  return get(get<ScalarFieldTag>(
      ActionTesting::get_databox_tag<
          tmpl::front<typename Metavariables::component_list>, VarsTag>(runner,
                                                                        0)));
}

template <typename Metavariables>
const MockFilter* get_mock_filter(
    const ActionTesting::MockRuntimeSystem<Metavariables>& runner) {
  const auto& filter_ref = ActionTesting::get_databox_tag<
      tmpl::front<typename Metavariables::component_list>, FilterTag>(runner,
                                                                      0);
  return dynamic_cast<const MockFilter*>(&filter_ref);
}

using MapTag = domain::CoordinateMaps::Tags::CoordinateMap<1, Frame::Grid,
                                                           Frame::Inertial>;
using GridCoordinatesTag = domain::Tags::Coordinates<1, Frame::Grid>;
using ScaleMap =
    domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                          domain::CoordinateMaps::TimeDependent::CubicScale<1>>;
using IdentityMap = domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                          domain::CoordinateMaps::Identity<1>>;
using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<1>;

template <typename Metavariables>
struct MovingMapElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              tmpl::list<VarsTag, MeshTag, ElementTag, StepTag, ::Tags::Time,
                         ::Tags::TimeStepId, NextTimeStepIdTag, FilterTag,
                         InvJacTag, JacTag, MapTag, GridCoordinatesTag>,
              tmpl::list<Parallel::Tags::FromGlobalCache<
                  domain::Tags::FunctionsOfTimeInitialize, Metavariables>>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<typename Metavariables::filter_action>>>;
};

template <bool BeforeUpdate>
struct MovingMapMetavariables : Metavariables<BeforeUpdate> {
  using component_list =
      tmpl::list<MovingMapElementArray<MovingMapMetavariables>>;
};

template <bool BeforeUpdate = false, typename Callback>
void with_moving_map(const TimeStepId& time_step_id,
                     const TimeStepId& next_time_step_id,
                     const double expiration_time,
                     std::unique_ptr<FilterBase> filter, MapTag::type map,
                     Callback&& callback) {
  using metavars = MovingMapMetavariables<BeforeUpdate>;
  using component = MovingMapElementArray<metavars>;
  domain::Tags::FunctionsOfTime::type functions_of_time{};
  functions_of_time["Expansion"] = std::make_unique<Polynomial>(
      -0.5, std::array{DataVector{0.5}, DataVector{1.}}, expiration_time);
  // An unrelated expired function must not delay this map's filter.
  functions_of_time["Unrelated"] = std::make_unique<Polynomial>(
      -0.5, std::array{DataVector{0.}, DataVector{0.}}, 0.);
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {}, {std::move(functions_of_time)}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {Vars(5, 3.),
       Mesh<1>{5, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto},
       Element<1>{ElementId<1>{0}, {}, domain::topologies::hypercube<1>},
       uint64_t{0}, time_step_id.substep_time(), time_step_id,
       next_time_step_id, std::move(filter),
       // Deliberately distinct cached matrices identify the fast path.
       InvJacTag::type{5_st, 0.5}, JacTag::type{5_st, 2.}, std::move(map),
       GridCoordinatesTag::type{5_st, 1.}});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  std::forward<Callback>(callback)(make_not_null(&runner),
                                   std::type_identity<component>{});
}

std::unique_ptr<ScaleMap> make_scale_map() {
  return std::make_unique<ScaleMap>(
      domain::CoordinateMaps::TimeDependent::CubicScale<1>{10., "Expansion",
                                                           "Expansion"});
}

template <bool BeforeUpdate>
void test_jacobian_times() {
  const Slab slab{0., 1.};
  const TimeStepId step_start{true, 0, slab.start()};
  const auto step_size = slab.duration() / 4;
  const auto predictor_id = step_start.next_substep(step_size, 1.);
  const auto corrector_id = predictor_id.next_step(step_size);
  for (const auto& current_id : {step_start, predictor_id}) {
    const auto& next_id =
        current_id.substep() == 0 ? predictor_id : corrector_id;
    const bool cached = BeforeUpdate or current_id == predictor_id;
    CAPTURE(BeforeUpdate, current_id, next_id);
    with_moving_map<BeforeUpdate>(
        current_id, next_id, 1., std::make_unique<MockFilter>(true, true, true),
        make_scale_map(),
        [cached, current_time = current_id.substep_time()](const auto runner,
                                                           auto component_v) {
          using component = typename decltype(component_v)::type;
          CHECK(ActionTesting::next_action_if_ready<component>(runner, 0));
          CHECK(get_scalar_field(*runner) == DataVector(5, 6.));
          const auto* filter = get_mock_filter(*runner);
          REQUIRE(filter != nullptr);
          CHECK(filter->jac_values() == DataVector(5, cached ? 2. : 1.25));
          CHECK(filter->inv_jac_values() == DataVector(5, cached ? 0.5 : 0.8));
          CHECK(get<0, 0>(ActionTesting::get_databox_tag<component, JacTag>(
                    *runner, 0)) == DataVector(5, 2.));
          CHECK(get<0, 0>(ActionTesting::get_databox_tag<component, InvJacTag>(
                    *runner, 0)) == DataVector(5, 0.5));
          CHECK(ActionTesting::get_databox_tag<component, ::Tags::Time>(
                    *runner, 0) == current_time);
        });
  }
}

void test_static_map_uses_cached_jacobians() {
  const Slab slab{0., 1.};
  const TimeStepId step_start{true, 0, slab.start()};
  with_moving_map(
      step_start, step_start.next_substep(slab.duration() / 4, 1.), 0.,
      std::make_unique<MockFilter>(true, true, true),
      std::make_unique<IdentityMap>(), [](const auto runner, auto component_v) {
        using component = typename decltype(component_v)::type;
        CHECK(ActionTesting::next_action_if_ready<component>(runner, 0));
        CHECK(get_scalar_field(*runner) == DataVector(5, 6.));
        const auto* filter = get_mock_filter(*runner);
        REQUIRE(filter != nullptr);
        CHECK(filter->jac_values() == DataVector(5, 2.));
        CHECK(filter->inv_jac_values() == DataVector(5, 0.5));
      });
}

struct ExtendExpansion {
  static void apply(
      const gsl::not_null<domain::Tags::FunctionsOfTime::type*> functions) {
    functions->at("Expansion")->update(0.125, DataVector{1.}, 1.);
  }
};

void test_wait_for_future_jacobians() {
  const Slab slab{0., 1.};
  const TimeStepId step_start{true, 0, slab.start()};
  with_moving_map(
      step_start, step_start.next_substep(slab.duration() / 4, 1.), 0.125,
      std::make_unique<MockFilter>(true, true, true), make_scale_map(),
      [](const auto runner, auto component_v) {
        using component = typename decltype(component_v)::type;
        CHECK(not ActionTesting::next_action_if_ready<component>(runner, 0));
        CHECK(get_scalar_field(*runner) == DataVector(5, 3.));
        const auto* filter = get_mock_filter(*runner);
        REQUIRE(filter != nullptr);
        CHECK(not filter->saw_jac());
        auto& cache = ActionTesting::cache<component>(*runner, 0);
        Parallel::mutate<domain::Tags::FunctionsOfTime, ExtendExpansion>(cache);
        CHECK(ActionTesting::next_action_if_ready<component>(runner, 0));
        CHECK(get_scalar_field(*runner) == DataVector(5, 6.));
        CHECK(filter->jac_values() == DataVector(5, 1.25));
        CHECK(filter->inv_jac_values() == DataVector(5, 0.8));
      });
}

void test_no_jacobians_skip_future_map() {
  const Slab slab{0., 1.};
  const TimeStepId step_start{true, 0, slab.start()};
  for (const bool no_filter : {false, true}) {
    std::unique_ptr<FilterBase> filter =
        no_filter
            ? std::unique_ptr<FilterBase>{std::make_unique<
                  Filters::None<1, tmpl::list<ScalarFieldTag>>>(std::nullopt)}
            : std::unique_ptr<FilterBase>{
                  std::make_unique<MockFilter>(true, true, false)};
    with_moving_map(
        step_start, step_start.next_substep(slab.duration() / 4, 1.), 0.,
        std::move(filter), make_scale_map(),
        [no_filter](const auto runner, auto component_v) {
          using component = typename decltype(component_v)::type;
          CHECK(ActionTesting::next_action_if_ready<component>(runner, 0));
          CHECK(get_scalar_field(*runner) ==
                DataVector(5, no_filter ? 3. : 6.));
        });
  }
}

template <bool BeforeUpdate>
void test_unused_self_start_step() {
  const Slab slab{0., 1.};
  const TimeStepId current_id{true, -2, slab.start() + slab.duration() / 4};
  const TimeStepId reset_id{true, -1, slab.start()};
  with_moving_map<BeforeUpdate>(
      current_id, reset_id, 0., std::make_unique<MockFilter>(true, true, true),
      make_scale_map(), [](const auto runner, auto component_v) {
        using component = typename decltype(component_v)::type;
        CHECK(ActionTesting::next_action_if_ready<component>(runner, 0));
        CHECK(get_scalar_field(*runner) ==
              DataVector(5, BeforeUpdate ? 6. : 3.));
      });
}

void test_none_filter_skips() {
  run_action(std::make_unique<Filters::None<1, tmpl::list<ScalarFieldTag>>>(
                 std::nullopt),
             3.0, 0, [](const auto& runner) {
               CHECK(get_scalar_field(runner) == DataVector(5, 3.0));
             });
}

// Test: MockFilter with both gating flags false does not apply.
void test_no_gating_flags_skips() {
  run_action(std::make_unique<MockFilter>(/*apply_substep=*/false,
                                          /*apply_this_step=*/false,
                                          /*need_jacs=*/false),
             3.0, 0, [](const auto& runner) {
               CHECK(get_scalar_field(runner) == DataVector(5, 3.0));
             });
}

// Test: apply_volume_filter_on_substep=true → filter applies.
void test_substep_flag_applies_filter() {
  run_action(std::make_unique<MockFilter>(/*apply_substep=*/true,
                                          /*apply_this_step=*/false,
                                          /*need_jacs=*/false),
             3.0, 0, [](const auto& runner) {
               CHECK(get_scalar_field(runner) == DataVector(5, 6.0));
             });
}

// Test: apply_volume_filter_on_this_step=true → filter applies.
void test_step_flag_applies_filter() {
  run_action(std::make_unique<MockFilter>(/*apply_substep=*/false,
                                          /*apply_this_step=*/true,
                                          /*need_jacs=*/false),
             3.0, 0, [](const auto& runner) {
               CHECK(get_scalar_field(runner) == DataVector(5, 6.0));
             });
}

// Exercise both action placements: before UpdateU the variables belong to
// TimeStepId, and after UpdateU they belong to Next<TimeStepId>. Keep both
// tags in the box with different substeps to catch selecting the wrong one.
// A predictor at the end time of a step remains a substep; only the corrected
// result is a full step.
template <bool BeforeUpdate>
void test_filter_cadence() {
  const Slab slab{0., 1.};
  const TimeStepId step_start{true, 0, slab.start()};
  const auto step_size = slab.duration() / 4;
  const auto predictor_id = step_start.next_substep(step_size, 1.);
  const auto corrector_id = predictor_id.next_step(step_size);
  for (const auto& time_step_id : {step_start, predictor_id}) {
    const auto& next_time_step_id =
        time_step_id.substep() == 0 ? predictor_id : corrector_id;
    for (const bool apply_substep : {false, true}) {
      for (const bool apply_this_step : {false, true}) {
        CAPTURE(BeforeUpdate, time_step_id, next_time_step_id, apply_substep,
                apply_this_step);
        const bool at_full_step = BeforeUpdate
                                      ? time_step_id.substep() == 0
                                      : next_time_step_id.substep() == 0;
        const bool should_filter =
            apply_substep or (at_full_step and apply_this_step);
        run_action<BeforeUpdate>(
            std::make_unique<MockFilter>(apply_substep, apply_this_step,
                                         /*need_jacs=*/false),
            3., 2,
            [should_filter](const auto& runner) {
              CHECK(get_scalar_field(runner) ==
                    DataVector(5, should_filter ? 6. : 3.));
            },
            next_time_step_id, time_step_id);
      }
    }
  }
}

// Test: need_jacobians=true → Jacobians populated in apply_in_volume.
void test_jacobians_passed_when_needed() {
  run_action(std::make_unique<MockFilter>(/*apply_substep=*/true,
                                          /*apply_this_step=*/false,
                                          /*need_jacs=*/true),
             3.0, 0, [](const auto& runner) {
               CHECK(get_scalar_field(runner) == DataVector(5, 6.0));
               const MockFilter* mock = get_mock_filter(runner);
               REQUIRE(mock != nullptr);
               CHECK(mock->saw_inv_jac());
               CHECK(mock->saw_jac());
             });
}

// Test: need_jacobians=false → Jacobians are nullopt in apply_in_volume.
void test_jacobians_not_passed_when_not_needed() {
  run_action(std::make_unique<MockFilter>(/*apply_substep=*/true,
                                          /*apply_this_step=*/false,
                                          /*need_jacs=*/false),
             3.0, 0, [](const auto& runner) {
               CHECK(get_scalar_field(runner) == DataVector(5, 6.0));
               const MockFilter* mock = get_mock_filter(runner);
               REQUIRE(mock != nullptr);
               CHECK(not mock->saw_inv_jac());
               CHECK(not mock->saw_jac());
             });
}

// Test: a filter that does not support the mesh triggers an ERROR naming the
// filter.
void test_unsupported_mesh_errors() {
  CHECK_THROWS_WITH(
      run_action(std::make_unique<MockFilter>(/*apply_substep=*/true,
                                              /*apply_this_step=*/false,
                                              /*need_jacs=*/false,
                                              /*supports_mesh=*/false),
                 3.0, 0, [](const auto& /*runner*/) {}),
      Catch::Matchers::ContainsSubstring("MockFilter") and
          Catch::Matchers::ContainsSubstring("does not support"));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Actions.SpectralFilter",
                  "[Unit][ParallelAlgorithms][Actions]") {
  register_classes_with_charm<MockFilter,
                              Filters::None<1, tmpl::list<ScalarFieldTag>>,
                              ScaleMap, IdentityMap, Polynomial>();
  test_none_filter_skips();
  test_no_gating_flags_skips();
  test_substep_flag_applies_filter();
  test_step_flag_applies_filter();
  test_filter_cadence<false>();
  test_filter_cadence<true>();
  test_jacobians_passed_when_needed();
  test_jacobians_not_passed_when_not_needed();
  test_jacobian_times<false>();
  test_jacobian_times<true>();
  test_static_map_uses_cached_jacobians();
  test_wait_for_future_jacobians();
  test_no_jacobians_skip_future_map();
  test_unused_self_start_step<false>();
  test_unused_self_start_step<true>();
  test_unsupported_mesh_errors();
}
