// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var0 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var1 : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};

struct MockContributeReductionData {
  struct Results {
    observers::ObservationId observation_id;
    std::string subfile_name;
    std::vector<std::string> reduction_names;
    double time;
    size_t number_of_grid_points;
    std::vector<double> max_values;
    std::vector<double> min_values;
    std::vector<double> l2_norm_values;
  };
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static Results results;

  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex, typename... Ts>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id,
                    observers::ArrayComponentId /*sender_array_id*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& reduction_names,
                    Parallel::ReductionData<Ts...>&& reduction_data) noexcept {
    reduction_data.finalize();
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.reduction_names = reduction_names;
    results.time = std::get<0>(reduction_data.data());
    results.number_of_grid_points = std::get<1>(reduction_data.data());
    results.max_values = std::get<2>(reduction_data.data());
    results.min_values = std::get<3>(reduction_data.data());
    results.l2_norm_values = std::get<4>(reduction_data.data());
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
MockContributeReductionData::Results MockContributeReductionData::results{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

template <typename Metavariables>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeReductionData>;
  using with_these_simple_actions = tmpl::list<MockContributeReductionData>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event, tmpl::list<Events::ObserveNorms<Var0, Var1>>>>;
  };

  enum class Phase { Initialization, Testing, Exit };
};

template <typename ObserveEvent>
void test(const std::unique_ptr<ObserveEvent> observe) {
  using metavariables = Metavariables;
  using element_component = ElementComponent<metavariables>;
  using observer_component = MockObserverComponent<metavariables>;
  const typename element_component::array_index array_index(0);
  const size_t num_points = 5;
  const double observation_time = 2.0;
  Variables<tmpl::list<Var0, Var1>> vars(num_points);
  // Fill the variables with some data.  It doesn't matter much what,
  // but integers are nice in that we don't have to worry about
  // roundoff error.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::iota(vars.data(), vars.data() + vars.size(), 1.0);

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      0);
  ActionTesting::emplace_group_component<observer_component>(&runner);

  const auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<metavariables>, ::Tags::Time,
      Tags::Variables<typename decltype(vars)::tags_list>>>(
      metavariables{}, observation_time, vars);

  const auto ids_to_register =
      observers::get_registration_observation_type_and_key(*observe, box);
  const observers::ObservationKey expected_observation_key_for_reg(
      "/reduction0.dat");
  CHECK(ids_to_register->first == observers::TypeOfObservation::Reduction);
  CHECK(ids_to_register->second == expected_observation_key_for_reg);

  observe->run(box,
               ActionTesting::cache<element_component>(runner, array_index),
               array_index, std::add_pointer_t<element_component>{});

  // Process the data
  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& results = MockContributeReductionData::results;
  CHECK(results.observation_id.value() == observation_time);
  CHECK(results.subfile_name == "/reduction0");
  CHECK(results.reduction_names[0] == "Time");
  CHECK(results.time == observation_time);
  CHECK(results.reduction_names[1] == "NumberOfPoints");
  CHECK(results.number_of_grid_points == num_points);
  // Check max values
  CHECK(results.reduction_names[2] == "Max(Var0)");
  CHECK(results.reduction_names[3] == "Max(Var0)");
  CHECK(results.max_values == std::vector<double>{5.0, 5.0});

  // Check min values
  CHECK(results.reduction_names[4] == "Min(Var1_x)");
  CHECK(results.reduction_names[5] == "Min(Var1_y)");
  CHECK(results.reduction_names[6] == "Min(Var1_z)");
  CHECK(results.reduction_names[7] == "Min(Var1)");
  CHECK(results.min_values == std::vector<double>{6.0, 11.0, 16.0, 6.0});

  // Check L2 norms
  CHECK(results.reduction_names[8] == "L2Norm(Var1)");
  CHECK(results.reduction_names[9] == "L2Norm(Var1_x)");
  CHECK(results.reduction_names[10] == "L2Norm(Var1_y)");
  CHECK(results.reduction_names[11] == "L2Norm(Var1_z)");
  CHECK(results.l2_norm_values[0] == approx(23.72762103540934575));
  CHECK(results.l2_norm_values[1] == approx(8.12403840463596083));
  CHECK(results.l2_norm_values[2] == approx(13.076696830622021));
  CHECK(results.l2_norm_values[3] == approx(18.055470085267789));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ObserveNorms", "[Unit][Evolution]") {
  test(std::make_unique<Events::ObserveNorms<Var0, Var1>>(
      Events::ObserveNorms<Var0, Var1>{"reduction0",
                                       {{"Var0", "Max", "Individual"},
                                        {"Var1", "Min", "Individual"},
                                        {"Var0", "Max", "Sum"},
                                        {"Var1", "L2Norm", "Sum"},
                                        {"Var1", "L2Norm", "Individual"},
                                        {"Var1", "Min", "Sum"}}}));

  INFO("create/serialize");
  Parallel::register_factory_classes_with_charm<Metavariables>();
  const auto factory_event =
      TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables>(
          "ObserveNorms:\n"
          "  SubfileName: reduction0\n"
          "  TensorsToObserve:\n"
          "  - Name: Var0\n"
          "    NormType: Max\n"
          "    Components: Individual\n"
          "  - Name: Var1\n"
          "    NormType: Min\n"
          "    Components: Individual\n"
          "  - Name: Var0\n"
          "    NormType: Max\n"
          "    Components: Sum\n"
          "  - Name: Var1\n"
          "    NormType: L2Norm\n"
          "    Components: Sum\n"
          "  - Name: Var1\n"
          "    NormType: L2Norm\n"
          "    Components: Individual\n"
          "  - Name: Var1\n"
          "    NormType: Min\n"
          "    Components: Sum\n");
  auto serialized_event = serialize_and_deserialize(factory_event);
  test(std::move(serialized_event));
}
