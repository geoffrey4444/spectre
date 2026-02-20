// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/Triggers/CompletionCriteria.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace test_tags {
struct ObserveCount : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<>;
  static type create_from_options() { return 0; }
};
}  // namespace test_tags

struct RecordObserveFieldsEvent : public Event {
 public:
  explicit RecordObserveFieldsEvent(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(RecordObserveFieldsEvent);  // NOLINT
#pragma GCC diagnostic pop

  using compute_tags_for_observation_box = tmpl::list<>;
  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Test event to mimic final ObserveFields output.";

  RecordObserveFieldsEvent() = default;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& /*array_index*/,
                  const Component* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    struct IncrementObserveCount {
      static void apply(const gsl::not_null<size_t*> count) { ++(*count); }
    };
    Parallel::mutate<test_tags::ObserveCount, IncrementObserveCount>(cache);
  }

  using is_ready_argument_tags = tmpl::list<>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }
};

PUP::able::PUP_ID RecordObserveFieldsEvent::my_PUP_ID = 0;  // NOLINT

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using const_global_cache_tags =
      tmpl::list<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks,
                 gh::bbh::Tags::MaxCommonHorizonSuccesses>;
  using mutable_global_cache_tags =
      tmpl::list<gh::bbh::Tags::GaugeConstraintExceeded,
                 gh::bbh::Tags::ThreeIndexConstraintExceeded,
                 gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold,
                 gh::bbh::Tags::CommonHorizonSuccessCount,
                 gh::bbh::Tags::MaxCommonHorizonSuccessesReached,
                 gh::bbh::Tags::CompletionRequested,
                 gh::bbh::Tags::StopSlabNumber, test_tags::ObserveCount>;

  using simple_tags = tmpl::list<Tags::Time, Tags::TimeStepId>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<
                     Parallel::Phase::Initialization,
                     tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
                 Parallel::PhaseActions<
                     Parallel::Phase::Testing,
                     tmpl::list<evolution::Actions::RunEventsAndTriggers<
                         Triggers::WhenToCheck::AtSlabs>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event,
                   tmpl::list<RecordObserveFieldsEvent, Events::Completion>>,
        tmpl::pair<Trigger,
                   tmpl::push_back<Triggers::logical_triggers,
                                   gh::bbh::Triggers::CompletionCriteria>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.GeneralizedHarmonic.BbhCompletionCriteriaSlabLatchAction",
    "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables>();
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      tuples::TaggedTuple<
          gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks,
          gh::bbh::Tags::MaxCommonHorizonSuccesses,
          Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>{
          size_t{0}, size_t{100},
          TestHelpers::test_creation<EventsAndTriggers, Metavariables>(
              "- Trigger: BbhCompletionCriteria\n"
              "  Events:\n"
              "    - RecordObserveFieldsEvent\n"
              "    - Completion\n")},
      tuples::TaggedTuple<gh::bbh::Tags::GaugeConstraintExceeded,
                          gh::bbh::Tags::ThreeIndexConstraintExceeded,
                          gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold,
                          gh::bbh::Tags::CommonHorizonSuccessCount,
                          gh::bbh::Tags::MaxCommonHorizonSuccessesReached,
                          gh::bbh::Tags::CompletionRequested,
                          gh::bbh::Tags::StopSlabNumber,
                          test_tags::ObserveCount>{
          false, false, false, size_t{0}, false, false,
          std::numeric_limits<size_t>::max(), size_t{0}}};

  ActionTesting::emplace_component<component>(&runner, 0);
  ActionTesting::emplace_component<component>(&runner, 1);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto& cache = ActionTesting::cache<component>(runner, 0);
  Parallel::mutate<gh::bbh::Tags::CompletionRequested,
                   gh::bbh::Mutators::SetCompletionRequested>(cache);

  const Slab slab(0.0, 1.0);
  for (const int idx : std::array{0, 1}) {
    auto& box =
        ActionTesting::get_databox<component>(make_not_null(&runner), idx);
    db::mutate<Tags::TimeStepId, Tags::Time>(
        [&slab](const gsl::not_null<TimeStepId*> time_step_id,
                const gsl::not_null<double*> time) {
          *time_step_id = TimeStepId(true, 7, slab.start());
          *time = slab.start().value();
        },
        make_not_null(&box));
    ActionTesting::next_action<component>(make_not_null(&runner), idx);
  }

  CHECK_FALSE(ActionTesting::get_terminate<component>(runner, 0));
  CHECK_FALSE(ActionTesting::get_terminate<component>(runner, 1));
  CHECK(Parallel::get<gh::bbh::Tags::StopSlabNumber>(cache) == 8_st);
  CHECK(Parallel::get<test_tags::ObserveCount>(cache) == 0_st);

  for (const int idx : std::array{0, 1}) {
    auto& box =
        ActionTesting::get_databox<component>(make_not_null(&runner), idx);
    db::mutate<Tags::TimeStepId, Tags::Time>(
        [&slab](const gsl::not_null<TimeStepId*> time_step_id,
                const gsl::not_null<double*> time) {
          *time_step_id = TimeStepId(true, 8, slab.start());
          *time = slab.start().value();
        },
        make_not_null(&box));
    ActionTesting::next_action<component>(make_not_null(&runner), idx);
  }

  CHECK(ActionTesting::get_terminate<component>(runner, 0));
  CHECK(ActionTesting::get_terminate<component>(runner, 1));
  CHECK(Parallel::get<test_tags::ObserveCount>(cache) == 2_st);
  CHECK(Parallel::get<gh::bbh::Tags::StopSlabNumber>(cache) == 8_st);
}
