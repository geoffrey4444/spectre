// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/BoundaryHistory.tpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/UpdateLtsStepperError.hpp"
#include "Utilities/Gsl.hpp"

namespace {
struct Var : db::SimpleTag {
  using type = double;
};
struct System {
  using variables_tag = Var;
};
using Update = UpdateLtsStepperError<System>;

void test_boundary_error() {
  const Slab slab{0., 1.};
  const TimeSteppers::AdamsMoultonPc<true> stepper{4};
  const StepperErrorTolerances tolerances{
      StepperErrorTolerances::Estimates::StepperOrder, 1., 0.};
  for (const int denominator : {16, 32, 64, 128, 256}) {
    CAPTURE(denominator);
    const TimeDelta dt{slab, {1, denominator}};
    TimeSteppers::History<double> volume{4}, unsplit{4};
    TimeSteppers::BoundaryHistory<double, double, double> boundary;
    for (int i = 0; i < 3; ++i) {
      const TimeStepId id{true, 0, Time{slab, {i, denominator}}};
      const double t = id.step_time().value();
      const double y = std::pow(t, 5) / 5.;
      volume.insert(id, y, 0.);
      unsplit.insert(id, y, std::pow(t, 4));
      boundary.local().insert(id, 4, std::pow(t, 4));
      boundary.remote().insert(id, 4, 0.);
    }
    const auto coupling = [](double local, double /*remote*/) { return local; };
    const auto start_id = volume.back().time_step_id;
    const auto predictor_id = stepper.next_time_id(start_id, dt);
    const auto end_id = stepper.next_time_id(predictor_id, dt);
    double y = 0.;
    stepper.update_u(make_not_null(&y), volume, dt);
    stepper.add_boundary_delta(&y, boundary, dt, coupling);
    typename Update::predictor_tag::type predictor{};
    typename Update::errors_type errors{};
    Update::apply(make_not_null(&predictor), make_not_null(&errors), stepper,
                  LtsMode::Conservative, start_id, predictor_id, true, volume,
                  tolerances, y);
    REQUIRE(predictor.has_value());
    CHECK(predictor->second == y);
    const double endpoint = std::pow(predictor_id.substep_time(), 4);
    volume.insert(predictor_id, y, 0.);
    unsplit.insert(predictor_id, y, endpoint);
    boundary.local().insert(predictor_id, 4, endpoint);
    boundary.remote().insert(predictor_id, 4, 0.);
    errors[1] = stepper.update_u(make_not_null(&y), volume, dt, tolerances);
    REQUIRE(errors[1].has_value());
    CHECK(errors[1]->step_error() == 0.);
    stepper.add_boundary_delta(&y, boundary, dt, coupling);
    double reference_y = 0.;
    const auto reference =
        stepper.update_u(make_not_null(&reference_y), unsplit, dt, tolerances);
    REQUIRE(reference.has_value());
    CHECK(y == reference_y);
    REQUIRE(reference->step_error() > 0.);
    Update::apply(make_not_null(&predictor), make_not_null(&errors), stepper,
                  LtsMode::Conservative, predictor_id, end_id, true, volume,
                  tolerances, y);
    CHECK(errors[1]->step_error() == reference->step_error());
    // Keep a larger preexisting volume bound and reuse allocated storage.
    errors[1]->errors[errors[1]->order] = 1.;
    Update::apply(make_not_null(&predictor), make_not_null(&errors), stepper,
                  LtsMode::Conservative, predictor_id, end_id, true, volume,
                  tolerances, y);
    CHECK(errors[1]->step_error() == 1.);
    // Global stepping must retain its existing error and avoid the buffer.
    Update::apply(make_not_null(&predictor), make_not_null(&errors), stepper,
                  LtsMode::Off, predictor_id, end_id, true, volume, tolerances,
                  y);
    CHECK_FALSE(predictor.has_value());
    CHECK(errors[1]->step_error() == 1.);
  }
}

void test_filtered_predictor() {
  const Slab slab{0., 1.};
  const TimeDelta dt{slab, {1, 16}};
  const TimeSteppers::AdamsMoultonPc<true> stepper{4};
  const StepperErrorTolerances tolerances{
      StepperErrorTolerances::Estimates::StepperOrder, 1., 0.};
  TimeSteppers::History<double> history{4};
  for (int i = 0; i < 3; ++i) {
    history.insert(TimeStepId{true, 0, Time{slab, {i, 16}}}, 0., 1.);
  }
  const auto start_id = history.back().time_step_id;
  const auto predictor_id = stepper.next_time_id(start_id, dt);
  const auto end_id = stepper.next_time_id(predictor_id, dt);
  double y = 0.;
  stepper.update_u(make_not_null(&y), history, dt);
  typename Update::predictor_tag::type predictor{};
  typename Update::errors_type errors{};
  Update::apply(make_not_null(&predictor), make_not_null(&errors), stepper,
                LtsMode::Conservative, start_id, predictor_id, true, history,
                tolerances, y);
  y = 0.;  // Project the predictor's discarded mode before its RHS evaluation.
  history.insert(predictor_id, y, 1. - y);
  errors[1] = stepper.update_u(make_not_null(&y), history, dt, tolerances);
  REQUIRE(errors[1].has_value());
  CHECK(errors[1]->step_error() == 0.);
  Update::apply(make_not_null(&predictor), make_not_null(&errors), stepper,
                LtsMode::Conservative, predictor_id, end_id, true, history,
                tolerances, y);
  CHECK(errors[1]->step_error() < 1.e-15);
  CHECK(y > 0.);
  const TimeSteppers::AdamsMoultonPc<true> variable_stepper{std::nullopt};
  const double old_error = errors[1]->step_error();
  Update::apply(make_not_null(&predictor), make_not_null(&errors),
                variable_stepper, LtsMode::Conservative, predictor_id, end_id,
                true, history, tolerances, y);
  CHECK_FALSE(predictor.has_value());
  CHECK(errors[1]->step_error() == old_error);
  Update::apply(make_not_null(&predictor), make_not_null(&errors), stepper,
                LtsMode::Conservative, start_id, predictor_id, false, history,
                tolerances, y);
  CHECK_FALSE(predictor.has_value());
  const TimeStepId self_start{true, -1, start_id.step_time()};
  Update::apply(make_not_null(&predictor), make_not_null(&errors), stepper,
                LtsMode::Conservative, self_start, predictor_id, true, history,
                tolerances, y);
  CHECK_FALSE(predictor.has_value());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.UpdateLtsStepperError", "[Unit][Time]") {
  test_boundary_error();
  test_filtered_predictor();
}
