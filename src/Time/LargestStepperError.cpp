// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/LargestStepperError.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/GhStepperErrorComparison.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <typename T>
double largest_stepper_error(const T& values, const T& errors,
                             const StepperErrorTolerances& tolerances) {
  using std::abs;
  // Outer call to max() may be from blaze or an identity function
  // from ContainerHelpers.hpp for doubles.
  using ::max;
  // Inner max() is either blaze or std::max.
  using std::max;
  const double result = max(abs(errors) /
      (tolerances.absolute +
       tolerances.relative * max(abs(values), abs(values + errors))));
  if constexpr (std::is_same_v<T, DataVector>) {
    gh_stepper_error_comparison::report_complete(values, errors, tolerances,
                                               result);
  }
  return result;
}

namespace gh_stepper_error_comparison {
namespace {
bool enabled() {
  static const bool result = []() {
    const char* value = std::getenv("SPECTRE_LTS_ERROR_COMPARISON");
    return value != nullptr and std::strcmp(value, "1") == 0;
  }();
  return result;
}
thread_local bool evaluating_volume = false;
struct VolumeRecord {
  size_t size, order, winner;
  double time, step, error, defect, absolute, relative;
};
std::mutex records_mutex;
std::unordered_map<const double*, VolumeRecord> records;
}  // namespace

VolumeScope::VolumeScope() : active_(enabled()) {
  if (active_) {
    previous_ = evaluating_volume;
    evaluating_volume = true;
  }
}
VolumeScope::~VolumeScope() {
  if (active_) {
    evaluating_volume = previous_;
  }
}
void record_volume(const DataVector& values, const DataVector& errors,
                   const StepperErrorTolerances& tolerances, const double time,
                   const double step, const size_t order,
                   const double reported_error, const size_t winner) {
  if (not enabled()) {
    return;
  }
  const std::lock_guard<std::mutex> lock(records_mutex);
  records.insert_or_assign(values.data(), VolumeRecord{
      values.size(), order, winner, time, step, reported_error, errors[winner],
      tolerances.absolute, tolerances.relative});
}
void report_complete(const DataVector& values, const DataVector& errors,
                     const StepperErrorTolerances& tolerances,
                     const double reported_error) {
  if (not enabled() or evaluating_volume) {
    return;
  }
  VolumeRecord volume{};
  {
    const std::lock_guard<std::mutex> lock(records_mutex);
    const auto found = records.find(values.data());
    if (found == records.end()) {
      return;
    }
    volume = found->second;
    records.erase(found);
  }
  if (values.size() != volume.size or errors.size() != volume.size or
      volume.size == 0 or volume.size % 50 != 0 or
      tolerances.absolute != volume.absolute or
      tolerances.relative != volume.relative) {
    std::fprintf(stdout,
        "[AMPC_LTS_PAIR_INVALID] t=%.17g reason=layout_or_tolerance_mismatch\n",
        volume.time);
    std::fflush(stdout);
    return;
  }
  size_t winner = 0;
  double maximum = -1.;
  for (size_t i = 0; i < errors.size(); ++i) {
    const double denominator = tolerances.absolute + tolerances.relative *
        std::max(std::abs(values[i]), std::abs(values[i] + errors[i]));
    const double normalized = std::abs(errors[i]) / denominator;
    if (normalized > maximum) {
      maximum = normalized;
      winner = i;
    }
  }
  const size_t points = volume.size / 50;
  const double at_volume = errors[volume.winner];
  const double denominator = tolerances.absolute + tolerances.relative *
      std::max(std::abs(values[volume.winner]),
               std::abs(values[volume.winner] + at_volume));
  // Matching uses the same saved history allocation; volume-only calls are
  // excluded by VolumeScope. The full-error action calls this function with
  // that same history after boundary integration and before filtering.
  std::fprintf(stdout,
      "[AMPC_LTS_PAIR] t=%.17g h=%.17g q=%zu n=%zu volume_E=%.17g "
      "complete_E=%.17g checked_complete_E=%.17g volume_component=%zu "
      "volume_point=%zu complete_component=%zu complete_point=%zu "
      "volume_defect=%.17g complete_defect_at_volume_winner=%.17g "
      "difference_at_volume_winner=%.17g complete_E_at_volume_winner=%.17g\n",
      volume.time, volume.step, volume.order, points, volume.error,
      reported_error, maximum, volume.winner / points, volume.winner % points,
      winner / points, winner % points, volume.defect, at_volume,
      at_volume - volume.defect, std::abs(at_volume) / denominator);
  std::fflush(stdout);
}
}  // namespace gh_stepper_error_comparison

#define TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                \
  template double largest_stepper_error(                    \
      const TYPE(data) & values, const TYPE(data) & errors, \
      const StepperErrorTolerances& tolerances);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, std::complex<double>, DataVector,
                                      ComplexDataVector))

#undef INSTANTIATE
#undef TYPE
