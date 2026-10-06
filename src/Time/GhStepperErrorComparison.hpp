// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once
#include <cstddef>
class DataVector;
class StepperErrorTolerances;

// Campaign-only diagnostics. No history, class layout, or error policy changes.
namespace gh_stepper_error_comparison {
class VolumeScope {
 public:
  VolumeScope();
  ~VolumeScope();
  VolumeScope(const VolumeScope&) = delete;
  VolumeScope& operator=(const VolumeScope&) = delete;
 private:
  bool active_;
  bool previous_ = false;
};
void record_volume(const DataVector& values, const DataVector& errors,
                   const StepperErrorTolerances& tolerances, double time,
                   double step, size_t order, double reported_error,
                   size_t winner);
void report_complete(const DataVector& values, const DataVector& errors,
                     const StepperErrorTolerances& tolerances,
                     double reported_error);
}  // namespace gh_stepper_error_comparison
