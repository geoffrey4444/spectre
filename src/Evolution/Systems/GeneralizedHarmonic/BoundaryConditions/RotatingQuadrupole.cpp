// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/RotatingQuadrupole.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace gh::BoundaryConditions {
RotatingQuadrupole::RotatingQuadrupole(
    const double amplitude, const double angular_frequency, const double phase,
    const double start_time, const double ramp_up_duration,
    const std::optional<double> stop_time, const double ramp_down_duration,
    const Options::Context& context)
    : amplitude_(amplitude),
      angular_frequency_(angular_frequency),
      phase_(phase),
      start_time_(start_time),
      ramp_up_duration_(ramp_up_duration),
      stop_time_(stop_time),
      ramp_down_duration_(ramp_down_duration) {
  if (not std::isfinite(amplitude_) or amplitude_ < 0. or
      not std::isfinite(angular_frequency_) or angular_frequency_ == 0. or
      not std::isfinite(phase_) or not std::isfinite(start_time_) or
      not std::isfinite(ramp_up_duration_) or ramp_up_duration_ <= 0. or
      not std::isfinite(ramp_down_duration_) or ramp_down_duration_ <= 0. or
      not std::isfinite(start_time_ + ramp_up_duration_)) {
    PARSE_ERROR(context,
                "RotatingQuadrupole requires finite parameters, nonnegative "
                "Amplitude, nonzero AngularFrequency, and positive ramp "
                "durations with a finite ramp-up end time.");
  }
  if (stop_time_.has_value() and
      (not std::isfinite(*stop_time_) or
       *stop_time_ < start_time_ + ramp_up_duration_ or
       not std::isfinite(*stop_time_ + ramp_down_duration_))) {
    PARSE_ERROR(context,
                "RotatingQuadrupole StopTime must be finite and at least "
                "StartTime + RampUpDuration, and the ramp-down end time "
                "must be finite.");
  }
}

double RotatingQuadrupole::envelope(const double time) const {
  if (time <= start_time_ or
      (stop_time_.has_value() and time >= *stop_time_ + ramp_down_duration_)) {
    return 0.;
  }
  const auto smoothstep = [](const double x) {
    return x * x * x * (10. + x * (-15. + 6. * x));
  };
  if (time < start_time_ + ramp_up_duration_) {
    return smoothstep((time - start_time_) / ramp_up_duration_);
  }
  if (stop_time_.has_value() and time > *stop_time_) {
    // Evaluate S(1-x), equivalent to 1-S(x), to avoid cancellation near zero.
    return smoothstep((*stop_time_ + ramp_down_duration_ - time) /
                      ramp_down_duration_);
  }
  return 1.;
}

tnsr::aa<double, 3, Frame::Inertial> RotatingQuadrupole::operator()(
    const double time) const {
  tnsr::aa<double, 3, Frame::Inertial> result{0.};
  const double amplitude = amplitude_ * envelope(time);
  if (amplitude == 0.) {
    return result;
  }
  const double angle = angular_frequency_ * (time - start_time_) + phase_;
  get<1, 1>(result) = amplitude * cos(angle);
  get<2, 2>(result) = -get<1, 1>(result);
  get<1, 2>(result) = amplitude * sin(angle);
  return result;
}

void RotatingQuadrupole::pup(PUP::er& p) {
  p | amplitude_;
  p | angular_frequency_;
  p | phase_;
  p | start_time_;
  p | ramp_up_duration_;
  p | stop_time_;
  p | ramp_down_duration_;
}
}  // namespace gh::BoundaryConditions
