// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonicRollOn.hpp"

#include <cmath>
#include <utility>

#include "Options/ParseError.hpp"

namespace gh::gauges {
DampedHarmonicRollOn::DampedHarmonicRollOn(
    AnalyticChristoffel initial_gauge, const double start, const double window,
    const double width, const std::array<double, 3>& amplitudes,
    const std::array<int, 3>& exponents, const Options::Context& context)
    : initial_gauge_(std::move(initial_gauge)),
      parameters_(start, window, width, amplitudes, exponents) {
  if (not std::isfinite(start) or not std::isfinite(window) or window <= 0.0) {
    PARSE_ERROR(context,
                "RollOnStartTime must be finite and RollOnTimeWindow must "
                "be finite and strictly positive.");
  }
}

DampedHarmonicRollOn::DampedHarmonicRollOn(CkMigrateMessage* const msg)
    : GaugeCondition(msg) {}

std::unique_ptr<GaugeCondition> DampedHarmonicRollOn::get_clone() const {
  return std::make_unique<DampedHarmonicRollOn>(*this);
}

void DampedHarmonicRollOn::pup(PUP::er& p) {
  GaugeCondition::pup(p);
  p | initial_gauge_;
  p | parameters_;
}

// NOLINTNEXTLINE
PUP::able::PUP_ID DampedHarmonicRollOn::my_PUP_ID = 0;
}  // namespace gh::gauges
