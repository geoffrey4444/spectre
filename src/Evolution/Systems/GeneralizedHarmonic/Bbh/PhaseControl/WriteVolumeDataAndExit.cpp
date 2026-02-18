// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Bbh/PhaseControl/WriteVolumeDataAndExit.hpp"

namespace gh::bbh::phase_control {
void WriteVolumeDataAndExit::pup(PUP::er& p) { PhaseChange::pup(p); }

PUP::able::PUP_ID WriteVolumeDataAndExit::my_PUP_ID = 0;  // NOLINT
}  // namespace gh::bbh::phase_control
