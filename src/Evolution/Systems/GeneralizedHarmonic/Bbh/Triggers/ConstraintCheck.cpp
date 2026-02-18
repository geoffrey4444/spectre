// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Bbh/Triggers/ConstraintCheck.hpp"

#include <pup.h>

namespace gh::bbh::Triggers {
void ConstraintCheck::pup(PUP::er& p) { DenseTrigger::pup(p); }

PUP::able::PUP_ID ConstraintCheck::my_PUP_ID = 0;  // NOLINT
}  // namespace gh::bbh::Triggers
