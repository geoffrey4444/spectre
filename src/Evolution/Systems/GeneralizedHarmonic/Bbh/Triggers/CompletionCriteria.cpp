// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Bbh/Triggers/CompletionCriteria.hpp"

#include <pup.h>

namespace gh::bbh::Triggers {
void CompletionCriteria::pup(PUP::er& p) { Trigger::pup(p); }

PUP::able::PUP_ID CompletionCriteria::my_PUP_ID = 0;  // NOLINT
}  // namespace gh::bbh::Triggers
