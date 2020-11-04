// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunctions.hpp"

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"

#include <pup.h>

namespace GeneralizedHarmonic::ConstraintDamping {
template <size_t VolumeDim, typename Fr>
DampingFunctions<VolumeDim, Fr>::DampingFunctions(
    std::unique_ptr<DampingFunction<VolumeDim, Fr>*> damping_function_gamma0,
    std::unique_ptr<DampingFunction<VolumeDim, Fr>*> damping_function_gamma1,
    std::unique_ptr<DampingFunction<VolumeDim, Fr>*>
        damping_function_gamma2) noexcept {
  gamma0 = std::move(damping_function_gamma0);
  gamma1 = std::move(damping_function_gamma1);
  gamma2 = std::move(damping_function_gamma2);
}

// clang-tidy: google-runtime-references
template <size_t VolumeDim, typename Fr>
void DampingFunctions<VolumeDim, Fr>::pup(PUP::er& p) noexcept {  // NOLINT
  p | gamma0;
  p | gamma1;
  p | gamma2;
}
}  // namespace GeneralizedHarmonic::ConstraintDamping
