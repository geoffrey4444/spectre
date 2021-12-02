// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

#include "Utilities/ErrorHandling/Error.hpp"

#include <csignal>

#ifdef __APPLE__
#ifdef __arm64__
#include <fenv.h>
#else
#include <xmmintrin.h>
#endif
#else
#include <cfenv>
#endif

namespace {

#ifdef __APPLE__
#ifdef __arm64__
auto old_fpcr = 0;
#else
auto old_mask = _mm_getcsr();
#endif
#endif

[[noreturn]] void fpe_signal_handler(int /*signal*/) {
  ERROR("Floating point exception!");
}
}  // namespace

void enable_floating_point_exceptions() {
#ifdef __APPLE__
#ifdef __arm64__
  uint64_t fpcr = __builtin_arm_rsr64("FPCR") | __fpcr_trap_overflow |
                  __fpcr_trap_invalid | __fpcr_trap_divbyzero;
  __builtin_arm_wsr64("FPCR", fpcr);
  std::signal(SIGILL, fpe_signal_handler);
#else
  _mm_setcsr(_MM_MASK_MASK &
             ~(_MM_MASK_OVERFLOW | _MM_MASK_INVALID | _MM_MASK_DIV_ZERO));
#endif
#else
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif
  std::signal(SIGFPE, fpe_signal_handler);
}

void disable_floating_point_exceptions() {
#ifdef __APPLE__
#ifdef __arm64__
  __builtin_arm_wsr64("FPCR", old_fpcr);
  std::signal(SIGILL, SIG_DFL);
#else
  _mm_setcsr(old_mask);
#endif
#else
  fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif
}
