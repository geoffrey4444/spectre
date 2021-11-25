// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

#include "Utilities/ErrorHandling/Error.hpp"

#include <csignal>

#ifdef __APPLE__
// #include <xmmintrin.h>
#else
#include <cfenv>
#endif

namespace {

#ifdef __APPLE__
// auto old_mask = _mm_getcsr();
#endif

[[noreturn]] void fpe_signal_handler(int /*signal*/) {
  ERROR("Floating point exception!");
}
}  // namespace

void enable_floating_point_exceptions() {
#ifdef __APPLE__
  // _mm_setcsr(_MM_MASK_MASK &
  //            ~(_MM_MASK_OVERFLOW | _MM_MASK_INVALID | _MM_MASK_DIV_ZERO));
#else
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  signal(SIGFPE, fpe_signal_handler);
#endif
}

void disable_floating_point_exceptions() {
#ifdef __APPLE__
  //  _mm_setcsr(old_mask);
#else
  fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif
}
