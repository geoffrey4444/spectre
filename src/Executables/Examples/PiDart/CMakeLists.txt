# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Informer
  Options
  Parallel
  Utilities
)

add_spectre_parallel_executable(
  PiDart
  PiDart
  Executables/Examples/PiDart
  Metavariables
  "${LIBS_TO_LINK}"
)
