# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Lapack)

set(LIBRARY_SOURCES
    GeneralizedEigenvalue.cpp
   )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE ErrorHandling
  )
