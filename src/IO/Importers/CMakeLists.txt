# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Importers)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ElementDataReader.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Domain
  DomainStructure
  IO
  Spectral
  )

add_subdirectory(Actions)
