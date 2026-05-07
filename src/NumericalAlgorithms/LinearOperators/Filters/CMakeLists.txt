# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Filters/Filter.hpp
  Filters/Hypercube.hpp
  Filters/Hypercube.tpp
  Filters/None.hpp
  Filters/None.tpp
  Filters/SphericalShell.hpp
  Filters/SphericalShell.tpp
  )
