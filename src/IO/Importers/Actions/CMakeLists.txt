# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Importers)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ReadSpecThirdOrderPiecewisePolynomial.hpp
  ReadVolumeData.hpp
  ReceiveVolumeData.hpp
  RegisterWithElementDataReader.hpp
  )
