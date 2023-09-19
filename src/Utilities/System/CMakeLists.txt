# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SystemUtilities)

add_spectre_library(${LIBRARY} ParallelInfo.hpp)

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Abort.cpp
  ParallelInfo.cpp
  Prefetch.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Abort.hpp
  Exit.hpp
  ParallelInfo.hpp
  Prefetch.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Charmxx::charmxx
  )

set_target_properties(${LIBRARY} PROPERTIES
  FRAMEWORK TRUE
  FRAMEWORK_VERSION A
  MACOSX_FRAMEWORK_IDENTIFIER org.spectre-code.${LIBRARY}
  # MACOSX_FRAMEWORK_INFO_PLIST Info.plist
  # "current version" in semantic format in Mach-O binary file
  VERSION 1.0.0
  # "compatibility version" in semantic format in Mach-O binary file
  SOVERSION 1.0.0
  PUBLIC_HEADER ParallelInfo.hpp
  XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "SXS Collaboration"
)
