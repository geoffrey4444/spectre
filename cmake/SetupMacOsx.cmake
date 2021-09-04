# Distributed under the MIT License.
# See LICENSE.txt for details.

if(APPLE)
  set(SPECTRE_MACOSX_MIN "10.9")
  if(DEFINED MACOSX_MIN)
    set(SPECTRE_MACOSX_MIN "${MACOSX_MIN}")
  endif()
  set(MACOS_SYS_LIB_PATH "${MACOS_SYS_LIB_ROOT}/usr/lib")
  message("macOS syslibpath: ${MACOS_SYS_LIB_PATH}")
  set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -mmacosx-version-min=${SPECTRE_MACOSX_MIN} \
     -Wl,-L${MACOS_SYS_LIB_PATH},-lSystem")
  message(STATUS "Minimum macOS version: ${SPECTRE_MACOSX_MIN}")
  # set(CMAKE_STATIC_LINKER_FLAGS
  #   "${CMAKE_STATIC_LINKER_FLAGS} -L${MACOS_SYS_LIB_PATH} -lSystem")
endif()
