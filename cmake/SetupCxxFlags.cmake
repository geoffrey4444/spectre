# Distributed under the MIT License.
# See LICENSE.txt for details.

option(DEBUG_SYMBOLS "Add -g to CMAKE_CXX_FLAGS if ON, -g0 if OFF." ON)

option(OVERRIDE_ARCH "The architecture to use. Default is native." OFF)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSPECTRE_DEBUG")

if(NOT ${DEBUG_SYMBOLS})
  string(REPLACE "-g " "-g0 " CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
endif()

# Always build with -g so we can view backtraces, etc. when production code
# fails. This can be overridden by passing `-D DEBUG_SYMBOLS=OFF` to CMake
if(${DEBUG_SYMBOLS})
  set_property(TARGET SpectreFlags
    APPEND PROPERTY INTERFACE_COMPILE_OPTIONS -g)
endif(${DEBUG_SYMBOLS})

# Always compile only for the current architecture. This can be overridden
# by passing `-D OVERRIDE_ARCH=THE_ARCHITECTURE` to CMake
if("${OVERRIDE_ARCH}" STREQUAL "apple_silicon")
  set_property(TARGET SpectreFlags
      APPEND PROPERTY
      INTERFACE_COMPILE_OPTIONS
      $<$<COMPILE_LANGUAGE:CXX>:>)
elseif(NOT "${OVERRIDE_ARCH}" STREQUAL "OFF")
  set_property(TARGET SpectreFlags
      APPEND PROPERTY
      INTERFACE_COMPILE_OPTIONS
      # The -mno-avx512f flag is necessary to avoid a Blaze 3.8 bug. The flag
      # should be re-enabled when we can insist on Blaze 3.9 which will include
      # a fix that allows this vectorization flag again.
      $<$<COMPILE_LANGUAGE:CXX>:-march=${OVERRIDE_ARCH} -mno-avx512f>)
else()
  set_property(TARGET SpectreFlags
      APPEND PROPERTY
      INTERFACE_COMPILE_OPTIONS
      $<$<COMPILE_LANGUAGE:CXX>:-march=native -mno-avx512f>)
endif()

# We always want a detailed backtrace of template errors to make debugging them
# easier
set_property(TARGET SpectreFlags
  APPEND PROPERTY
  INTERFACE_COMPILE_OPTIONS
  $<$<COMPILE_LANGUAGE:CXX>:-ftemplate-backtrace-limit=0>)

# By default, the LLVM optimizer assumes floating point exceptions are ignored.
create_cxx_flag_target("-ffp-exception-behavior=maytrap" SpectreFpExceptions)
target_link_libraries(
  SpectreFlags
  INTERFACE
  SpectreFpExceptions
  )
