# Distributed under the MIT License.
# See LICENSE.txt for details.

option(DEBUG_SYMBOLS "Add -g to CMAKE_CXX_FLAGS if ON, -g0 if OFF." ON)

option(OVERRIDE_ARCH "The architecture to use. Default is native." OFF)

if(APPLE)
  if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    # Because of a bug in macOS on Apple Silicon, executables larger than
    # 2GB in size cannot run. The -Oz flag minimizes executable size, to
    # avoid this bug.
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSPECTRE_DEBUG -Oz")
  else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSPECTRE_DEBUG")
  endif()
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSPECTRE_DEBUG")
endif()

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
if(NOT "${OVERRIDE_ARCH}" STREQUAL "OFF")
  set_property(TARGET SpectreFlags
      APPEND PROPERTY
      INTERFACE_COMPILE_OPTIONS
      # The -mno-avx512f flag is necessary to avoid a Blaze 3.8 bug. The flag
      # should be re-enabled when we can insist on Blaze 3.9 which will include
      # a fix that allows this vectorization flag again.
      $<$<COMPILE_LANGUAGE:CXX>:-march=${OVERRIDE_ARCH} -mno-avx512f>)
else()
  # Apple Silicon Macs do not support the -march flag or the -mno-avx512f flag
  if(APPLE)
    if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64")
      set_property(TARGET SpectreFlags
          APPEND PROPERTY
          INTERFACE_COMPILE_OPTIONS
          $<$<COMPILE_LANGUAGE:CXX>:>)
    else()
      set_property(TARGET SpectreFlags
        APPEND PROPERTY
        INTERFACE_COMPILE_OPTIONS
        $<$<COMPILE_LANGUAGE:CXX>:-march=native -mno-avx512f>)
    endif()
  else()
    set_property(TARGET SpectreFlags
        APPEND PROPERTY
        INTERFACE_COMPILE_OPTIONS
        $<$<COMPILE_LANGUAGE:CXX>:-march=native -mno-avx512f>)
  endif()
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
