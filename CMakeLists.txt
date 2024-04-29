#[[
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]

cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)

# RPP Version
set(VERSION "1.7.0")

# Set Project Version and Language
project(rpp VERSION ${VERSION} LANGUAGES CXX)

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

find_program(MAKE_NSIS_EXE makensis)
find_program(RPMBUILD_EXE rpmbuild)
find_program(DPKG_EXE dpkg)

# ROCm Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Avoid setting the default path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "RPP default ROCm installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# RPP Default Options
set(DEFAULT_BUILD_TYPE "Release")
option(BUILD_WITH_AMD_ADVANCE "Build RPP for advanced AMD GPU Architecture"    ON)

# Set message options
if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(Blue        "${Esc}[34m")
    set(BoldBlue    "${Esc}[1;34m")
    set(Magenta     "${Esc}[35m")
    set(Cyan        "${Esc}[36m")
    set(White       "${Esc}[37m")
endif()

# Set backend options
if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
  set(BACKEND "CPU")
  message("-- ${Magenta}Apple macOS Detected -- GPU Support turned OFF${ColourReset}")
endif()

if(NOT DEFINED BACKEND)
    set(BACKEND "HIP")   # set default backend to HIP
else()
    if("${BACKEND}" STREQUAL "OPENCL" OR "${BACKEND}" STREQUAL "OCL" OR "${BACKEND}" STREQUAL "OpenCL")
        set(BACKEND "OCL")
    elseif("${BACKEND}" STREQUAL "HIP" OR "${BACKEND}" STREQUAL "hip")
        set(BACKEND "HIP")
    elseif("${BACKEND}" STREQUAL "host" OR "${BACKEND}" STREQUAL "HOST" OR "${BACKEND}" STREQUAL "CPU" OR "${BACKEND}" STREQUAL "cpu")
        set(BACKEND "CPU")
    else()
        message("-- ${Yellow}Warning: ${PROJECT_NAME} backend option: ${BACKEND} -- unknown${ColourReset}")
        message("-- ${Yellow}Warning: ${PROJECT_NAME} default backend will be enforced${ColourReset}")
        set(BACKEND "HIP")
    endif()
endif()

message("-- ${BoldBlue}AMD ROCm Performance Primitives (RPP) Version -- ${VERSION}${ColourReset}")
message("-- ${BoldBlue}AMD RPP install path set to -- ${CMAKE_INSTALL_PREFIX}${ColourReset}")
message("-- ${BoldBlue}AMD RPP Backend set to -- ${BACKEND}${ColourReset}")

# RPP Build Type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "RPP Default Build Type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
if(CMAKE_BUILD_TYPE MATCHES Debug)
    # -O0  -- Don't Optimize output file
    # -g -- generate debugging information
    # -Og -- Optimize for debugging experience rather than speed or size
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Og")
else()
    # -O3 -- Optimize output file
    # TBD:Verify Requirement -Ofast - Optimize for speed disregarding exact standards compliance
    # -DNDEBUG -- turn off asserts
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Ofast -DNDEBUG")
endif()
message("-- ${BoldBlue}AMD RPP Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")

if(WIN32 AND CMAKE_CROSSCOMPILING)
    set(WINE_CMD "wine")
else()
    set(WINE_CMD)
endif()

message("-- ${Cyan}RPP Developer Options${ColourReset}")
message("-- ${Cyan}     -D BUILD_WITH_AMD_ADVANCE=${BUILD_WITH_AMD_ADVANCE} [Turn ON/OFF Build for AMD advanced GPUs(default:ON)]${ColourReset}")
message("-- ${Yellow}     NOTE: For ROCm Version less than 6.0.0+ Use -D BUILD_WITH_AMD_ADVANCE=OFF${ColourReset}")
message("-- ${Cyan}     -D BACKEND=${BACKEND} [Select RPP Backend [options:CPU/OPENCL/HIP](default:HIP)]${ColourReset}")
message("-- ${Cyan}     -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} [Select RPP build type [options:Debug/Release](default:Release)]${ColourReset}")

# CMake checks and setups
if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "${Red}${PROJECT_NAME} -- In-source builds are not allowed. Invoke cmake from build directory${ColourReset}")
endif()
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Half
find_package(HALF REQUIRED)
include_directories(${HALF_INCLUDE_DIRS})

# OpenMP
find_package(OpenMP REQUIRED)
if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    endif()
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # STD Filesystem
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} stdc++fs)
endif()
include_directories(${OpenMP_INCLUDE_DIRS})
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} OpenMP::OpenMP_CXX)

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} Threads::Threads)

# set global compiler flags for the project
# -fPIC -- Generate position-independent code if possible.
# -mavx2 -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and AVX2 built-in functions and code generation.
# -mfma -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and FMA built-in functions and code generation.
# -std=gnu++17 -- Conform to the ISO 2017 C++ standard with GNU extensions.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -mavx2 -mf16c -mfma -std=gnu++17")

# Check AVX/SSE support
include(CheckCXXCompilerFlag)

# Check AVX flag
check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
if(COMPILER_SUPPORTS_AVX)
    add_definitions(-D__AVX2__)
else()
    # Check SSE flag
    check_cxx_compiler_flag("-msse4" COMPILER_SUPPORTS_SSE)
    if(COMPILER_SUPPORTS_SSE)
        add_definitions(-D__SSE4_1__)
    endif()
endif()

# Backend specific settings
if("${BACKEND}" STREQUAL "HIP")
    # TBD: to be removed once STATIC becomes the only option within HIP backend, and all STATIC compile time switches are removed.
    add_definitions(-DSTATIC)

    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()

    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    set(DEFAULT_AMDGPU_TARGETS "gfx803;gfx900;gfx906;gfx908;gfx90a;gfx940;gfx1030;gfx1100")
    if (BUILD_WITH_AMD_ADVANCE)
        set(DEFAULT_AMDGPU_TARGETS ${DEFAULT_AMDGPU_TARGETS} "gfx941;gfx942")
    endif()
    set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message("-- ${White}${PROJECT_NAME} -- Using HIP - Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER}${ColourReset})
        # TBD: Set external flag to notify GPU_SUPPORT and HIP backend availability - Compatible with rocAL and MIVisionX
        add_definitions(-DGPU_SUPPORT)
        # For cmakedefine01 in rpp_backend.h.in
        set(RPP_BACKEND_HIP 1)
        # To use RPP_BACKEND_HIP
        add_definitions(-DRPP_BACKEND_HIP=${RPP_BACKEND_HIP})
        set(COMPILER_FOR_HIP ${ROCM_PATH}/llvm/bin/clang++)
        include_directories(${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR})
        link_directories(${HIP_PATH}/${CMAKE_INSTALL_LIBDIR})
        # link To HIP Host library -- [hip::host] to use host API
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)

        # Add modules
        set(MODULES_LIST modules)
        foreach(MOD_NAME ${MODULES_LIST})
            add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/${MOD_NAME})
        endforeach(MOD_NAME)

        message("-- ${Green}${PROJECT_NAME} set to build with HIP backend${ColourReset}")
    else()
        set(BACKEND "CPU")
        message("-- ${Yellow}HIP not found! ${PROJECT_NAME} set to build with CPU backend${ColourReset}")
    endif()
elseif("${BACKEND}" STREQUAL "OCL")
    # Find package OpenCL
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message("-- ${White}${PROJECT_NAME} -- Using OpenCL - LIB:" ${OpenCL_LIBRARIES} "\tInclude:" ${OpenCL_INCLUDE_DIRS}${ColourReset})
        # Set external flag to notify GPU_SUPPORT and OPENCL backend availability - Compatible with rocAL and MIVisionX
        add_definitions(-DGPU_SUPPORT)
        set(RPP_BACKEND_OPENCL 1) # For cmakedefine01 in rpp_backend.h.in
        add_definitions(-DRPP_BACKEND_OPENCL=${RPP_BACKEND_OPENCL}) # To use RPP_BACKEND_HIP
        include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)

        # Add modules
        set(MODULES_LIST modules)
        foreach(MOD_NAME ${MODULES_LIST})
            add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/${MOD_NAME})
        endforeach(MOD_NAME)

        # Set compiler and flags
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OpenCL_LIBRARIES})
        add_definitions(-DRPP_CACHE_DIR="${CACHE_DIR}")

        message("-- ${Green}${PROJECT_NAME} set to build with GPU_SUPPORT using OpenCL backend${ColourReset}")
    else()
        set(BACKEND "CPU")
        message("-- ${Yellow}OpenCL not found! ${PROJECT_NAME} falling back to build for CPU backend${ColourReset}")
    endif()

endif()
# CPU backend specific settings
if("${BACKEND}" STREQUAL "CPU")
    # Add modules
    set(MODULES_LIST modules)
    foreach(MOD_NAME ${MODULES_LIST})
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/${MOD_NAME})
    endforeach(MOD_NAME)
endif()

configure_file("${PROJECT_SOURCE_DIR}/include/rpp_backend.h.in" "${PROJECT_BINARY_DIR}/include/rpp_backend.h")

# Enable SIMD for HOST code (in both OpenCL and HIP backends)
if(NOT DEFINED SIMD_ENABLE)
    SET(SIMD_ENABLE "1")
endif()

add_library(${PROJECT_NAME} SHARED include/rpp.h $<TARGET_OBJECTS:modules>)
if("${SIMD_ENABLE}" STREQUAL "1")
    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_SIMD_INTRINSICS=1)
    message("-- ${Green}${PROJECT_NAME} set to build with ENABLE_SIMD_INTRINSICS${ColourReset}")
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_SIMD_INTRINSICS=0)
    message("-- ${Red}${PROJECT_NAME} set to build without ENABLE_SIMD_INTRINSICS${ColourReset}")
endif()

message("-- ${White}${PROJECT_NAME} -- Using Compiler - Path:" ${CMAKE_CXX_COMPILER} "\tVersion:" ${CMAKE_CXX_COMPILER_VERSION} "\tCompiler:" ${CMAKE_CXX_COMPILER_ID}${ColourReset})
message("-- ${White}${PROJECT_NAME} -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
message("-- ${White}${PROJECT_NAME} -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")
target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${PROJECT_NAME} PROPERTIES
                      VERSION ${PROJECT_VERSION}
                      SOVERSION ${PROJECT_VERSION_MAJOR})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ROCM_PATH}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/common
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/func_specific
        ${INCLUDE_LIST}
)

if( "${BACKEND}" STREQUAL "HIP")
    target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/hip
    )
elseif( "${BACKEND}" STREQUAL "OCL")
    target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/cl
    )
endif()

add_subdirectory(addkernels)

# Installation
# install RPP libs -- {ROCM_PATH}/lib
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)

# install RPP include files -- {ROCM_PATH}/include/rpp/
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rpp
       COMPONENT dev
       PATTERN "*.h.in" EXCLUDE)
install(FILES ${PROJECT_BINARY_DIR}/include/rpp_backend.h
       DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rpp
       COMPONENT dev)

# install Test
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test COMPONENT test)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/utilities/test_suite/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test COMPONENT test)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/utilities/rpp-performancetests DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test COMPONENT test)

# set license information
set(CPACK_RESOURCE_FILE_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT runtime)
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan COMPONENT asan)
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR}-dev COMPONENT dev)
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR}-test COMPONENT test)

# CTest - Needs RPP Installed
enable_testing()
include(CTest)
add_subdirectory(utilities/test_suite)

# Set package information
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_NAME          "rpp")
set(CPACK_PACKAGE_LICENSE       "MIT")
set(CPACK_PACKAGE_CONTACT       "mivisionx support <mivisionx.support@amd.com>")
set(CPACK_PACKAGE_VENDOR        "AMD ROCm")
set(CPACK_PACKAGE_GROUP         "Development/Tools")
set(CPACK_PACKAGE_HOMEPAGE      "https://github.com/ROCm/rpp")

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
endif()

# Set the dependent packages
set(RPP_DEBIAN_PACKAGE_LIST  "rocm-hip-runtime, half")
set(RPP_RPM_PACKAGE_LIST     "rocm-hip-runtime, half")
set(RPP_DEBIAN_DEV_PACKAGE_LIST  "rocm-hip-runtime-dev")
set(RPP_RPM_DEV_PACKAGE_LIST     "rocm-hip-runtime-devel")

# package release
set(CPACK_DEBIAN_FILE_NAME       "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME          "RPM-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
set(CPACK_RPM_PACKAGE_RELEASE    "local")
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
endif()

# '%{?dist}' breaks manual builds on debian systems due to empty Provides
execute_process(COMMAND rpm --eval %{?dist}
                RESULT_VARIABLE PROC_RESULT
                OUTPUT_VARIABLE EVAL_RESULT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
    string (APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()

set(CPACK_PACKAGING_INSTALL_PREFIX  ${CMAKE_INSTALL_PREFIX})
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE   ${CPACK_PACKAGE_HOMEPAGE})
set(CPACK_RPM_PACKAGE_URL           ${CPACK_PACKAGE_HOMEPAGE})
set(CPACK_RPM_PACKAGE_AUTOREQPROV   "no")

# Debian package
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "rocm-core, ${RPP_DEBIAN_PACKAGE_LIST}")
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${PROJECT_NAME}-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "rocm-core, ${PROJECT_NAME}, ${RPP_DEBIAN_DEV_PACKAGE_LIST}")
set(CPACK_DEBIAN_TEST_PACKAGE_NAME "${PROJECT_NAME}-test")
set(CPACK_DEBIAN_TEST_PACKAGE_DEPENDS "rocm-core, ${PROJECT_NAME}-dev")
set(CPACK_DEBIAN_PACKAGE_LICENSE "MIT" )
# Debian package specific variable for ASAN
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "rocm-core-asan" )
# RPM package
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES "rocm-core, ${RPP_RPM_PACKAGE_LIST}")
set(CPACK_RPM_RUNTIME_PACKAGE_PROVIDES "${PROJECT_NAME}")
set(CPACK_RPM_RUNTIME_PACKAGE_OBSOLETES "${PROJECT_NAME}")
set(CPACK_RPM_DEV_PACKAGE_NAME "${PROJECT_NAME}-devel")
set(CPACK_RPM_DEV_PACKAGE_REQUIRES "rocm-core, ${PROJECT_NAME}, ${RPP_RPM_DEV_PACKAGE_LIST}")
set(CPACK_RPM_DEV_PACKAGE_PROVIDES "${PROJECT_NAME}-devel")
set(CPACK_RPM_DEV_PACKAGE_OBSOLETES "${PROJECT_NAME}-devel")
set(CPACK_RPM_TEST_PACKAGE_NAME "${PROJECT_NAME}-test")
set(CPACK_RPM_TEST_PACKAGE_REQUIRES "rocm-core, ${PROJECT_NAME}-devel")
set(CPACK_RPM_TEST_PACKAGE_PROVIDES "${PROJECT_NAME}-test")
set(CPACK_RPM_TEST_PACKAGE_OBSOLETES "${PROJECT_NAME}-test")
set(CPACK_RPM_PACKAGE_LICENSE "MIT" )
# RPM package specific variable for ASAN
set(CPACK_RPM_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan" )

if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_RUNTIME_PACKAGE_REQUIRES ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core-asan," "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_TEST_PACKAGE_REQUIRES ${CPACK_RPM_TEST_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core-asan," "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_TEST_PACKAGE_DEPENDS ${CPACK_DEBIAN_TEST_PACKAGE_DEPENDS})
endif()

if(ENABLE_ASAN_PACKAGING)
  # ASAN Package requires asan component with only libraries and license file
  set(CPACK_COMPONENTS_ALL asan)
else()
  set(CPACK_COMPONENTS_ALL runtime dev test)
endif()

if(APPLE)
    set(CPACK_GENERATOR "Bundle")
    set(CPACK_BUNDLE_NAME ${CPACK_PACKAGE_NAME})
    set(CPACK_BUNDLE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/.github/rpp.png")
    set(CPACK_BUNDLE_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/.github/info.plist")
else()
    set(CPACK_GENERATOR "TGZ;ZIP")
    if(EXISTS ${MAKE_NSIS_EXE})
        list(APPEND CPACK_GENERATOR "NSIS")
    endif()
    if(EXISTS ${RPMBUILD_EXE})
        list(APPEND CPACK_GENERATOR "RPM")
        message("-- ${White}AMD ROCm RPP RunTime Package -- ${CPACK_RPM_DEV_PACKAGE_REQUIRES}${ColourReset}")
        message("-- ${White}AMD ROCm RPP Dev Package -- ${CPACK_RPM_DEV_PACKAGE_DEPENDS}${ColourReset}")
        message("-- ${White}AMD ROCm RPP Test Package -- ${CPACK_RPM_TEST_PACKAGE_DEPENDS}${ColourReset}")
    endif()
    if(EXISTS ${DPKG_EXE})
        list(APPEND CPACK_GENERATOR "DEB")
        message("-- ${White}AMD ROCm RPP RunTime Package -- ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS}${ColourReset}")
        message("-- ${White}AMD ROCm RPP Dev Package -- ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS}${ColourReset}")
        message("-- ${White}AMD ROCm RPP Test Package -- ${CPACK_DEBIAN_TEST_PACKAGE_DEPENDS}${ColourReset}")
    endif()
endif()

include(CPack)

cpack_add_component(runtime
   DISPLAY_NAME "RPP Runtime Package"
   DESCRIPTION "ROCm Performance Primitives library is a comprehensive high performance \
computer vision library for AMD CPUs and GPUs with HOST/HIP/OpenCL back-ends.
RPP runtime package provides rpp library and license.txt")

cpack_add_component(dev
    DISPLAY_NAME "RPP Develop Package"
    DESCRIPTION "ROCm Performance Primitives library is a comprehensive high performance \
computer vision library for AMD CPUs and GPUs with HOST/HIP/OpenCL back-ends. \
RPP develop package provides rpp library, header files, and license.txt")

cpack_add_component(test
    DISPLAY_NAME "RPP Test Package"
    DESCRIPTION "ROCm Performance Primitives library is a comprehensive high performance \
computer vision library for AMD CPUs and GPUs with HOST/HIP/OpenCL back-ends. \
RPP test package provides rpp test suite")

cpack_add_component(asan
    DISPLAY_NAME "RPP ASAN Package"
    DESCRIPTION "ROCm Performance Primitives library is a comprehensive high performance \
computer vision library for AMD CPUs and GPUs with HOST/HIP/OpenCL back-ends. \
RPP ASAN package provides rpp ASAN libraries")
