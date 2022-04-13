# Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.5)
project(amd_rpp)
set(VERSION "0.95")

find_program(MAKE_NSIS_EXE makensis)
find_program(RPMBUILD_EXE rpmbuild)
find_program(DPKG_EXE dpkg)

# ROCm Path
set(ROCM_PATH /opt/rocm CACHE PATH "RPP default ROCm installation path")
# Avoid setting the default path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH}/rpp CACHE PATH "RPP default ROCm installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

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

message("-- ${BoldBlue}Radeon Performance Primitives (RPP) Version -- ${VERSION}${ColourReset}")
message("-- ${BoldBlue}RPP install path set to -- ${CMAKE_INSTALL_PREFIX}${ColourReset}")
message("-- ${Yellow}CMake minimum version required: ${CMAKE_MINIMUM_REQUIRED_VERSION}${ColourReset}")
message("-- ${Green}CMake version found: ${CMAKE_VERSION}${ColourReset}")

include(CheckCXXCompilerFlag)

# Set defaults
if (NOT DEFINED BACKEND)
    SET(BACKEND "HIP")
endif()

if(WIN32 AND CMAKE_CROSSCOMPILING)
    set(WINE_CMD "wine")
else()
    set(WINE_CMD)
endif()

# Cmake checks and setups
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "-- ${Red}In-source builds are not allowed. Invoke cmake from build directory${ColourReset}")
endif ()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Set Compilers for HIP and OPENCL backends
set(COMPILER_FOR_HIP ${ROCM_PATH}/bin/hipcc)
set(COMPILER_FOR_OPENCL clang++)

# Backend specific settings
if( "${BACKEND}" STREQUAL "HIP" OR "${BACKEND}" STREQUAL "hip")
    add_definitions(-DSTATIC) # This line is to be removed once STATIC becomes the only option within HIP backend, and all STATIC compile time switches are removed.

    # Find package
    find_package(HIP REQUIRED)

    # Add modules
    set(MODULES_LIST modules)
    foreach(MOD_NAME ${MODULES_LIST})
        add_subdirectory(${CMAKE_SOURCE_DIR}/src/${MOD_NAME})
    endforeach(MOD_NAME)

    # Set compiler and flags
    set(RPP_BACKEND_HIP 1)
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP})
    set(HIP_COMPILER ${CMAKE_CXX_COMPILER} CACHE PATH "")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HIPCC_FLAGS} -fopenmp -Ofast -msse4.2 -msse4.1 -mssse3 -mavx2 -mfma")
    link_libraries(stdc++)

    message("-- ${Green}${PROJECT_NAME} set to build with HIP backend${ColourReset}")
elseif("${BACKEND}" STREQUAL "OCL" OR "${BACKEND}" STREQUAL "OPENCL" OR "${BACKEND}" STREQUAL "OpenCL")
    # Find package
    find_package(OpenCL REQUIRED)

    # Add modules
    set(MODULES_LIST modules)
    foreach(MOD_NAME ${MODULES_LIST})
        add_subdirectory(${CMAKE_SOURCE_DIR}/src/${MOD_NAME})
    endforeach(MOD_NAME)

    # Set compiler and flags
    set(RPP_BACKEND_OPENCL 1)
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_OPENCL}) # GCC and G++ donst work for creating .so file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Ofast -msse4.2 -msse4.1 -mssse3 -mavx2 -mfma")
    set(LINK_LIST ${OpenCL_LIBRARIES} ${LINK_LIST})
    set(INCLUDE_LIST ${OpenCL_INCLUDE_DIRS} ${INCLUDE_LIST})
    add_definitions(-DRPP_CACHE_DIR="${CACHE_DIR}")

    message("-- ${Green}${PROJECT_NAME} set to build with OpenCL backend${ColourReset}")
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

# Target setting
option(Boost_USE_STATIC_LIBS "Use boost static libraries" OFF)
set(BOOST_COMPONENTS filesystem system)
add_definitions(-DBOOST_ALL_NO_LIB=1)
find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
find_package(OpenMP)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ROCM_PATH}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${INCLUDE_LIST}
)

add_subdirectory(addkernels)

# Installation
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX} PATTERN "*.h.in" EXCLUDE)
install(FILES ${PROJECT_BINARY_DIR}/include/rpp_backend.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)


# Set package information
set(CPACK_PACKAGE_VERSION       ${VERSION})
set(CPACK_PACKAGE_NAME          "rpp")
set(CPACK_PACKAGE_LICENSE       "MIT")
set(CPACK_PACKAGE_CONTACT       "mivisionx support <mivisionx.support@amd.com>")
set(CPACK_PACKAGE_VENDOR        "AMD Radeon")
set(CPACK_PACKAGE_ICON          "${CMAKE_SOURCE_DIR}/docs/images/MIVisionX.bmp")
set(CPACK_PACKAGE_GROUP         "Development/Tools")
set(CPACK_PACKAGE_HOMEPAGE      "https://gpuopen-professionalcompute-libraries.github.io/rpp/")

set(CPACK_DEBIAN_PACKAGE_DESCRIPTION   "Radeon Performance Primitives library is a comprehensive high performance computer vision library
    Radeon Performance Primitives (RPP) library is a comprehensive high performance \
    computer vision library for AMD (CPU and GPU) with HIP and OpenCL back-ends")
set(CPACK_RPM_PACKAGE_SUMMARY          "Radeon Performance Primitives (RPP) library is a comprehensive high performance computer vision library for AMD (CPU and GPU) with HIP and OpenCL back-ends")

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
endif()

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

set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

set(CPACK_DEBIAN_PACKAGE_HOMEPAGE   ${CPACK_PACKAGE_HOMEPAGE})
set(CPACK_RPM_PACKAGE_URL           ${CPACK_PACKAGE_HOMEPAGE})
set(CPACK_RPM_PACKAGE_AUTOREQPROV   "no")

if(APPLE)
    set(CPACK_GENERATOR "Bundle")
    set(CPACK_BUNDLE_NAME ${CPACK_PACKAGE_NAME})
    set(CPACK_BUNDLE_ICON ${CPACK_PACKAGE_ICON})
    set(CPACK_BUNDLE_PLIST "${CMAKE_SOURCE_DIR}/.github/info.plist")
else()
    set(CPACK_GENERATOR "TGZ;ZIP")
    if(EXISTS ${MAKE_NSIS_EXE})
        list(APPEND CPACK_GENERATOR "NSIS")
    endif()
    if(EXISTS ${RPMBUILD_EXE})
        list(APPEND CPACK_GENERATOR "RPM")
    endif()
    if(EXISTS ${DPKG_EXE})
        list(APPEND CPACK_GENERATOR "DEB")
    endif()
endif()

include(CPack)
