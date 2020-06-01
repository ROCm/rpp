# Copyright (c) 2015 - 2020 Advanced Micro Devices, Inc. All rights reserved.
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
set(VERSION "0.4")

# ROCm Path
set(ROCM_PATH /opt/rocm CACHE PATH "ROCm installation path")
# need to do this to avoid setting the default path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH}/rpp CACHE PATH "RPP default ROCm installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# set message options
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(BoldBlue    "${Esc}[1;34m")
endif()

message("-- ${BoldBlue}Radeon Performance Primitives (RPP) Version -- ${VERSION}${ColourReset}")
message("-- ${BoldBlue}RPP install path set to -- ${CMAKE_INSTALL_PREFIX}${ColourReset}")
message("-- ${Yellow}cmake minimum version required: ${CMAKE_MINIMUM_REQUIRED_VERSION}${ColourReset}")
message("-- ${Green}cmake version found: ${CMAKE_VERSION}${ColourReset}")

include(CheckCXXCompilerFlag)
# set default backend to OPENCL
if (NOT DEFINED BACKEND)
    SET(BACKEND "OCL")
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

# Set Compilers
# set(COMPILER_FOR_OPENCL ${ROCM_PATH}/bin/hipcc)
# set(COMPILER_FOR_HIP ${ROCM_PATH}/bin/hipcc)
# set(COMPILER_FOR_HIPOC ${ROCM_PATH}/bin/hipcc)
set(COMPILER_FOR_HIP ${ROCM_PATH}/bin/hipcc)
set(COMPILER_FOR_OPENCL clang++)

include(ClangTidy)
enable_clang_tidy(
    CHECKS
        *
        -cert-env33-c
        -android-cloexec-fopen
        # Yea we shouldn't be using rand()
        -cert-msc30-c
        -cert-msc50-cpp
        -clang-analyzer-alpha.core.CastToStruct
        -clang-analyzer-optin.performance.Padding
        -clang-diagnostic-deprecated-declarations
        -clang-diagnostic-extern-c-compat
        -clang-diagnostic-unused-command-line-argument
        -cppcoreguidelines-pro-bounds-array-to-pointer-decay
        -cppcoreguidelines-pro-bounds-constant-array-index
        -cppcoreguidelines-pro-bounds-pointer-arithmetic
        -cppcoreguidelines-pro-type-member-init
        -cppcoreguidelines-pro-type-reinterpret-cast
        -cppcoreguidelines-pro-type-union-access
        -cppcoreguidelines-pro-type-vararg
        -cppcoreguidelines-special-member-functions
        -fuchsia-*
        -google-explicit-constructor
        -google-readability-braces-around-statements
        -google-readability-todo
        -google-runtime-int
        -google-runtime-references
        -hicpp-braces-around-statements
        -hicpp-explicit-conversions
        -hicpp-no-array-decay
        # We really shouldn't use bitwise operators with signed integers, but
        # opencl leaves us no choice
        -hicpp-signed-bitwise
        -hicpp-special-member-functions
        -hicpp-use-equals-default
        -hicpp-use-override
        -llvm-header-guard
        -llvm-include-order
        -misc-misplaced-const
        -modernize-pass-by-value
        -modernize-use-default-member-init
        -modernize-use-equals-default
        -modernize-use-transparent-functors
        -performance-unnecessary-value-param
        -readability-braces-around-statements
        -readability-else-after-return
        -readability-named-parameter
        -hicpp-use-auto
	    -modernize-use-auto
        -cppcoreguidelines-avoid-magic-numbers
        -readability-magic-numbers
        -cppcoreguidelines-macro-usage
        -misc-non-private-member-variables-in-classes
        -cppcoreguidelines-non-private-member-variables-in-classes
        -readability-isolate-declaration
        -cppcoreguidelines-avoid-c-arrays
        -hicpp-avoid-c-arrays
        -modernize-avoid-c-arrays
        -readability-uppercase-literal-suffix
        -hicpp-uppercase-literal-suffix
        -cert-msc32-c
        -cert-msc51-cpp
        -bugprone-exception-escape

        ${CLANG_TIDY_CHECKS}
    ERRORS
        *
        -readability-inconsistent-declaration-parameter-name
    HEADER_FILTER
        ".*hpp"
    EXTRA_ARGS
        -DMIOPEN_USE_CLANG_TIDY

)
include(CppCheck)
enable_cppcheck(
    CHECKS
        all
    SUPPRESS
        ConfigurationNotChecked
        unmatchedSuppression
        unusedFunction
        # constStatement
        noExplicitConstructor
        passedByValue
        # preprocessorErrorDirective
        unusedStructMember
        # Ignore initializer lists in the tests
        useInitializationList:*test/*.cpp
        *:*.cl
        *:*src/kernels/*.h
        knownConditionTrueFalse:*src/kernels/composable_kernel/*/*
        wrongPrintfScanfArgNum:*src/kernels/composable_kernel/*/*
        unusedScopedObject:*src/kernels/composable_kernel/*/*
    FORCE
    SOURCES
        addkernels/
        # driver/
        include/
        src/
        test/
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    DEFINE
        MIOPEN_USE_MIOPENGEMM=1
)

# Find required packages
find_package(OpenCL REQUIRED) # OpenCL_INCLUDE_DIRS, OpenCL_LIBRARIES

set( MODULES_LIST modules)

# look for and register extractkernel
find_program(EXTRACTKERNEL_BIN extractkernel
    PATH_SUFFIXES bin
    PATHS ${ROCM_PATH}
)
if(EXTRACTKERNEL_BIN)
    message(STATUS "${Green}extractkernel found: ${EXTRACTKERNEL_BIN}${ColourReset}")
    set(EXTRACTKERNEL_BIN "${EXTRACTKERNEL_BIN}")
else()
    message(FATAL_ERROR "${Red}extractkernel not found${ColourReset}")
endif()

# Online assembler
find_program(MIOPEN_AMDGCN_ASSEMBLER
    NAMES clang
    PATHS ${MIOPEN_AMDGCN_ASSEMBLER_PATH} ${ROCM_PATH}
    PATH_SUFFIXES /opencl/bin/x86_64
    NO_DEFAULT_PATH
)
message(STATUS "${Yellow}AMDGCN assembler: ${MIOPEN_AMDGCN_ASSEMBLER}${ColourReset}")

foreach (MOD_NAME ${MODULES_LIST})
    add_subdirectory(${CMAKE_SOURCE_DIR}/src/${MOD_NAME})
endforeach(MOD_NAME)

# Backend Specific Settings

if( "${BACKEND}" STREQUAL "HIP")
    find_package(HIP REQUIRED)
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HIPCC_FLAGS} -fopenmp -Ofast -msse4.2 -msse4.1 -mssse3 -mavx2")

    #set( LINK_LIST  ${ROCM_PATH}/lib/libhip_hcc.so  ${ROCM_PATH}/lib/libhip_hcc.so ${LINK_LIST})
   find_program(HIP_OC_COMPILER clang-ocl
        PATH_SUFFIXES bin
        PATHS ${ROCM_PATH}
    )
    if(HIP_OC_COMPILER)
        message(STATUS "${Green}hip compiler: ${HIP_OC_COMPILER}${ColourReset}")
        set(HIP_OC_COMPILER "${HIP_OC_COMPILER}")
    else()
        message(FATAL_ERROR "${Red}clang-ocl not found${ColourReset}")
    endif()

    set(RPP_BACKEND_HIP 1)
    find_program(HIP_OC_COMPILER clang-ocl
        PATH_SUFFIXES bin
        PATHS ${ROCM_PATH}
    )
    if(HIP_OC_COMPILER)
        message(STATUS "${Green}hip compiler: ${HIP_OC_COMPILER}${ColourReset}")
        set(HIP_OC_COMPILER "${HIP_OC_COMPILER}")
    else()
        message(FATAL_ERROR "${Red}clang-ocl not found${ColourReset}")
    endif()

    link_libraries(stdc++)

    # A hack to make this work without the device enumerator
    link_libraries(-amdgpu-target=gfx803 -amdgpu-target=gfx900 -Wno-unused-command-line-argument)
    CHECK_CXX_COMPILER_FLAG(-amdgpu-target=gfx906 -Werror, GFX_906_FLAG_VALID)
    if(GFX_906_FLAG_VALID)
        link_libraries(-amdgpu-target=gfx906)
    endif()
    # end hack

    set(HIP_COMPILER ${CMAKE_CXX_COMPILER} CACHE PATH "")

    link_libraries(stdc++)

    # A hack to make this work without the device enumerator
    link_libraries(-amdgpu-target=gfx803 -amdgpu-target=gfx900 -Wno-unused-command-line-argument)
    CHECK_CXX_COMPILER_FLAG(-amdgpu-target=gfx906 -Werror, GFX_906_FLAG_VALID)
    if(GFX_906_FLAG_VALID)
        link_libraries(-amdgpu-target=gfx906)
    endif()
    # end hack
    add_definitions( -DHIP_OC_COMPILER=${HIP_OC_COMPILER})

    set(HIP_COMPILER ${CMAKE_CXX_COMPILER} CACHE PATH "")

    if ("${COMPILE}" STREQUAL "HIPRTC")
	add_compile_definitions(HIPRTC)
    elseif ("${COMPILE}" STREQUAL "HSACOO")
	add_compile_definitions(HSACOO)
    endif()
elseif( "${BACKEND}" STREQUAL "OCL")
    set(RPP_BACKEND_OPENCL 1)
    set(CMAKE_CXX_COMPILER  ${COMPILER_FOR_OPENCL}) # GCC and G++ donst work for creating .so file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Ofast -msse4.2 -msse4.1 -mssse3 -mavx2")
    set( LINK_LIST ${OpenCL_LIBRARIES} ${LINK_LIST})
    set( INCLUDE_LIST ${OpenCL_INCLUDE_DIRS} ${INCLUDE_LIST})
    add_definitions(-DRPP_CACHE_DIR="${CACHE_DIR}")
endif()

if (NOT DEFINED SIMD_ENABLE)
    SET(SIMD_ENABLE "1")
endif()

add_library(${PROJECT_NAME} SHARED include/rpp.h $<TARGET_OBJECTS:modules>)
if( "${SIMD_ENABLE}" STREQUAL "1")
    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_SIMD_INTRINSICS=1)
    message("-- ${Green}${PROJECT_NAME} built with ENABLE_SIMD_INTRINSICS${ColourReset}")
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_SIMD_INTRINSICS=0)
    message("-- ${Red}${PROJECT_NAME} not built with ENABLE_SIMD_INTRINSICS${ColourReset}")
endif()

# target setting
option(Boost_USE_STATIC_LIBS "Use boost static libraries" ON)
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
install (TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install (DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX})
