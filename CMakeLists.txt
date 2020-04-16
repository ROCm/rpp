cmake_minimum_required(VERSION 3.15)
project(amd_rpp VERSION 0.2)


include(CheckCXXCompilerFlag)

if(WIN32 AND CMAKE_CROSSCOMPILING)
    set(WINE_CMD "wine")
else()
    set(WINE_CMD)
endif()

# Cmake checks and setups
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are not allowed. Invoke cmake from build directory")
endif ()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")


# Set Compilers
# set(COMPILER_FOR_OPENCL /opt/rocm/bin/hipcc)
# set(COMPILER_FOR_HIP /opt/rocm/bin/hipcc)
# set(COMPILER_FOR_HIPOC /opt/rocm/bin/hipcc)

set(COMPILER_FOR_HIP /opt/rocm/bin/hipcc)
# set(COMPILER_FOR_HIPOC /opt/rocm/bin/hipcc)
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


add_library(${PROJECT_NAME} SHARED include/rpp.h)

# Find required packages
find_package(OpenCL REQUIRED) # OpenCL_INCLUDE_DIRS, OpenCL_LIBRARIES


set( MODULES_LIST modules)


# look for and register extractkernel
find_program(EXTRACTKERNEL_BIN extractkernel
    PATH_SUFFIXES bin
    PATHS /opt/rocm
)
if(EXTRACTKERNEL_BIN)
    message(STATUS "extractkernel found: ${EXTRACTKERNEL_BIN}")
    set(EXTRACTKERNEL_BIN "${EXTRACTKERNEL_BIN}")
else()
    message(FATAL_ERROR "extractkernel not found")
endif()

# Online assembler
find_program(MIOPEN_AMDGCN_ASSEMBLER
    NAMES clang
    PATHS ${MIOPEN_AMDGCN_ASSEMBLER_PATH} /opt/rocm
    PATH_SUFFIXES /opencl/bin/x86_64
    NO_DEFAULT_PATH
)
message(STATUS "AMDGCN assembler: ${MIOPEN_AMDGCN_ASSEMBLER}")


foreach (MOD_NAME ${MODULES_LIST})
    add_subdirectory(${CMAKE_SOURCE_DIR}/src/${MOD_NAME})
endforeach(MOD_NAME)

# Backend Specific Settings

if( "${BACKEND}" STREQUAL "HIP")
    find_package(HIP REQUIRED)
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HIPCC_FLAGS} -fopenmp -Ofast -msse4.2 -msse4.1 -mssse3 -mavx2")

    #set( LINK_LIST  /opt/rocm/lib/libhip_hcc.so  /opt/rocm/lib/libhip_hcc.so ${LINK_LIST})
   find_program(HIP_OC_COMPILER clang-ocl
        PATH_SUFFIXES bin
        PATHS /opt/rocm
    )
    if(HIP_OC_COMPILER)
        message(STATUS "hip compiler: ${HIP_OC_COMPILER}")
        set(HIP_OC_COMPILER "${HIP_OC_COMPILER}")
    else()
        message(FATAL_ERROR "clang-ocl not found")
    endif()

    set(RPP_BACKEND_HIP 1)
    find_program(HIP_OC_COMPILER clang-ocl
        PATH_SUFFIXES bin
        PATHS /opt/rocm
    )
    if(HIP_OC_COMPILER)
        message(STATUS "hip compiler: ${HIP_OC_COMPILER}")
        set(HIP_OC_COMPILER "${HIP_OC_COMPILER}")
    else()
        message(FATAL_ERROR "clang-ocl not found")
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

if( "${SIMD_ENABLE}" STREQUAL "1")
    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_SIMD_INTRINSICS=1)
    message("-- ${Green}${PROJECT_NAME} built with ENABLE_SIMD_INTRINSICS")
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_SIMD_INTRINSICS=0)
    message("-- ${Green}${PROJECT_NAME} not built with ENABLE_SIMD_INTRINSICS")
endif()


# target setting
option(Boost_USE_STATIC_LIBS "Use boost static libraries" ON)
set(BOOST_COMPONENTS filesystem system)
add_definitions(-DBOOST_ALL_NO_LIB=1)
find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(${PROJECT_NAME} ${MODULES_LIST} ${SUB_MODULES_LIST} ${SUB_MODULES_LIST_HIP} ${LINK_LIST} ${LINK_LIST_HIP} OpenMP::OpenMP_CXX)
endif()
target_link_libraries(${PROJECT_NAME} ${MODULES_LIST} ${LINK_LIST} ${LINK_LIST_HIP})
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        /opt/rocm/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${INCLUDE_LIST}
)

add_subdirectory(addkernels)

# Installation
install (TARGETS amd_rpp
         LIBRARY DESTINATION /opt/rocm/rpp/lib
)
install (DIRECTORY include
         DESTINATION /opt/rocm/rpp/
)
