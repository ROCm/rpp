cmake_minimum_required(VERSION 3.5)
project(amd_rpp VERSION 0.1)

# Cmake checks and setups
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Invoke cmake from build directory")
endif ()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

#set(COMPILER_FOR_OPENCL /opt/rocm/bin/hipcc)
#set(COMPILER_FOR_HIP /opt/rocm/bin/hipcc)
#set(COMPILER_FOR_HIPOC /opt/rocm/bin/hipcc)
# Set Compilers
set(COMPILER_FOR_HIP /opt/AMD/aocc-compiler-2.0.0/bin/clang++)
set(COMPILER_FOR_HIPOC /opt/AMD/aocc-compiler-2.0.0/bin/clang++)
set(COMPILER_FOR_OPENCL /opt/AMD/aocc-compiler-2.0.0/bin/clang++)

add_library(${PROJECT_NAME} SHARED include/rpp.h)

# Find required packages
find_package(OpenCL REQUIRED) # OpenCL_INCLUDE_DIRS, OpenCL_LIBRARIES

#TODO: Future Work Build list can be set from CLI ; CACHED and to be used
set( MODULES_LIST "imgaug")

set( SUB_MODULES_LIST "")
foreach (MOD_NAME ${MODULES_LIST})
    add_subdirectory(${CMAKE_SOURCE_DIR}/src/${MOD_NAME})
endforeach(MOD_NAME)

# Backend Specific Settings

if( "${BACKEND}" STREQUAL "HIP")
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    #set( LINK_LIST  /opt/rocm/lib/libhip_hcc.so  /opt/rocm/lib/libhip_hcc.so ${LINK_LIST})

    link_libraries(-amdgpu-target=gfx803 -amdgpu-target=gfx900 -Wno-unused-command-line-argument)
elseif( "${BACKEND}" STREQUAL "OCL")
    set(CMAKE_CXX_COMPILER  ${COMPILER_FOR_OPENCL})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set( LINK_LIST ${OpenCL_LIBRARIES} ${LINK_LIST})
    set( INCLUDE_LIST ${OpenCL_INCLUDE_DIRS} ${INCLUDE_LIST})
endif()

# target setting
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(${PROJECT_NAME} ${MODULES_LIST} ${SUB_MODULES_LIST} ${LINK_LIST} OpenMP::OpenMP_CXX)
endif()
target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${INCLUDE_LIST}
        )

# Installation
install (TARGETS amd_rpp
        LIBRARY DESTINATION /opt/rocm/rpp/lib
        )
install (DIRECTORY include
        DESTINATION /opt/rocm/rpp/
        )
