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

cmake_minimum_required (VERSION 3.5)

project(rpp-test)

# make test with CTest
enable_testing()
include(CTest)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- ${PROJECT_NAME} INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# find RPP
find_library(RPP_LIBRARY NAMES rpp HINTS ${ROCM_PATH}/lib)
find_path(RPP_INCLUDE_DIR NAMES rpp.h PATHS ${ROCM_PATH}/include/rpp)

if(RPP_LIBRARY AND RPP_INCLUDE_DIR)
    set(RPP_FOUND TRUE)
    message("-- ${White}${PROJECT_NAME}: Using RPP -- \n\tLibraries:${RPP_LIBRARY} \n\tIncludes:${RPP_INCLUDE_DIR}${ColourReset}")
endif()

if(NOT RPP_FOUND)
    message("-- ${Yellow}${PROJECT_NAME} requires RPP. Install RPP before running CTests")
else()
    set(RPP_BACKEND_HIP_FOUND 0)
    if(EXISTS ${RPP_INCLUDE_DIR}/rpp_backend.h)
        file(READ ${RPP_INCLUDE_DIR}/rpp_backend.h RPP_BACKEND_FILE)
        string(REGEX MATCH "RPP_BACKEND_HIP ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    endif()
    if(NOT DEFINED BACKEND AND RPP_BACKEND_HIP_FOUND)
        set(BACKEND "HIP")
    elseif(NOT DEFINED BACKEND)
        set(BACKEND "CPU")
    endif()
endif(NOT RPP_FOUND)

# find required libraries
set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 QUIET)
find_package(NIFTI QUIET)
find_library(libsnd_LIBS
     NAMES sndfile libsndfile
     PATHS ${CMAKE_SYSTEM_PREFIX_PATH} ${LIBSND_ROOT_DIR} "/usr/local"
     PATH_SUFFIXES lib lib64)

if(Python3_FOUND)
    # Running all HOST tests
    add_test(
        NAME rpp_qa_tests_tensor_host_all
        COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HOST/runTests.py --qa_mode 1 --batch_size 3
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    if(NIFTI_FOUND)
        add_test(
            NAME rpp_qa_tests_tensor_voxel_host_all
            COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HOST/runVoxelTests.py --qa_mode 1 --batch_size 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif(NIFTI_FOUND)
    if(libsnd_LIBS)
        add_test(
            NAME rpp_qa_tests_tensor_audio_host_all
            COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HOST/runAudioTests.py --qa_mode 1 --batch_size 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif(libsnd_LIBS)

    if( "${BACKEND}" STREQUAL "HIP")
        # Running all HIP tests
        add_test(
            NAME rpp_qa_tests_tensor_hip_all
            COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HIP/runTests.py --qa_mode 1 --batch_size 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        if(NIFTI_FOUND)
            add_test(
                NAME rpp_qa_tests_tensor_voxel_hip_all
                COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HIP/runVoxelTests.py --qa_mode 1 --batch_size 3
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
        endif(NIFTI_FOUND)

    elseif( "${BACKEND}" STREQUAL "OCL")
        # TBD: Add OCL Tests
        message("-- ${Yellow}${PROJECT_NAME} Warning: OpenCL tests not enabled${ColourReset}")

    endif()
else()
  message("-- ${Yellow}${PROJECT_NAME} Warning: Python3 must be installed to run RPP test_suite successfully${ColourReset}")
endif(Python3_FOUND)