################################################################################
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

cmake_minimum_required (VERSION 3.5)
project(test_suite)
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
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HOST/runTests.py --qa_mode 1 --batch_size 3
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    if(NIFTI_FOUND)
        add_test(
            NAME rpp_qa_tests_tensor_voxel_host_all
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HOST/runTests_voxel.py --qa_mode 1 --batch_size 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif(NIFTI_FOUND)
    if(libsnd_LIBS)
        add_test(
            NAME rpp_qa_tests_tensor_audio_host_all
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HOST/runAudioTests.py --qa_mode 1 --batch_size 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif(libsnd_LIBS)

    # Running case by case - one sample for brightness (in tensor_host) + one sample for slice (in tensor_voxel_host) + one sample for non_silent_region (in tensor_audio_host)
    add_test(
        NAME rpp_qa_tests_tensor_host_brightness
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HOST/runTests.py --qa_mode 1 --batch_size 3 --case_list 0
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    if(NIFTI_FOUND)
        add_test(
            NAME rpp_qa_tests_tensor_voxel_host_slice
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HOST/runTests_voxel.py --qa_mode 1 --batch_size 3 --case_list 1
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif(NIFTI_FOUND)
    if(libsnd_LIBS)
        add_test(
            NAME rpp_qa_tests_tensor_audio_host_non_silent_region
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HOST/runAudioTests.py --case_list 0 --qa_mode 1 --batch_size 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif(libsnd_LIBS)

    if( "${BACKEND}" STREQUAL "HIP")

        # Running all HIP tests
        add_test(
            NAME rpp_qa_tests_tensor_hip_all
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HIP/runTests.py --qa_mode 1 --batch_size 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        if(NIFTI_FOUND)
            add_test(
                NAME rpp_qa_tests_tensor_voxel_hip_all
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HIP/runTests_voxel.py --qa_mode 1 --batch_size 3
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
        endif(NIFTI_FOUND)

        # Running case by case - one sample for brightness (in tensor_hip) and one sample for slice (in tensor_voxel_hip)
        add_test(
            NAME rpp_qa_tests_tensor_hip_brightness
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HIP/runTests.py --qa_mode 1 --batch_size 3 --case_list 0
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        if(NIFTI_FOUND)
            add_test(
                NAME rpp_qa_tests_tensor_voxel_hip_slice
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HIP/runTests_voxel.py --qa_mode 1 --batch_size 3 --case_list 1
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
        endif(NIFTI_FOUND)
    elseif( "${BACKEND}" STREQUAL "OCL")
        # TBD: Add OCL Tests

    endif()

else()
  message("-- ${Yellow}Test Warning: Python3 must be installed to run RPP test_suite successfully!${ColourReset}")
endif(Python3_FOUND)
