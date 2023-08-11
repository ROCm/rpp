# Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.5)
project(test_suite/HOST)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake)
set(ROCM_PATH /opt/rocm CACHE PATH "default ROCm installation path")

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

find_package(OpenCV QUIET)
find_package(TurboJpeg QUIET)
# BOOST
set(BOOST_COMPONENTS filesystem system)
find_package(Boost COMPONENTS ${BOOST_COMPONENTS} REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})
# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -ggdb -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} OpenMP::OpenMP_CXX)

if(TurboJpeg_FOUND)
    message("-- ${Green}${PROJECT_NAME} set to build with rpp and TurboJpeg${ColourReset}")
    include_directories(${TurboJpeg_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${TurboJpeg_LIBRARIES_DIR})
else()
    message("-- ${Yellow}Error: TurboJpeg must be installed to install ${PROJECT_NAME} successfully!${ColourReset}")
endif()

if (OpenCV_FOUND)
    message("-- ${Green}${PROJECT_NAME} set to build with rpp and OpenCV${ColourReset}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rpp /usr/local/include)
    link_directories(${ROCM_PATH}/lib /usr/local/lib)

    add_executable(Tensor_host Tensor_host.cpp)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")
    target_link_libraries(Tensor_host ${OpenCV_LIBS} -lturbojpeg -lrpp pthread ${LINK_LIBRARY_LIST})

else()
    message("-- ${Yellow}Error: OpenCV must be installed to install ${PROJECT_NAME} successfully!${ColourReset}")
endif()