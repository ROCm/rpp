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
project(test_suite/HOST)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rpp/test/cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

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
find_package(NIFTI QUIET)
find_library(libsnd_LIBS
    NAMES sndfile libsndfile
    PATHS ${CMAKE_SYSTEM_PREFIX_PATH} ${LIBSND_ROOT_DIR} "/usr/local"
    PATH_SUFFIXES lib lib64)

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
    message("-- ${Red}Error: TurboJpeg must be installed to install ${PROJECT_NAME} successfully!${ColourReset}")
endif()

if (OpenCV_FOUND)
    message("-- ${Green}${PROJECT_NAME} set to build with rpp and OpenCV${ColourReset}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rpp /usr/local/include)
    link_directories(${ROCM_PATH}/lib /usr/local/lib)

    add_executable(Tensor_host Tensor_host.cpp)
    add_executable(Tensor_misc_host Tensor_misc_host.cpp)
    add_executable(BatchPD_host_pkd3 ${ROCM_PATH}/share/rpp/test/rpp-performancetests/HOST_NEW/BatchPD_host_pkd3.cpp)
    add_executable(BatchPD_host_pln1 ${ROCM_PATH}/share/rpp/test/rpp-performancetests/HOST_NEW/BatchPD_host_pln1.cpp)
    add_executable(BatchPD_host_pln3 ${ROCM_PATH}/share/rpp/test/rpp-performancetests/HOST_NEW/BatchPD_host_pln3.cpp)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
    target_link_libraries(Tensor_host ${OpenCV_LIBS} -lturbojpeg -lrpp pthread ${LINK_LIBRARY_LIST})
    target_link_libraries(BatchPD_host_pkd3 ${OpenCV_LIBS} -lturbojpeg -lrpp pthread ${LINK_LIBRARY_LIST})
    target_link_libraries(BatchPD_host_pln1 ${OpenCV_LIBS} -lturbojpeg -lrpp pthread ${LINK_LIBRARY_LIST})
    target_link_libraries(BatchPD_host_pln3 ${OpenCV_LIBS} -lturbojpeg -lrpp pthread ${LINK_LIBRARY_LIST})
    target_link_libraries(Tensor_misc_host ${OpenCV_LIBS} -lturbojpeg -lrpp pthread ${LINK_LIBRARY_LIST})
else()
    message("-- ${Red}Error: OpenCV must be installed to install ${PROJECT_NAME} successfully!${ColourReset}")
endif()

if(NIFTI_FOUND AND OpenCV_FOUND)
    message("-- ${Green}${PROJECT_NAME} set to build with rpp and libniftiio and OpenCV ${ColourReset}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rpp /usr/local/include)
    link_directories(${ROCM_PATH}/lib /usr/local/lib)

    add_executable(Tensor_voxel_host Tensor_voxel_host.cpp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
    target_link_libraries(Tensor_voxel_host ${OpenCV_LIBS} -lturbojpeg -lrpp pthread ${LINK_LIBRARY_LIST}  ${NIFTI_PACKAGE_PREFIX}NIFTI::${NIFTI_PACKAGE_PREFIX}niftiio)
else()
    message("-- ${Yellow}Warning: libniftiio must be installed to install ${PROJECT_NAME}/Tensor_voxel_host successfully!${ColourReset}")
endif()

if(NOT libsnd_LIBS)
    message("-- ${Yellow}Warning: libsndfile must be installed to install ${PROJECT_NAME}/Tensor_audio_host successfully!${ColourReset}")
else()
    message("-- ${Green}${PROJECT_NAME} set to build with rpp and libsndfile ${ColourReset}")
    include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rpp /usr/local/include)
    link_directories(${ROCM_PATH}/lib /usr/local/lib)
    include_directories(${SndFile_INCLUDE_DIRS})
    link_directories(${SndFile_LIBRARIES_DIR} /usr/local/lib/)

    add_executable(Tensor_host_audio Tensor_host_audio.cpp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
    if(NOT APPLE)
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} stdc++fs)
    endif()
    target_link_libraries(Tensor_host_audio ${libsnd_LIBS} -lsndfile -lrpp pthread ${LINK_LIBRARY_LIST})
endif()
