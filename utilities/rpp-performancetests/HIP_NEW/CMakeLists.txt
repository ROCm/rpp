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

cmake_minimum_required(VERSION 3.10)
project(rpp_performancetests/HIP_NEW)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake)
set(ROCM_PATH /opt/rocm CACHE PATH "default ROCm installation path")
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

find_package(HIP QUIET)
find_package(OpenCV QUIET)

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

if(HIP_FOUND AND OpenCV_FOUND)
    message("-- ${Green}${PROJECT_NAME} set to build with rpp, hip and OpenCV${ColourReset}")
    set(COMPILER_FOR_HIP ${ROCM_PATH}/bin/amdclang++)
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP})

    include_directories(${hip_INCLUDE_DIRS} ${hip_INCLUDE_DIRS}/hip)
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rpp /usr/local/include)
    link_directories(${ROCM_PATH}/lib /usr/local/lib)

    add_executable(BatchPD_hip_pkd3 BatchPD_hip_pkd3.cpp)
    add_executable(BatchPD_hip_pln1 BatchPD_hip_pln1.cpp)
    add_executable(BatchPD_hip_pln3 BatchPD_hip_pln3.cpp)
    add_executable(Tensor_hip_pkd3 Tensor_hip_pkd3.cpp)
    add_executable(Tensor_hip_pln1 Tensor_hip_pln1.cpp)
    add_executable(Tensor_hip_pln3 Tensor_hip_pln3.cpp)
    # add_executable(Single_hip Single_hip.cpp)
    add_executable(uniqueFunctionalities_hip uniqueFunctionalities_hip.cpp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_SUPPORT=1 -DRPP_BACKEND_HIP=1 -std=gnu++14")
    target_link_libraries(BatchPD_hip_pkd3 ${OpenCV_LIBS} -lrpp ${hip_LIBRARIES} pthread ${LINK_LIBRARY_LIST} hip::device)
    target_link_libraries(BatchPD_hip_pln1 ${OpenCV_LIBS} -lrpp ${hip_LIBRARIES} pthread ${LINK_LIBRARY_LIST} hip::device)
    target_link_libraries(BatchPD_hip_pln3 ${OpenCV_LIBS} -lrpp ${hip_LIBRARIES} pthread ${LINK_LIBRARY_LIST} hip::device)
    target_link_libraries(Tensor_hip_pkd3 ${OpenCV_LIBS} -lrpp ${hip_LIBRARIES} pthread ${LINK_LIBRARY_LIST} hip::device)
    target_link_libraries(Tensor_hip_pln1 ${OpenCV_LIBS} -lrpp ${hip_LIBRARIES} pthread ${LINK_LIBRARY_LIST} hip::device)
    target_link_libraries(Tensor_hip_pln3 ${OpenCV_LIBS} -lrpp ${hip_LIBRARIES} pthread ${LINK_LIBRARY_LIST} hip::device)
    # target_link_libraries(Single_hip ${OpenCV_LIBS} -lrpp ${hip_LIBRARIES} pthread ${LINK_LIBRARY_LIST hip::device)
    target_link_libraries(uniqueFunctionalities_hip ${OpenCV_LIBS} -lrpp ${hip_LIBRARIES} pthread ${LINK_LIBRARY_LIST} hip::device)
else()
    message("-- ${Yellow}Error: OpenCV and hip must be installed to install ${PROJECT_NAME} successfully!${ColourReset}")
endif()