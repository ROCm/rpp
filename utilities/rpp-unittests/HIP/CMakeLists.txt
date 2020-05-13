cmake_minimum_required (VERSION 2.8)
project (Rpp_test)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenCV REQUIRED )

include_directories (${OpenCV_INCLUDE_DIRS})
include_directories (/opt/rocm/opencl/include/)
include_directories (/opt/rocm/include)
include_directories (/opt/rocm/rpp/include)
link_directories    (/opt/rocm/lib)
link_directories    (/opt/rocm/rpp/lib/)

add_executable(BatchPD_hip BatchPD_hip.cpp)
add_executable(Single_hip Single_hip.cpp)
set(COMPILER_FOR_HIP /opt/rocm/bin/hipcc)

set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHIP_COMPILE=1 -DRPP_BACKEND_HIP=1 -fopenmp -std=c++14") 

target_link_libraries(BatchPD_hip ${OpenCV_LIBS} -I/opt/rocm/rpp/include/ -I/opt/rocm/opencl/include/ -I/opt/rocm/include/ -L/opt/rocm/rpp/lib/ -lamd_rpp -lboost_filesystem -lboost_system -L/opt/rocm/hip/lib/ -lhiprtc)
target_link_libraries(Single_hip ${OpenCV_LIBS} -I/opt/rocm/rpp/include/ -I/opt/rocm/opencl/include/ -I/opt/rocm/include/ -L/opt/rocm/rpp/lib/ -lamd_rpp -lboost_filesystem -lboost_system -L/opt/rocm/hip/lib/ -lhiprtc)
