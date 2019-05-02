cmake_minimum_required(VERSION 3.5)
project(amd_rpp VERSION 0.1)


if (BACKEND_OCL)
    set(CMAKE_CXX_COMPILER /usr/bin/gcc)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -lOpenCL -lm")
    add_library(brightness_contrast SHARED src/imgaug/cl/cl_brightness_contrast.c
                                           src/imgaug/cpu/host_brightness_contrast.cpp
                                           src/imgaug/rppi_brightness_contrast.cpp)
    link_directories(/opt/rocm/opencl/lib/)
    add_definitions(-DOCL_COMPILE)
elseif(BACKEND_HIP)
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
    link_directories(opt/rocm/hip/lib)
    add_library(brightness_contrast SHARED src/imgaug/hip/hip_brightness_contrast.cpp
                                           src/imgaug/cpu/host_brightness_contrast.cpp
                                           src/imgaug/rppi_brightness_contrast.cpp)
    add_definitions(-DHIP_COMPILE)
endif()

if(BACKEND_CPU_ALONE)
    set(CMAKE_CXX_COMPILER /usr/bin/gcc)
    add_library(brightness_contrast SHARED src/imgaug/cpu/host_brightness_contrast.cpp
                                           src/imgaug/rppi_brightness_contrast.cpp)
endif()


target_include_directories(brightness_contrast
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)
