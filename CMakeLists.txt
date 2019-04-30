cmake_minimum_required(VERSION 3.5)
project(amd-rpp VERSION 0.1)


if (BACKEND_OCL)
    SET(CMAKE_CXX_COMPILER /usr/bin/gcc)
    link_directories(/opt/rocm/opencl/lib/)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -lOpenCL -lm")

    add_library(brightness_contrast SHARED src/imgaug/cl/cl_brightness_contrast.c)

elseif(BACKEND_HIP)

    SET(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
    link_directories(opt/rocm/hip/lib)

    add_library(brightness_contrast SHARED src/imgaug/hip/hip_brightness_contract.cpp)

endif()


target_include_directories(brightness_contrast
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)
