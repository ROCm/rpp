cmake_minimum_required(VERSION 3.5)
project(amd-rpp VERSION 0.1 LANGUAGES CXX)

SET(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
link_directories(opt/rocm/hip/lib)


add_library(brightness_contrast SHARED src/imgaug/hip/hip_brightness_contract.cpp)

target_include_directories(brightness_contrast
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)
