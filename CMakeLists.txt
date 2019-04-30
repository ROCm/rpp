cmake_minimum_required(VERSION 3.5)
project(amd_rpp VERSION 0.1 LANGUAGES CXX)


add_subdirectory(${CMAKE_SOURCE_DIR}/src/imgaug)

add_library( rpp SHARED)

target_include_directories(brightness_contrast
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)
