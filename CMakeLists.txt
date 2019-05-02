cmake_minimum_required(VERSION 3.5)
project(amd_rpp VERSION 0.1)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are not allowed. Invoke cmake from build directory")
endif ()

add_library(${PROJECT_NAME} SHARED )

add_subdirectory(${CMAKE_SOURCE_DIR}/src/imgaug)
target_link_libraries(${PROJECT_NAME} imgaug )



target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)
