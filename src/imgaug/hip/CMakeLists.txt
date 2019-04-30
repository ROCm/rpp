cmake_minimum_required(VERSION 3.5)
project("${PROJECT_NAME}_hip" LANGUAGES CXX)

set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)

file( GLOB HIP_CPP "*.cpp" "*.hpp" )

add_library( ${PROJECT_NAME} STATIC ${HIP_CPP} )
target_include_directories( ${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)
