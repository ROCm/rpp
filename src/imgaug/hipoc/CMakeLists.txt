cmake_minimum_required(VERSION 3.5)
project(imgaug-hipoc)


file( GLOB MOD_CPP "*.cpp" )
add_library( ${PROJECT_NAME} STATIC ${MOD_CPP})
set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIPOC}) # GCC and G++ donst work for creating .so file
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -lm ")

add_definitions(-DHIP_COMPILE)

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories( ${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCL_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
    )
