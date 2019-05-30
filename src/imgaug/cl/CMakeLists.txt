cmake_minimum_required(VERSION 3.5)
project(imgaug-opencl)

file( GLOB MOD_CPP "*.cpp" )
add_library( ${PROJECT_NAME} STATIC ${MOD_CPP})

set(CMAKE_CXX_COMPILER ${COMPILER_FOR_OPENCL}) # GCC and G++ donst work for creating .so file
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -lm")

add_definitions(-DMOD_CL_PATH="\\"${CMAKE_CURRENT_LIST_DIR}/kernel/"\\")
add_definitions(-DOCL_COMPILE)


set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries (${PROJECT_NAME} ${OpenCL_LIBRARIES} )
target_include_directories( ${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCL_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)
