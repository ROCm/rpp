cmake_minimum_required(VERSION 3.5)
project(imgaug-opencl)

file( GLOB MOD_CPP "*.cpp" )
add_library( ${PROJECT_NAME} STATIC ${MOD_CPP})

set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc) # GCC and G++ donst work for creating .so file
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -lOpenCL -lm")

add_definitions(-DMOD_CL_PATH="\\"${CMAKE_CURRENT_LIST_DIR}/kernel/"\\")

link_directories(/opt/rocm/opencl/lib/x86_64/)
target_link_libraries (${PROJECT_NAME} OpenCL )
add_definitions(-DOCL_COMPILE)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories( ${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        /opt/rocm/opencl/include/
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)