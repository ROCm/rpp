cmake_minimum_required(VERSION 3.5)
project(arithmetic) #Rename the Project name

file( GLOB MOD_CPP "*.cpp" )
add_library( ${PROJECT_NAME} STATIC ${MOD_CPP})

if( "${BACKEND}" STREQUAL "HIP")
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
    add_definitions(-DHIP_COMPILE)
elseif( "${BACKEND}" STREQUAL "OCL")
    add_subdirectory( cl )
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc) # GCC and G++ donst work for creating .so file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -lOpenCL -lm")
    link_directories(/opt/rocm/opencl/lib/x86_64/)
    target_link_libraries (${PROJECT_NAME} OpenCL  ${PROJECT_NAME}-opencl)
    target_include_directories(${PROJECT_NAME} PUBLIC /opt/rocm/opencl/include/)
    add_definitions(-DOCL_COMPILE)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories( ${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)
