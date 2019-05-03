cmake_minimum_required(VERSION 3.5)
project(imgaug) #Rename the Project name

file( GLOB MOD_CPP "*.cpp" )
add_library( ${PROJECT_NAME} STATIC ${MOD_CPP})

if( "${BACKEND}" STREQUAL "HIP")
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
    add_definitions(-DHIP_COMPILE)
elseif( "${BACKEND}" STREQUAL "OCL")
    message (SEND_ERROR "Module Not implemented")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories( ${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)
