cmake_minimum_required(VERSION 3.5)
project(modules-hip)

if( BACKEND_HIPOC)
    link_libraries(-amdgpu-target=gfx803 -amdgpu-target=gfx900 -Wno-unused-command-line-argument)
    file( GLOB MOD_CPP "*.cpp" )
if ("${COMPILE}" STREQUAL "HIPRTC")
	add_definitions(-DHIPRTC)
elseif ("${COMPILE}" STREQUAL "HSACOO")
	add_definitions(-DHSACOO)
elseif ("${COMPILE}" STREQUAL "STATIC")
	add_definitions(-DSTATIC)
    file( GLOB KERNEL_CPP "kernel/*.cpp" )
endif()
    add_library( ${PROJECT_NAME} OBJECT ${MOD_CPP} ${KERNEL_CPP})
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP}) # GCC and G++ donst work for creating .so file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -g3 -std=c++14 ")

    set(HIP_INC "/opt/rocm/hip/include")
    add_definitions(-DMOD_CL_PATH="\\"${CMAKE_CURRENT_LIST_DIR}/kernel/"\\")
    # add_definitions(-DMOD_CL_PATH="${CMAKE_CURRENT_LIST_DIR}/kernel/")
    add_definitions(-DOCL_COMPILE)
    add_definitions(-DRPP_BACKEND_HIP)
    add_definitions(-DHIP_COMPILE)
    # add_definitions(-DRPP_BACKEND_HIP)
    add_definitions( -DHIP_OC_COMPILER=${HIP_OC_COMPILER})

    set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries (${PROJECT_NAME} ${OpenCL_LIBRARIES} )
    target_include_directories( ${PROJECT_NAME}
        PUBLIC
            ${CMAKE_SOURCE_DIR}/include
            ${OpenCL_INCLUDE_DIRS}
            ${HIP_INC}
            /opt/rocm/
            /opt/rocm/include
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/include
        )
endif()
