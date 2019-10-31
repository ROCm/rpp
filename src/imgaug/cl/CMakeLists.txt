cmake_minimum_required(VERSION 3.5)
project(imgaug-opencl)

if( BACKEND_HIPOC)
    file( GLOB MOD_CL_KERNEL "kernel/*.cl" )
    foreach (KERNEL_FILE ${MOD_CL_KERNEL})
        get_filename_component(KERN_FILE_NAME ${KERNEL_FILE} NAME)
        add_library( ${KERN_FILE_NAME} ${KERNEL_FILE})
        set_target_properties(${KERN_FILE_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(${KERN_FILE_NAME} PROPERTIES LINKER_LANGUAGE CXX)
    endforeach(KERNEL_FILE)
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIPOC})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    add_definitions(-DHIP_COMPILE)
    #add_definitions(-DMOD_CL_PATH="\\"${CMAKE_CURRENT_LIST_DIR}/kernel/"\\")
    add_definitions(-DMOD_CL_PATH="${CMAKE_CURRENT_LIST_DIR}/kernel/")

    link_libraries(-amdgpu-target=gfx803 -amdgpu-target=gfx900 -Wno-unused-command-line-argument)
else()
    file( GLOB MOD_CPP "*.cpp" )
    add_library( ${PROJECT_NAME} OBJECT ${MOD_CPP})
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_OPENCL}) # GCC and G++ donst work for creating .so file
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    #add_definitions(-DMOD_CL_PATH="\\"${CMAKE_CURRENT_LIST_DIR}/kernel/"\\")
    add_definitions(-DMOD_CL_PATH="${CMAKE_CURRENT_LIST_DIR}/kernel/")

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
endif()
