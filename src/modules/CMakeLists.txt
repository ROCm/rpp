#[[
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]

project(modules)

if( "${BACKEND}" STREQUAL "HIP")
    file(GLOB RPP_KERNELS hip/kernel/*.cpp)
elseif( "${BACKEND}" STREQUAL "OCL")
    file(GLOB RPP_KERNELS cl/kernel/*.cl)
endif()

list(APPEND Rpp_Source ${PROJECT_BINARY_DIR}/rpp_kernels.h)
message("-- ${Green}${PROJECT_NAME} -- Adding custom commands to rpp_kernels${ColourReset}")
add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/rpp_kernels.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS addkernels ${RPP_KERNELS} ${RPP_KERNEL_INCLUDES}
    COMMAND ${WINE_CMD} $<TARGET_FILE:addkernels> -guard GUARD_RPP_KERNELS_HPP_ -target ${PROJECT_BINARY_DIR}/rpp_kernels.h -source ${RPP_KERNELS}
    COMMENT "Inlining RPP kernels"
    )

function(add_kernels KERNEL_FILES)
    set(INIT_KERNELS_LIST)
    foreach(KERNEL_FILE ${KERNEL_FILES})
        if("${CMAKE_VERSION}" VERSION_LESS 3.0)
            configure_file(${KERNEL_FILE} ${KERNEL_FILE}.delete)
        else()
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${KERNEL_FILE})
        endif()
        get_filename_component(BASE_NAME ${KERNEL_FILE} NAME_WE)
        string(TOUPPER "${BASE_NAME}" KEY_NAME)
        string(MAKE_C_IDENTIFIER "${KEY_NAME}" VAR_NAME)
        list(APPEND INIT_KERNELS_LIST "    { \"${KEY_NAME}\", std::string(reinterpret_cast<const char*>(${VAR_NAME}), ${VAR_NAME}_SIZE) }")
    endforeach()
    string(REPLACE ";" ",\n" INIT_KERNELS "${INIT_KERNELS_LIST}")
    configure_file(kernels/kernel.cpp.in ${PROJECT_BINARY_DIR}/kernel.cpp)
endfunction()

function(add_kernel_includes KERNEL_FILES)
    set(INIT_KERNELS_LIST)
    foreach(KERNEL_FILE ${KERNEL_FILES})
        if("${CMAKE_VERSION}" VERSION_LESS 3.0)
            configure_file(${KERNEL_FILE} ${KERNEL_FILE}.delete)
        else()
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${KERNEL_FILE})
        endif()
        get_filename_component(BASE_NAME ${KERNEL_FILE} NAME_WE)
        get_filename_component(FILE_NAME ${KERNEL_FILE} NAME)
        string(TOUPPER "${BASE_NAME}" KEY_NAME)
        string(MAKE_C_IDENTIFIER "${KEY_NAME}" VAR_NAME)
        list(APPEND INIT_KERNELS_LIST "    { \"${FILE_NAME}\", std::string(reinterpret_cast<const char*>(${VAR_NAME}), ${VAR_NAME}_SIZE) }")
    endforeach()
    string(REPLACE ";" ",\n" INIT_KERNELS "${INIT_KERNELS_LIST}")
    configure_file(kernels/kernel_includes.cpp.in ${PROJECT_BINARY_DIR}/kernel_includes.cpp)
endfunction()

file(GLOB CPPFILES "*.cpp" )
list(APPEND Rpp_Source ${CPPFILES})
list(APPEND Rpp_Source ${PROJECT_BINARY_DIR}/kernel.cpp)
list(APPEND Rpp_Source ${PROJECT_BINARY_DIR}/kernel_includes.cpp)

if("${TIME_INFO}" STREQUAL "1")
    add_definitions(-DTIME_INFO)
endif()

# Backend specific settings

if( "${BACKEND}" STREQUAL "HIP")
    # Add HIP kernels
    file(GLOB MOD_HIP_CPP "hip/*.cpp" )
    file(GLOB MOD_HIP_CPP_KERNELS "hip/kernel/*.cpp" )
    list(APPEND Rpp_Source ${CPPFILES} ${MOD_HIP_CPP} ${MOD_HIP_CPP_KERNELS})
    message("-- ${Green}HIP kernels added${ColourReset}")

    # Set compiler flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HIPCC_FLAGS}")
    set_source_files_properties(rppt_tensor_audio_augmentations.cpp PROPERTIES COMPILE_FLAGS -mno-fma)
    set_source_files_properties(rppt_tensor_statistical_operations.cpp PROPERTIES COMPILE_FLAGS -mno-fma) # no-fma flag added to get the exact output match with golden outputs

    # Add HIP specific preprocessor flags
    add_definitions(-DHIP_COMPILE)

    # Add HIP specific includes
    set(ROCM_INC ${ROCM_PATH}/include/)
    list(APPEND HIP_LOCAL_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/include/hip/ ${CMAKE_SOURCE_DIR}/src/include/common/)
    set(INCLUDE_LIST ${ROCM_INC} ${HIP_LOCAL_INCLUDE_DIRS} ${INCLUDE_LIST})
elseif( "${BACKEND}" STREQUAL "OCL")
    # Add OpenCL kernels
    file(GLOB MOD_CL_CPP "cl/*.cpp" )
    list(APPEND Rpp_Source ${CPPFILES} ${MOD_CL_CPP})
    message("-- ${Green}OpenCL kernels added!${ColourReset}")
    set_source_files_properties(rppt_tensor_audio_augmentations.cpp PROPERTIES COMPILE_FLAGS -mno-fma)
    set_source_files_properties(rppt_tensor_statistical_operations.cpp PROPERTIES COMPILE_FLAGS -mno-fma) # no-fma flag added to get the exact output match with golden outputs

    # Add OpenCL specific preprocessor flags
    add_definitions(-DOCL_COMPILE)
    add_definitions(-DRPP_CACHE_DIR="${CACHE_DIR}")
    add_definitions(-DMOD_CL_PATH="${CMAKE_CURRENT_LIST_DIR}/cl/kernel/")

    # Add OpenCL specific includes
    set(ROCM_INC ${ROCM_PATH}/include/)
    list(APPEND OCL_LOCAL_INCLUDE_LIST ${CMAKE_SOURCE_DIR}/src/include/cl/ ${CMAKE_SOURCE_DIR}/src/include/common/)
    set(INCLUDE_LIST ${ROCM_INC} ${OCL_LOCAL_INCLUDE_LIST} ${INCLUDE_LIST})
elseif( "${BACKEND}" STREQUAL "CPU")
    # Add CPU specific includes
    set(INCLUDE_LIST ${CMAKE_SOURCE_DIR}/src/include/common/)
    set_source_files_properties(rppt_tensor_audio_augmentations.cpp PROPERTIES COMPILE_FLAGS -mno-fma)
    set_source_files_properties(rppt_tensor_statistical_operations.cpp PROPERTIES COMPILE_FLAGS -mno-fma) # no-fma flag added to get the exact output match with golden outputs
endif()
message("-- ${White}AMD RPP ${PROJECT_NAME} -- Include Directories:${INCLUDE_LIST}${ColourReset}")
add_compile_options("-Wno-unused-result")

# Kernels and includes additions

add_kernels("${RPP_KERNELS}")
add_kernel_includes("${RPP_KERNEL_INCLUDES}")

# Add library
add_library(${PROJECT_NAME} OBJECT ${Rpp_Source})

# Target settings
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories( ${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${ROCM_INC}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include/cpu
        ${CMAKE_SOURCE_DIR}/src/include/common
        ${CMAKE_SOURCE_DIR}/src/include/func_specific
        ${INCLUDE_LIST}
)

if( "${BACKEND}" STREQUAL "HIP")
    target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include/hip
    )
    target_link_libraries(${PROJECT_NAME} hip::device)
elseif( "${BACKEND}" STREQUAL "OCL")
    target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include/cl
    )
endif()
