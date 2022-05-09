project(modules)

if( "${BACKEND}" STREQUAL "HIP")
    file(GLOB RPP_KERNELS hip/kernel/*.cpp)
elseif( "${BACKEND}" STREQUAL "OCL")
    file(GLOB RPP_KERNELS cl/kernel/*.cl)
endif()

list(APPEND Rpp_Source ${PROJECT_BINARY_DIR}/rpp_kernels.h)
message("-- Added custom command for rpp_kernels")
add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/rpp_kernels.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS addkernels ${RPP_KERNELS} ${RPP_KERNEL_INCLUDES}
    COMMAND ${WINE_CMD} $<TARGET_FILE:addkernels> -guard GUARD_RPP_KERNELS_HPP_ -target ${PROJECT_BINARY_DIR}/rpp_kernels.h -source ${RPP_KERNELS}
    COMMENT "Inlining Rpp kernels"
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
    message("-- ${Green}HIP kernels added!${ColourReset}")

    # Set HIP compiler and flags
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HIPCC_FLAGS} -fopenmp -Ofast -msse4.2 -msse4.1 -mssse3 -mavx2 -mfma -g3 -std=c++14")

    # Add HIP specific preprocessor flags
    add_definitions(-DHIP_COMPILE)

    # Add HIP specific includes
    set(HIP_INC "/opt/rocm/hip/include")
    set(INCLUDE_LIST ${HIP_INC} ${INCLUDE_LIST})
    message("-- ${Blue}HIP include directories -- ${INCLUDE_LIST}${ColourReset}")

elseif( "${BACKEND}" STREQUAL "OCL")

    # Add OpenCL kernels
    file(GLOB MOD_CL_CPP "cl/*.cpp" )
    list(APPEND Rpp_Source ${CPPFILES} ${MOD_CL_CPP})
    message("-- ${Green}OpenCL kernels added!${ColourReset}")

    # Set OpenCL compiler and flags
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_OPENCL}) # GCC and G++ donst work for creating .so file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Ofast -msse4.2 -msse4.1 -mssse3 -mavx2 -mfma -g3 -std=c++14 -Wno-deprecated-declarations")

    # Add OpenCL specific preprocessor flags
    add_definitions(-DOCL_COMPILE)
    add_definitions(-DRPP_CACHE_DIR="${CACHE_DIR}")
    add_definitions(-DMOD_CL_PATH="${CMAKE_CURRENT_LIST_DIR}/cl/kernel/")

    # Add OpenCL specific includes
    set(ROCM_INC "/opt/rocm/include")
    list(APPEND OpenCL_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/include/cl/)
    set(INCLUDE_LIST ${OpenCL_INCLUDE_DIRS} ${INCLUDE_LIST} ${ROCM_INC})
    message("-- ${Blue}OpenCL include directories -- ${INCLUDE_LIST}${ColourReset}")

endif()

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
        ${HIP_INC}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${INCLUDE_LIST}
)