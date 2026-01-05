# FindDLPACK.cmake
# Finds the DLPack header-only library
#
# This module defines:
#  DLPACK_FOUND - System has DLPack
#  DLPACK_INCLUDE_DIRS - The DLPack include directories

# DLPack is header-only, so we just need to find the include directory

# Look for dlpack/dlpack.h in common locations
find_path(DLPACK_INCLUDE_DIRS
    NAMES dlpack/dlpack.h
    PATHS
        /usr/include
        /usr/local/include
        ${ROCM_PATH}/include
        $ENV{DLPACK_PATH}/include
        $ENV{HOME}/dlpack/include
    DOC "Path to DLPack include directory"
)

# Handle standard find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DLPACK
    FOUND_VAR DLPACK_FOUND
    REQUIRED_VARS DLPACK_INCLUDE_DIRS
    FAIL_MESSAGE "DLPack not found. Please install DLPack or set DLPACK_PATH"
)

# Mark as advanced (don't show in cmake-gui by default)
mark_as_advanced(DLPACK_INCLUDE_DIRS)

if(DLPACK_FOUND)
    if(NOT TARGET DLPACK::DLPACK)
        # Create an interface target for modern CMake usage
        add_library(DLPACK::DLPACK INTERFACE IMPORTED)
        set_target_properties(DLPACK::DLPACK PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${DLPACK_INCLUDE_DIRS}"
        )
    endif()
    
    message(STATUS "Found DLPack: ${DLPACK_INCLUDE_DIRS}")
endif()
