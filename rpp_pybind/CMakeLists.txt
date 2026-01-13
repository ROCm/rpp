cmake_minimum_required(VERSION 3.12)

# Add parent's cmake modules to path (for FindHALF.cmake and FindDLPACK.cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

set(TARGET_NAME rpp_pybind)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
if(NOT DEFINED CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
endif()

# Find required packages
find_package(pybind11 REQUIRED CONFIG)
find_package(HALF REQUIRED)
find_package(DLPACK REQUIRED)

# Find PyTorch
execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_CMAKE_PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
endif()

find_package(Torch QUIET)

# Multi-version Python Support
set(PYTHON_VERSIONS "3.13" "3.12" "3.11" "3.10" "3.9" "3.8")

if(PYTHON_VERSION_SUGGESTED)
    set(PYTHON_VERSIONS "${PYTHON_VERSION_SUGGESTED}")
endif()

# Build for each Python version
foreach(PY_VERSION ${PYTHON_VERSIONS})
    message(STATUS "Searching for Python ${PY_VERSION}")

    # Clear previous Python variables
    foreach(v IN ITEMS Python3_EXECUTABLE Python3_VERSION Python3_INCLUDE_DIRS Python3_LIBRARIES)
        unset(${v} CACHE)
        unset(${v})
    endforeach()

    # Find specific Python version
    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 "${PY_VERSION}" EXACT QUIET COMPONENTS Interpreter Development)

    if(NOT Python3_FOUND)
        message("-- ${Yellow}Python ${PY_VERSION} not found${ColourReset}")
        continue()
    endif()

    message("-- ${Green}Building for Python ${PY_VERSION}: ${Python3_EXECUTABLE}${ColourReset}")

    # Unique module target
    set(mod_tgt "rpp_pybind_${PY_VERSION}")
    
    # Build pybind module
    pybind11_add_module(${mod_tgt} MODULE rpp_pybind.cpp)
    
    # Include directories
    target_include_directories(${mod_tgt} PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
        ${HALF_INCLUDE_DIRS}
        ${DLPACK_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/api
    )
    
    # Add PyTorch includes if found
    if(Torch_FOUND)
        target_include_directories(${mod_tgt} PRIVATE ${TORCH_INCLUDE_DIRS})
        target_compile_definitions(${mod_tgt} PRIVATE WITH_PYTORCH=1)
    endif()
    
    # Link libraries
    target_link_libraries(${mod_tgt} PRIVATE rpp ${Python3_LIBRARIES})
    
    # Link PyTorch if found
    if(Torch_FOUND)
        target_link_libraries(${mod_tgt} PRIVATE ${TORCH_LIBRARIES})
    endif()
    
    add_dependencies(${mod_tgt} rpp)
    
    # DLPack always enabled (required)
    target_compile_definitions(${mod_tgt} PUBLIC ENABLE_DLPACK=1)
    
    # Get Python extension suffix
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
        OUTPUT_VARIABLE Python3_EXTENSION_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # Set module properties
    set_target_properties(${mod_tgt} PROPERTIES
        PREFIX ""
        SUFFIX "${Python3_EXTENSION_SUFFIX}"
        OUTPUT_NAME "rpp_pybind"
    )
    
    # Install .so file
    install(TARGETS ${mod_tgt}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT dev)
    
    message("-- ${Green}RPP pybind module ${mod_tgt} added${ColourReset}")
    
endforeach()

# Install Python package files
set(PYTHON_FILES
    __init__.py
    fn.py
    types.py
    utils.py
    test_suite.py
    test_unit.py
    test_performance.py
    test_qa.py
    example_simple.py
    run_all_tests.py
    README.md
)

foreach(file ${PYTHON_FILES})
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${file}"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/rpp_pybind
            COMPONENT dev)
endforeach()

message("-- ${Green}RPP Python bindings configured${ColourReset}")
