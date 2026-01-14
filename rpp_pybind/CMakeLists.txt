cmake_minimum_required(VERSION 3.12)

# Add parent's cmake modules to path (for FindHALF.cmake and FindDLPACK.cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

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

    # -------------------------------------------------------------------------
    # Find PyTorch from the same Python that we're building against
    # -------------------------------------------------------------------------
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_PREFIX_RC
        ERROR_QUIET
    )

    if(NOT TORCH_PREFIX_RC EQUAL 0 OR TORCH_CMAKE_PREFIX_PATH STREQUAL "")
        message("-- ${Yellow}PyTorch not found for Python ${PY_VERSION} (${Python3_EXECUTABLE}) - skipping pybind build${ColourReset}")
        continue()
    endif()

    # TorchConfig.cmake lives under this prefix path
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

    # Ensure find_package(Torch) re-runs for each Python version
    unset(Torch_DIR CACHE)
    unset(Torch_FOUND CACHE)

    find_package(Torch REQUIRED)

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
        OUTPUT_VARIABLE TORCH_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_LIBDIR_RC
        ERROR_QUIET
    )

    if(NOT TORCH_LIBDIR_RC EQUAL 0 OR TORCH_LIB_DIR STREQUAL "")
        message(FATAL_ERROR "Failed to detect torch lib directory for ${Python3_EXECUTABLE}")
    endif()

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
        ${TORCH_INCLUDE_DIRS}
    )

    # Compile definitions
    target_compile_definitions(${mod_tgt} PRIVATE WITH_PYTORCH=1)
    target_compile_definitions(${mod_tgt} PUBLIC ENABLE_DLPACK=1)

    # Link libraries
    target_link_libraries(${mod_tgt} PRIVATE rpp ${Python3_LIBRARIES} ${TORCH_LIBRARIES})

    # Torch's Python bridge library is required for pybind11 type casters for at::Tensor
    # (Fixes: undefined symbol: pybind11::detail::type_caster<at::Tensor>::load)
    target_link_libraries(${mod_tgt} PRIVATE "${TORCH_LIB_DIR}/libtorch_python.so")

    # Ensure torch libs are found at runtime without LD_LIBRARY_PATH
    set_target_properties(${mod_tgt} PROPERTIES
        BUILD_RPATH "${TORCH_LIB_DIR}"
        INSTALL_RPATH "${TORCH_LIB_DIR}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )

    add_dependencies(${mod_tgt} rpp)

    # Get Python extension suffix
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
        OUTPUT_VARIABLE Python3_EXTENSION_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Set module properties.
    # Output name is _rpp_pybind to avoid name collision with the rpp_pybind Python package.
    set_target_properties(${mod_tgt} PROPERTIES
        PREFIX ""
        SUFFIX "${Python3_EXTENSION_SUFFIX}"
        OUTPUT_NAME "_rpp_pybind"
    )

    # Install .so file inside the python package directory
    install(TARGETS ${mod_tgt}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/rpp_pybind
            COMPONENT dev)

    message("-- ${Green}RPP pybind module ${mod_tgt} added${ColourReset}")
endforeach()

# Install Python package files (pure Python)
# NOTE: We must preserve the directory structure (amd/..., amd/rpp/...) so that
# imports like `rpp_pybind.amd.rpp.fn` work.
install(
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/fn.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/rpp_pybind
    COMPONENT dev
)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/amd"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/rpp_pybind
    COMPONENT dev
    FILES_MATCHING PATTERN "*.py"
)

message("-- ${Green}RPP Python bindings configured${ColourReset}")
