cmake_minimum_required(VERSION 3.21)
project(py_rpp VERSION 0.2.0 LANGUAGES CXX)

include(GNUInstallDirs)

# ─── Helper functions ─────────────────────────────────────────────────────────

# Resolve the ROCm installation directory.
# Priority: ROCM_PATH env var → .therock_last_install state file → /opt/rocm.
function(py_rpp_find_rocm out_var)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(${out_var} "$ENV{ROCM_PATH}" PARENT_SCOPE)
        return()
    endif()
    foreach(_candidate
        "$ENV{HOME}/amd-workspace"
        "$ENV{HOME}/workspace"
        "/scratch/users/$ENV{USER}"
        "$ENV{HOME}"
    )
        set(_state "${_candidate}/.therock_last_install")
        if(EXISTS "${_state}")
            file(STRINGS "${_state}" _lines)
            foreach(_line IN LISTS _lines)
                if(_line MATCHES "^INSTALL_DIR=(.*)")
                    set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
                    return()
                endif()
            endforeach()
        endif()
    endforeach()
    set(${out_var} "/opt/rocm" PARENT_SCOPE)
endfunction()

# Stage Python package files into <dest_root>, preserving the relative path.
# Usage: py_rpp_stage_files(<dest_root> file1 file2 ...)
function(py_rpp_stage_files dest_root)
    foreach(_f IN LISTS ARGN)
        configure_file(
            "${CMAKE_SOURCE_DIR}/${_f}"
            "${dest_root}/${_f}"
            COPYONLY
        )
    endforeach()
endfunction()

# Apply GPU compile definitions and link HIP when building with HIP support.
function(py_rpp_configure_hip_target target)
    if(NOT TARGET hip::host)
        message(WARNING "hip::host not found; skipping HIP configuration for ${target}")
        return()
    endif()
    target_link_libraries(${target} PRIVATE hip::host)
    target_compile_definitions(${target} PRIVATE
        RPP_BACKEND_HIP
        GPU_SUPPORT          # unlocks #ifdef GPU_SUPPORT blocks in RPP headers
    )
    message(STATUS "py_rpp: HIP GPU backend enabled for ${target}")
endfunction()

# ─── ROCm discovery ───────────────────────────────────────────────────────────

py_rpp_find_rocm(_rocm_path)
set(ROCM_PATH "${_rocm_path}" CACHE PATH "ROCm installation path")
message(STATUS "py_rpp: ROCM_PATH = ${ROCM_PATH}")

# Prepend before find_package(rpp) so its rpp-config.cmake can find HIP.
list(PREPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake")

# ─── Dependencies ─────────────────────────────────────────────────────────────

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(rpp REQUIRED PATHS "${ROCM_PATH}/lib/cmake/rpp")

option(PY_RPP_ENABLE_HIP "Build with HIP GPU backend" ON)
if(PY_RPP_ENABLE_HIP)
    find_package(hip REQUIRED PATHS "${ROCM_PATH}/lib/cmake/hip")
endif()

# ─── Build options ────────────────────────────────────────────────────────────

option(PY_RPP_DEBUG_LOG
    "Compile-in call-trace logging (also runtime: PY_RPP_DEBUG=1)"
    OFF)

# ─── Sources ──────────────────────────────────────────────────────────────────

set(PY_RPP_SOURCES
    src/py_rpp.cpp
    src/ops/color.cpp
    src/ops/geometric.cpp
    src/ops/filter.cpp
    src/ops/morphological.cpp
    src/ops/bitwise.cpp
    src/ops/statistical.cpp
    src/ops/effects.cpp
)

set(PY_RPP_PYTHON_FILES
    py_rpp/__init__.py
    py_rpp/fn.py
    py_rpp/types.py
    py_rpp/utils.py
)

# ─── Extension module ─────────────────────────────────────────────────────────
# Underscore prefix (_py_rpp) follows the upstream AMD convention: the C
# extension is private; py_rpp/__init__.py re-exports the public API.

pybind11_add_module(_py_rpp ${PY_RPP_SOURCES})

target_compile_features(_py_rpp PRIVATE cxx_std_17)
target_include_directories(_py_rpp PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(_py_rpp PRIVATE rpp::rpp)

if(PY_RPP_ENABLE_HIP)
    py_rpp_configure_hip_target(_py_rpp)
else()
    target_compile_definitions(_py_rpp PRIVATE RPP_BACKEND_HOST)
    message(STATUS "py_rpp: CPU-only backend")
endif()

if(PY_RPP_DEBUG_LOG)
    target_compile_definitions(_py_rpp PRIVATE LOG_LEVEL_DEBUG)
    message(STATUS "py_rpp: LOG_LEVEL_DEBUG=ON (compile-time)")
else()
    message(STATUS "py_rpp: LOG_LEVEL_DEBUG=OFF (runtime: PY_RPP_DEBUG=1)")
endif()

# ─── Output layout ────────────────────────────────────────────────────────────
# Place _py_rpp.so under build/py_rpp/ so that PYTHONPATH=build exposes the
# package as `import py_rpp` without an install step.

set_target_properties(_py_rpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/py_rpp"
)

py_rpp_stage_files("${CMAKE_BINARY_DIR}" ${PY_RPP_PYTHON_FILES})

# ─── Install ──────────────────────────────────────────────────────────────────

install(TARGETS _py_rpp
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/py_rpp"
)
install(FILES ${PY_RPP_PYTHON_FILES}
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/py_rpp"
)
