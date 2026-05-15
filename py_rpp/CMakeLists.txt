cmake_minimum_required(VERSION 3.15)
project(py_rpp LANGUAGES CXX)

# ─── ROCM_PATH: env var → state file probe → /opt/rocm ───────────────────────
if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm installation path")
elseif(NOT DEFINED ROCM_PATH OR ROCM_PATH STREQUAL "")
    # Mirror the candidate-list search from set_rocm_env.sh
    set(_state_candidates
        "$ENV{HOME}/amd-workspace"
        "$ENV{HOME}/workspace"
        "/scratch/users/$ENV{USER}"
        "$ENV{HOME}"
    )
    set(_found_rocm "")
    foreach(_candidate IN LISTS _state_candidates)
        set(_state_file "${_candidate}/.therock_last_install")
        if(EXISTS "${_state_file}")
            file(STRINGS "${_state_file}" _state_lines)
            foreach(_line IN LISTS _state_lines)
                if(_line MATCHES "^INSTALL_DIR=(.*)")
                    set(_found_rocm "${CMAKE_MATCH_1}")
                    break()
                endif()
            endforeach()
            if(NOT _found_rocm STREQUAL "")
                break()
            endif()
        endif()
    endforeach()
    if(NOT _found_rocm STREQUAL "")
        set(ROCM_PATH "${_found_rocm}" CACHE PATH "ROCm installation path (auto)")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation path")
    endif()
endif()
message(STATUS "py_rpp: ROCM_PATH = ${ROCM_PATH}")

# Let transitive find_dependency() calls inside rpp-config.cmake locate HIP etc.
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake")

# ─── Python + pybind11 ────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# ─── RPP ──────────────────────────────────────────────────────────────────────
find_package(rpp REQUIRED PATHS ${ROCM_PATH}/lib/cmake/rpp)

# ─── HIP ──────────────────────────────────────────────────────────────────────
find_package(hip REQUIRED PATHS ${ROCM_PATH}/lib/cmake/hip)

# ─── Build ────────────────────────────────────────────────────────────────────
option(LOG_LEVEL_DEBUG "Always-on call-trace logging (also controllable at runtime via PY_RPP_DEBUG=1)" OFF)

pybind11_add_module(py_rpp src/py_rpp.cpp)

target_compile_features(py_rpp PRIVATE cxx_std_17)
target_compile_definitions(py_rpp PRIVATE RPP_BACKEND_HIP)

if(LOG_LEVEL_DEBUG)
    target_compile_definitions(py_rpp PRIVATE LOG_LEVEL_DEBUG)
    message(STATUS "py_rpp: LOG_LEVEL_DEBUG=ON (always-on logging)")
else()
    message(STATUS "py_rpp: LOG_LEVEL_DEBUG=OFF (runtime: PY_RPP_DEBUG=1)")
endif()

target_link_libraries(py_rpp PRIVATE
    rpp::rpp
    hip::host
)

# ─── Place .so inside build/py_rpp/ so `PYTHONPATH=build python -m pytest` works ─
set_target_properties(py_rpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/py_rpp
)
configure_file(
    ${CMAKE_SOURCE_DIR}/py_rpp/__init__.py
    ${CMAKE_BINARY_DIR}/py_rpp/__init__.py
    COPYONLY
)
