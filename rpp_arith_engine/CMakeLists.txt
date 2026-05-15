cmake_minimum_required(VERSION 3.15)
project(rpp_arith_engine LANGUAGES CXX)

# ─── ROCM_PATH (mirrors logic in py_rpp/CMakeLists.txt) ──────────────────────
if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm installation path")
elseif(NOT DEFINED ROCM_PATH OR ROCM_PATH STREQUAL "")
    set(_candidates
        "$ENV{HOME}/amd-workspace"
        "$ENV{HOME}/workspace"
        "/scratch/users/$ENV{USER}"
        "$ENV{HOME}"
    )
    set(_found "")
    foreach(_c IN LISTS _candidates)
        if(EXISTS "${_c}/.therock_last_install")
            file(STRINGS "${_c}/.therock_last_install" _lines)
            foreach(_l IN LISTS _lines)
                if(_l MATCHES "^INSTALL_DIR=(.*)")
                    set(_found "${CMAKE_MATCH_1}")
                    break()
                endif()
            endforeach()
            if(NOT _found STREQUAL "")
                break()
            endif()
        endif()
    endforeach()
    if(NOT _found STREQUAL "")
        set(ROCM_PATH "${_found}" CACHE PATH "ROCm (auto)")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation path")
    endif()
endif()
message(STATUS "rpp_arith_engine: ROCM_PATH = ${ROCM_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake")

# ─── Dependencies ─────────────────────────────────────────────────────────────
find_package(rpp REQUIRED PATHS ${ROCM_PATH}/lib/cmake/rpp)
find_package(hip REQUIRED PATHS ${ROCM_PATH}/lib/cmake/hip)

# ─── Library ──────────────────────────────────────────────────────────────────
add_library(rpp_arith_engine SHARED src/rpp_arith_engine.cpp)

target_compile_features(rpp_arith_engine PUBLIC cxx_std_17)
target_compile_definitions(rpp_arith_engine PUBLIC RPP_BACKEND_HIP)

target_include_directories(rpp_arith_engine
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${ROCM_PATH}/include
)
target_link_libraries(rpp_arith_engine PUBLIC rpp::rpp hip::host)

# ─── Test executable ──────────────────────────────────────────────────────────
add_executable(test_arith_engine tests/test_arith_engine.cpp)
target_link_libraries(test_arith_engine PRIVATE rpp_arith_engine)
