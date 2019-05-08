cmake_minimum_required(VERSION 3.5)
project(amd_rpp VERSION 0.1)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are not allowed. Invoke cmake from build directory")
endif ()

add_library(${PROJECT_NAME} SHARED include/rppi.h)

#TODO: Future Work Build list can be set from CLI ; CACHED and to be used
set( MODULES_LIST "imgaug")

foreach (MOD_NAME ${MODULES_LIST})
    add_subdirectory(${CMAKE_SOURCE_DIR}/src/${MOD_NAME})
endforeach(MOD_NAME)

if( "${BACKEND}" STREQUAL "HIP")
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
elseif( "${BACKEND}" STREQUAL "OCL")
    set(CMAKE_CXX_COMPILER /usr/bin/gcc)
endif()


set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${PROJECT_NAME} ${MODULES_LIST})



target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

install (TARGETS amd_rpp
         LIBRARY DESTINATION /opt/rocm/rpp/lib
)
install (DIRECTORY include
         DESTINATION /opt/rocm/rpp/include
)