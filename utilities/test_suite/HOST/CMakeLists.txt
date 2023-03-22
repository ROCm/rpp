cmake_minimum_required(VERSION 3.5)
project(test_suite/HOST)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake)
set(ROCM_PATH /opt/rocm CACHE PATH "default ROCm installation path")

# Set message options
if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(Blue        "${Esc}[34m")
    set(BoldBlue    "${Esc}[1;34m")
    set(Magenta     "${Esc}[35m")
    set(Cyan        "${Esc}[36m")
    set(White       "${Esc}[37m")
endif()

find_package(OpenCV QUIET)
find_package(TurboJpeg QUIET)
# BOOST
set(BOOST_COMPONENTS filesystem system)
find_package(Boost COMPONENTS ${BOOST_COMPONENTS} REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})
# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -ggdb -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} OpenMP::OpenMP_CXX)

if(TurboJpeg_FOUND)
    message("-- ${Green}${PROJECT_NAME} set to build with rpp and TurboJpeg${ColourReset}")
    include_directories(${TurboJpeg_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${TurboJpeg_LIBRARIES_DIR})
else()
    message("-- ${Yellow}Error: TurboJpeg must be installed to install ${PROJECT_NAME} successfully!${ColourReset}")
endif()

if (OpenCV_FOUND)
    message("-- ${Green}${PROJECT_NAME} set to build with rpp and OpenCV${ColourReset}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rpp /usr/local/include)
    link_directories(${ROCM_PATH}/lib /usr/local/lib)

    add_executable(Tensor_host Tensor_host.cpp)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")
    target_link_libraries(Tensor_host ${OpenCV_LIBS} -lturbojpeg -lamd_rpp pthread ${LINK_LIBRARY_LIST})

else()
    message("-- ${Yellow}Error: OpenCV must be installed to install ${PROJECT_NAME} successfully!${ColourReset}")
endif()