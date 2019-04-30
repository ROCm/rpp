cmake_minimum_required(VERSION 3.5)
project(imgaug) #Rename the Project name

file( GLOB MOD_CPP "*.cpp" "*.hpp" )
add_library( ${PROJECT_NAME} STATIC ${MOD_CPP})

if( "${BACKEND}" STREQUAL "HIP")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hip)
    target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_hip )
elseif( "${BACKEND}" STREQUAL "OCL")
    message (SEND_ERROR "Module Not implemented")
else() #CPU version
    message (SEND_ERROR "Module Not implemented")
endif()

target_include_directories( ${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)
