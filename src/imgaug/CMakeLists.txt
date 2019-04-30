cmake_minimum_required(VERSION 3.5)
project(rpp_imgaug LANGUAGES CXX) #Rename the Project name


if(DEFINED BACKEND_HIP)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hip)
    target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_hip.a  )
elseif(DEFINED BACKEND_OCL)
    message (SEND_ERROR "Module Not implemented")
else() #CPU version
    message (SEND_ERROR "Module Not implemented")
endif()

file( GLOB MOD_CPP ".cpp" ".hpp" )
add_library( ${PROJECT_NAME} STATIC ${MOD_CPP} )
target_include_directories( ${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)
