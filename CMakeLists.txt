cmake_minimum_required(VERSION 3.10)
project(pffft VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

message(STATUS "Including local pffft")
add_library(${PROJECT_NAME} pffft.c pffft.h pffft.hpp)
add_library(surge::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} INTERFACE .)

if (${SURGE_PFFFT_BUILD_TESTS})
    add_subdirectory(src-test)
endif()