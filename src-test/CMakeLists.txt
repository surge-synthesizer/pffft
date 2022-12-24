project(surge-pffft-tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cpp compiles.cpp sine.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE .)
target_link_libraries(${PROJECT_NAME} PRIVATE surge::pffft)
