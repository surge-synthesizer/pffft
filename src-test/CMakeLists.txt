project(surge-pffft-tests LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} main.cpp compiles.cpp convolve.cpp fftpack.c sine.cpp validity.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE .)
target_link_libraries(${PROJECT_NAME} PRIVATE surge::pffft)
