project(tests)
cmake_minimum_required(VERSION 3.16)

file(GLOB FILES *.cpp)
add_executable(tests ${FILES})
target_link_libraries(tests PRIVATE kernel_launcher)

