project(matmul_example)
cmake_minimum_required(VERSION 3.16)

add_executable(example_matmul main.cpp)
target_link_libraries(example_matmul PRIVATE kernel_launcher)