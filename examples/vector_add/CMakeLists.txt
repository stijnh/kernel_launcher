project(matmul_example)
cmake_minimum_required(VERSION 3.16)

add_executable(example_vector_add main.cpp)
target_link_libraries(example_vector_add PRIVATE kernel_launcher)