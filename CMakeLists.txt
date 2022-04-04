cmake_minimum_required(VERSION 3.16)
project(kernel_launcher)
set(CMAKE_BUILD_TYPE Release)

file(GLOB sources "src/*cpp")
add_library(kernel_launcher ${sources})

include_directories(include)
include_directories(thirdparty)
add_compile_options(-Wall -Wextra -pedantic)

find_package(Threads REQUIRED)
link_libraries(kernel_launcher Threads::Threads)

#find_package(CUDA REQUIRED)
#link_libraries(kernel_launcher ${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY})
link_libraries(kernel_launcher cuda cudart)
link_libraries(kernel_launcher nvrtc)  # Not offered by CUDA package?

add_subdirectory(examples)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if( supported )
    message(STATUS "IPO / LTO enabled")
    set_property(TARGET kernel_launcher PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
