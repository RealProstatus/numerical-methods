cmake_minimum_required(VERSION 3.24)

project(numerical_methods LANGUAGES CXX)

option(BUILD_SINGLE_CPU "Build the single-node CPU executable." ON)
option(BUILD_SINGLE_GPU "Reserve the single-node GPU target. Implementation is not added yet." OFF)
option(BUILD_MULTI_CPU  "Build the multi-node MPI CPU executable." OFF)
option(BUILD_MULTI_GPU "Reserve the multi-node GPU target. Implementation is not added yet." OFF)
option(BUILD_TESTS "Enable GoogleTest-based tests." OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
#    message(FATAL_ERROR
#        "This project requires the Intel oneAPI C++ compiler. "
#        "Run CMake from an Intel oneAPI environment and configure with icx/icx-cl.")
#endif()

include(cmake/Dependencies.cmake)

if(BUILD_MULTI_CPU OR BUILD_MULTI_GPU)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

add_subdirectory(src)

if(BUILD_SINGLE_CPU)
    add_subdirectory(apps/single_cpu)
endif()

if(BUILD_SINGLE_GPU)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_subdirectory(apps/single_gpu) 
endif()

if(BUILD_MULTI_CPU)
    add_subdirectory(apps/multi_cpu)
endif()

if(BUILD_MULTI_GPU)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(FATAL_ERROR "multi_gpu is reserved, but MPI+CUDA implementation sources are not added yet.")
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()