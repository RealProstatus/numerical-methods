cmake_minimum_required(VERSION 3.24)

project(numerical_methods LANGUAGES CXX)

option(BUILD_SINGLE_CPU "Build the single-node CPU executable." ON)
option(BUILD_SINGLE_GPU "Reserve the single-node GPU target." OFF)
option(BUILD_MULTI_CPU  "Build the multi-node MPI CPU executable." OFF)
option(BUILD_MULTI_GPU "Reserve the multi-node GPU target." OFF)
option(BUILD_TESTS "Enable GoogleTest-based tests." OFF)
option(BUILD_PYTHON_BINDINGS "Build Python extension module via nanobind" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(BUILD_SINGLE_GPU OR BUILD_MULTI_GPU)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(BUILD_MULTI_CPU OR BUILD_MULTI_GPU)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

# Подключение зависимостей проекта (Eigen, MKL)
include(cmake/Dependencies.cmake)


add_subdirectory(src)

if(BUILD_SINGLE_CPU)
    add_subdirectory(apps/single_cpu)
endif()

if(BUILD_SINGLE_GPU)
    add_subdirectory(apps/single_gpu) 
endif()

if(BUILD_MULTI_CPU)
    add_subdirectory(apps/multi_cpu)
endif()

if(BUILD_MULTI_GPU)
    message(FATAL_ERROR "multi_gpu is reserved, but MPI+CUDA implementation sources are not added yet.")
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()


if(BUILD_PYTHON_BINDINGS)
    find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    )
    list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_magnus_py STABLE_ABI NB_STATIC
        python/bindings/module.cpp
    )

    target_link_libraries(_magnus_py PRIVATE magnus_bridge)
    target_include_directories(_magnus_py PRIVATE ${CMAKE_SOURCE_DIR}/include)

    install(TARGETS _magnus_py
        LIBRARY DESTINATION pymagnus_hpc
        RUNTIME DESTINATION pymagnus_hpc
    )
endif()