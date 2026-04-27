add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)

if(MSVC)
    target_compile_definitions(project_options INTERFACE _USE_MATH_DEFINES NOMINMAX)
endif()

set(EIGEN_SUBMODULE_DIR "${CMAKE_CURRENT_LIST_DIR}/../3rdparty/eigen")

if(NOT EXISTS "${EIGEN_SUBMODULE_DIR}/Eigen/Core")
    message(FATAL_ERROR
        "Eigen submodule was not found. Run: git submodule update --init --recursive")
endif()

add_library(eigen_dependency INTERFACE)
target_include_directories(eigen_dependency INTERFACE "${EIGEN_SUBMODULE_DIR}")

find_package(MKL CONFIG REQUIRED)

if(NOT TARGET MKL::MKL)
    message(FATAL_ERROR "MKLConfig.cmake was found, but target MKL::MKL is unavailable.")
endif()

add_library(mkl_dependency INTERFACE)
target_link_libraries(mkl_dependency INTERFACE MKL::MKL)
