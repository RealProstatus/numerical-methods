set(CORE_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/matrix"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/utils"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/magnus"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/expm"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/integrators"
    "${CMAKE_CURRENT_SOURCE_DIR}/solvers/single_cpu"
)

add_library(computational_methods STATIC
    core/matrix/MatrixOperations.h
    core/utils/Utils.cpp
    core/utils/Utils.h
    core/utils/ProblemSetup.cpp
    core/utils/ProblemSetup.h
    core/magnus/ClassicMagnus.cpp
    core/magnus/ClassicMagnus.h
    core/magnus/AccMagnus.cpp
    core/magnus/AccMagnus.h
    core/magnus/RecursiveMagnus.cpp
    core/magnus/RecursiveMagnus.h
    core/magnus/Formula314Magnus.cpp
    core/magnus/Formula314Magnus.h
    core/expm/ChebyshevExpm.cpp
    core/expm/ChebyshevExpm.h
    core/expm/TaylorExpm.cpp
    core/expm/TaylorExpm.h
    core/expm/EigenPadeExpm.cpp
    core/expm/EigenPadeExpm.h
    core/integrators/RungeKutt.h
    core/include/AllComputationalMethods.h
)

if(BUILD_SINGLE_GPU)
    find_package(CUDAToolkit REQUIRED)
    
    target_sources(computational_methods PRIVATE core/matrix/MatrixOperations.cu)
    target_link_libraries(computational_methods PUBLIC CUDA::cublas)
    
    # Задаем архитектуру Ada Lovelace (8.9) для компиляции ядер под RTX 4060
    set_target_properties(computational_methods PROPERTIES CUDA_ARCHITECTURES 89)
else()
    target_sources(computational_methods PRIVATE core/matrix/MatrixOperations.cpp)
endif()

target_include_directories(computational_methods PUBLIC ${CORE_INCLUDE_DIRS})
target_link_libraries(computational_methods PUBLIC project_options mkl_dependency eigen_dependency)

add_library(one_cpu_solver STATIC
    solvers/single_cpu/PiecewiseMagnus.cpp
    solvers/single_cpu/PiecewiseMagnus.h
    solvers/single_cpu/Formula314ChebyshevSolver.cpp
    solvers/single_cpu/Formula314ChebyshevSolver.h
)

target_include_directories(one_cpu_solver PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/solvers/single_cpu"
)

target_link_libraries(one_cpu_solver PUBLIC computational_methods)

if(BUILD_MULTI_CPU)
    add_library(multi_cpu_solver STATIC
        solvers/multi_cpu/MpiMagnusChebyshev.cpp
        solvers/multi_cpu/MpiMagnusChebyshev.h
    )

    target_include_directories(multi_cpu_solver PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/solvers/multi_cpu"
    )

    target_link_libraries(multi_cpu_solver PUBLIC
        computational_methods
        one_cpu_solver
        MPI::MPI_CXX
    )
endif()