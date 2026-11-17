set(GOOGLETEST_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/googletest")

if(EXISTS "${GOOGLETEST_SUBMODULE_DIR}/CMakeLists.txt")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory("${GOOGLETEST_SUBMODULE_DIR}" "${CMAKE_BINARY_DIR}/3rdparty/googletest")
    message(STATUS "GoogleTest is configured.")
else()
    message(STATUS "GoogleTest submodule not found; building standalone native tests without GTest.")
endif()

add_executable(test_native_core test_native_core.cpp)

target_include_directories(test_native_core PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(test_native_core PRIVATE
    one_cpu_solver
    computational_methods
)

if(BUILD_SINGLE_GPU)
    target_compile_definitions(test_native_core PRIVATE MAGNUS_HAS_CUDA=1)
    target_include_directories(test_native_core PRIVATE
        "${CMAKE_SOURCE_DIR}/src/solvers/single_gpu"
    )
endif()

enable_testing()
add_test(NAME test_native_core COMMAND test_native_core)