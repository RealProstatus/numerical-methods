add_executable(single_cpu main.cpp)

target_link_libraries(single_cpu PRIVATE
    one_cpu_solver
    computational_methods
)
