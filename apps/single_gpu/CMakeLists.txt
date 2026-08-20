if(BUILD_SINGLE_GPU)
    add_executable(single_gpu_app gpu_verification.cu)
    
    set_target_properties(single_gpu_app PROPERTIES CUDA_ARCHITECTURES 89)
    
    target_include_directories(single_gpu_app PRIVATE "${CMAKE_SOURCE_DIR}/src/solvers/single_gpu")
    
    target_link_libraries(single_gpu_app PRIVATE computational_methods CUDA::cublas)
endif()