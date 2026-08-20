if(BUILD_SINGLE_CPU)
    add_subdirectory(single_cpu)
endif()

if(BUILD_MULTI_CPU)
    add_subdirectory(multi_cpu)
endif()

if(BUILD_SINGLE_GPU)
    add_subdirectory(single_gpu)
endif()