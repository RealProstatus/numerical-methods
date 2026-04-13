set(GOOGLETEST_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/googletest")

if(NOT EXISTS "${GOOGLETEST_SUBMODULE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "GoogleTest submodule was not found. Add it with: "
        "git submodule add https://github.com/google/googletest.git 3rdparty/googletest")
endif()

add_subdirectory("${GOOGLETEST_SUBMODULE_DIR}" "${CMAKE_BINARY_DIR}/3rdparty/googletest")

message(STATUS "GoogleTest is configured. Test sources will be added in a later step.")
