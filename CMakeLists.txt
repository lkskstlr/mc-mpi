cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(mc-mpi CXX)

add_library(libmc-mpi SHARED
    src/random.cpp
    src/particle.cpp
    src/layer.cpp
)

# Tests
add_executable(test src/test.cpp)
target_include_directories(test PRIVATE include)
set_target_properties(test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_link_libraries(test
    libmc-mpi
)