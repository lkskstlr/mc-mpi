cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(mc-mpi C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
  message(STATUS "Setting Build Type to Release (default)")
endif(NOT CMAKE_BUILD_TYPE)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -pedantic")

########## Required Packages ##########
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX) #For openmpi
if(NOT TARGET MPI::MPI_C)
# As in https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
    add_library(MPI::MPI_C IMPORTED INTERFACE)
    set_property(TARGET MPI::MPI_C
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_C_COMPILE_FLAGS})
    set_property(TARGET MPI::MPI_C
                 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_C_INCLUDE_PATH}")
    set_property(TARGET MPI::MPI_C
                 PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_C_LINK_FLAGS} ${MPI_C_LIBRARIES})
endif()


########## Libraries ##########
add_library(yaml SHARED
  src/yaml_loader.cpp
  src/yaml_dumper.cpp
)
target_include_directories(yaml PUBLIC include/yaml)
set_target_properties(yaml PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)


add_library(timer SHARED
  src/timer.cpp
)
target_include_directories(timer PUBLIC include/timer)
set_target_properties(timer PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(timer
  PUBLIC MPI::MPI_C
)


add_library(types INTERFACE)
target_include_directories(types INTERFACE include/types)
target_compile_features(types INTERFACE cxx_std_11)
target_link_libraries(types INTERFACE MPI::MPI_C)


add_library(layer SHARED src/layer.cpp src/random.cpp)
target_include_directories(layer PUBLIC include/layer)
target_include_directories(layer PRIVATE include/layer/private)
set_target_properties(layer PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(layer
  PUBLIC types
  PRIVATE OpenMP::OpenMP_CXX
)


add_library(comm SHARED
  src/stats.cpp
  src/async_comm-impl.cpp
  src/async_comm.cpp
  src/rma_comm.cpp
  src/rma_comm-impl.cpp
  src/particle_async_comm.cpp
  src/particle_rma_comm.cpp
  src/state_comm.cpp
)
target_include_directories(comm PUBLIC include/comm)
set_target_properties(comm PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(comm
  PUBLIC types MPI::MPI_C
)

add_library(mcmpi SHARED
  src/worker.cpp
  src/worker_sync.cpp
  src/worker_async.cpp
  src/worker_rma.cpp
)
target_compile_features(mcmpi PUBLIC cxx_lambda_init_captures)
target_include_directories(mcmpi PUBLIC include/mcmpi)
set_target_properties(mcmpi PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(mcmpi
  PUBLIC types layer timer comm MPI::MPI_C
  PRIVATE yaml
)

########## Scripts ##########
configure_file(scripts/save_run save_run COPYONLY)



########## Programs ##########
add_executable(main src/main.cpp)
set_target_properties(main PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(main PRIVATE mcmpi)


########## Tests ##########
add_executable(test_comm src_test/test_comm.cpp)
set_target_properties(test_comm PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_comm PRIVATE
  comm
  MPI::MPI_C
)


add_executable(test_state_comm src_test/test_state_comm.cpp)
set_target_properties(test_state_comm PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_state_comm PRIVATE
  comm
  MPI::MPI_C
)


add_executable(test_async_serialization
  src_test/test_async_serialization.cpp
)
set_target_properties(test_async_serialization PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_async_serialization PRIVATE
  comm
  MPI::MPI_C
)


add_executable(test_rma_serialization
  src_test/test_rma_serialization.cpp
)
set_target_properties(test_rma_serialization PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_rma_serialization PRIVATE
  comm
  MPI::MPI_C
)


add_executable(test_layer src_test/test_layer.cpp)
set_target_properties(test_layer PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_layer
  layer
)


add_executable(test_layer_perf src_test/test_layer_perf.cpp)
set_target_properties(test_layer_perf PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_layer_perf
  layer
)
