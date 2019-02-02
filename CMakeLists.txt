cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(mc-mpi CXX)

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

if(DEFINED ENV{CXX})
  message(STATUS "${Yellow}CXX=" $ENV{CXX} ". Make sure to use an MPI compatible compiler${ColourReset}. On unix system through e.g. 'export CXX=mpicxx'.")
else(DEFINED ENV{CXX})
  message(FATAL_ERROR "Please define the environment variable for the mpi capable cxx compiler. On unix system through e.g. 'export CXX=mpicxx'.")
endif(DEFINED ENV{CXX})


########## Required Packages ##########
find_package(OpenMP REQUIRED)

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


add_library(types INTERFACE)
target_include_directories(types INTERFACE include/types)
target_compile_features(types INTERFACE cxx_std_11)


add_library(layer SHARED src/layer.cpp src/random.cpp)
target_include_directories(layer PUBLIC include/layer)
target_include_directories(layer PRIVATE include/layer/private)
set_target_properties(layer PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(layer PUBLIC types)
target_link_libraries(layer PRIVATE
  OpenMP::OpenMP_CXX
)


add_library(comm SHARED
  src/stats.cpp
  src/async_comm-impl.cpp
  src/async_comm.cpp
  src/particle_comm.cpp
  src/state_comm.cpp
)
target_include_directories(comm PUBLIC include/comm)
set_target_properties(comm PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(comm PUBLIC types)


add_library(mcmpi SHARED src/worker.cpp)
target_include_directories(mcmpi PUBLIC include/mcmpi)
set_target_properties(mcmpi PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(mcmpi PUBLIC
  types
  layer
  timer
  comm
)
target_link_libraries(mcmpi PRIVATE
  yaml
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
target_link_libraries(main
  mcmpi
)



########## Tests ##########
add_executable(test_comm src_test/test_comm.cpp)
set_target_properties(test_comm PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_comm
  comm
)

add_executable(test_state_comm src_test/test_state_comm.cpp)
set_target_properties(test_state_comm PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_state_comm
  comm
)


add_executable(test_serialization src_test/test_serialization.cpp)
set_target_properties(test_serialization PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_serialization
  comm
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

add_executable(test_layer_omp src_test/test_layer_omp.cpp)
set_target_properties(test_layer_omp PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_layer_omp
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

add_executable(test_layer_omp_perf src_test/test_layer_omp_perf.cpp)
set_target_properties(test_layer_omp_perf PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_layer_omp_perf
  layer
)

add_executable(test_comm_perf src_test/test_comm_perf.cpp)
set_target_properties(test_comm_perf PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_comm_perf
  comm
  timer
  layer
)

add_executable(test_mpi_sync src_test/test_mpi_sync.cpp)
set_target_properties(test_mpi_sync PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_mpi_sync
  timer
  layer
)


add_executable(test_mpi_sync_event src_test/test_mpi_sync_event.cpp)
set_target_properties(test_mpi_sync_event PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_mpi_sync_event
  timer
  layer
)

add_executable(test_recv_perf src_test/test_recv_perf.cpp)
set_target_properties(test_recv_perf PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_recv_perf
  comm
  layer
)
