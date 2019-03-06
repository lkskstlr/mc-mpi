cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(mc-mpi C CXX)

cmake_policy(SET CMP0074 NEW)

option(CUDA_ENABLED "Build and test with CUDA support." OFF)


if(CUDA_ENABLED)
  # https://devblogs.nvidia.com/building-cuda-applications-cmake/
  message(STATUS "CUDA enabled")
  # Finding suitable compiler for nvcc -ccbin
  if(NOT DEFINED ENV{CUDAHOSTCXX})
    if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.8.5)
      set(CUDAGCC ${CMAKE_CXX_COMPILER})
    elseif()
      find_program(CUDAGCC g++-4.8)
    endif()
    if(NOT CUDAGCC)
      message(WARNING "Environment variable CUDAHOSTCXX was not set and g++-4.8 was not found. Probably nvcc will not work with standard g++.")
    else()
      message(STATUS "Environment variable CUDAHOSTCXX was not set. Using ${CUDAGCC} (default)")
      set(ENV{CUDAHOSTCXX} "${CUDAGCC}")
    endif()
  endif()
  enable_language(CUDA)
else()
  message(STATUS "CUDA disabled. Set CUDA_ENABLED option.")
endif()


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
  message(STATUS "Setting Build Type to Release (default)")
endif(NOT CMAKE_BUILD_TYPE)

########## Global Compile Flags ##########
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -pedantic")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -pedantic")
  else()
    message(STATUS "Old compiler version. Compiling without -Wall -Werror -pedantic")
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -pedantic")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -pedantic")
endif()

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


add_library(random SHARED src/random.cpp)
target_include_directories(random PUBLIC include/random)
target_link_libraries(random
  PUBLIC types
)


add_library(layer SHARED src/layer.cpp)
target_include_directories(layer PUBLIC include/layer)
set_target_properties(layer PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(layer
  PUBLIC types
  PRIVATE random OpenMP::OpenMP_CXX
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
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
    target_compile_features(mcmpi PUBLIC cxx_lambda_init_captures)
  elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8.4)
    #This actually work
  else()
    message(WARNING "The compiler version is ${CMAKE_CXX_COMPILER_VERSION} and it might not support init captures.")
  endif()
else()
  target_compile_features(mcmpi PUBLIC cxx_lambda_init_captures)
endif()
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

if(CUDA_ENABLED)
  add_library(gpu_errcheck INTERFACE)
  target_include_directories(gpu_errcheck INTERFACE include/gpu_errcheck)


  add_library(curandom STATIC src/curandom.cu)
  target_include_directories(curandom PUBLIC include/curandom)
  target_compile_features(curandom PRIVATE cxx_std_11)
  set_target_properties(curandom PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(curandom
    PUBLIC types
  )


  add_library(culayer STATIC src/culayer.cu src/culayer_kernel.cu)
  target_include_directories(culayer PUBLIC include/culayer)
  target_compile_features(culayer PRIVATE cxx_std_11)
  set_target_properties(culayer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(culayer
    PUBLIC types gpu_errcheck
  )
endif(CUDA_ENABLED)

########## Scripts ##########
configure_file(scripts/save_run save_run COPYONLY)
configure_file(scripts/run.batch run.batch COPYONLY)
configure_file(scripts/run.py run.py COPYONLY)



########## Programs ##########
add_executable(main src/main.cpp)
set_target_properties(main PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(main PRIVATE mcmpi)


########## Test Executables ##########
add_executable(test_comm src/test_comm.cpp)
set_target_properties(test_comm PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_comm PRIVATE
  comm
  MPI::MPI_C
)


add_executable(test_state_comm src/test_state_comm.cpp)
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
  src/test_async_serialization.cpp
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
  src/test_rma_serialization.cpp
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


add_executable(test_layer src/test_layer.cpp)
set_target_properties(test_layer PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_layer
  layer
)


add_executable(test_layer_perf src/test_layer_perf.cpp)
set_target_properties(test_layer_perf PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(test_layer_perf
  layer
)

if(CUDA_ENABLED)
  add_executable(test_gpu_errcheck src/test_gpu_errcheck.cu)
  target_link_libraries(test_gpu_errcheck
    PRIVATE gpu_errcheck
  )

  add_executable(test_curandom src/test_curandom.cu)
  target_compile_features(test_curandom PRIVATE cxx_std_11)
  set_target_properties(test_curandom PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(test_curandom
    PRIVATE curandom random types
  )

  add_executable(test_culayer src/test_culayer.cu)
  target_compile_features(test_culayer PRIVATE cxx_std_11)
  set_target_properties(test_culayer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(test_culayer
    PRIVATE culayer types layer gpu_errcheck
  )

  add_executable(test_cuatomic_add src/test_cuatomic_add.cu)
  target_compile_features(test_cuatomic_add PRIVATE cxx_std_11)
  set_target_properties(test_cuatomic_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(test_cuatomic_add
    PRIVATE gpu_errcheck
  )
endif(CUDA_ENABLED)

########## CTest ##########
enable_testing()

add_test(NAME TestLayer COMMAND test_layer)

add_test(NAME TestComm COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_comm ${MPIEXEC_POSTFLAGS})
set_tests_properties(TestComm PROPERTIES TIMEOUT 2)

add_test(NAME TestStateComm COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_state_comm ${MPIEXEC_POSTFLAGS})
set_tests_properties(TestStateComm PROPERTIES TIMEOUT 2)

add_test(NAME TestAsyncSerialization COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_async_serialization ${MPIEXEC_POSTFLAGS})
set_tests_properties(TestAsyncSerialization PROPERTIES TIMEOUT 2)

add_test(NAME TestRmaSerialization COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_rma_serialization ${MPIEXEC_POSTFLAGS})
set_tests_properties(TestRmaSerialization PROPERTIES TIMEOUT 2)

if(CUDA_ENABLED)
  add_test(NAME TestGpuErrcheck COMMAND ./test_gpu_errcheck)
  set_tests_properties(TestGpuErrcheck PROPERTIES WILL_FAIL ON)

  add_test(NAME TestCudaRandom COMMAND ./test_curandom)
  set_tests_properties(TestRmaSerialization PROPERTIES TIMEOUT 20)

  add_test(NAME TestCuLayer COMMAND ./test_culayer)
  set_tests_properties(TestCuLayer PROPERTIES TIMEOUT 20)
endif(CUDA_ENABLED)
