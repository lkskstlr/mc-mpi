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

OPTION(DEFINE_NDEBUG "NDEBUG is defined" ON) # Enabled by default
IF(DEFINE_NDEBUG)
    ADD_DEFINITIONS(-DNDEBUG)
ENDIF(DEFINE_NDEBUG)

add_library(libmc-mpi SHARED
    src/random.cpp
    src/layer.cpp
    src/worker.cpp
    src/timer.cpp
    src/async_comm-impl.cpp
    src/particle_comm.cpp
)
target_include_directories(libmc-mpi PUBLIC include)
set_target_properties(libmc-mpi PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Main
add_executable(main src/main.cpp)
target_include_directories(main PRIVATE include)
set_target_properties(main PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_link_libraries(main
    libmc-mpi
)

# Tests
add_executable(test_async_comm src_test/test_async_comm.cpp)
target_include_directories(test_async_comm PRIVATE include)
set_target_properties(test_async_comm PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_link_libraries(test_async_comm
    libmc-mpi
)

add_executable(test_serialization src_test/test_serialization.cpp)
target_include_directories(test_serialization PRIVATE include)
set_target_properties(test_serialization PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_link_libraries(test_serialization
    libmc-mpi
)