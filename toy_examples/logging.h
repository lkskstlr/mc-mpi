#ifndef NDEBUG
// * ========== * //
//   Logging      //
// * ========== * //
#include <stdio.h>

// The RjGxXlShLtSvbJsnXESj should effectively prevent any naming clashes with
// other variables
static FILE *logging_file_RjGxXlShLtSvbJsnXESj;

#define MCMPI_DEBUG(fmt, ...)                                                  \
  fprintf(logging_file_RjGxXlShLtSvbJsnXESj, "%s:%d:%s(): " fmt, __FILE__,     \
          __LINE__, __func__, __VA_ARGS__);

#ifdef MCMPI_LOG_VERBOSE
#define MCMPI_VERBOSE(fmt, ...)                                                \
  fprintf(logging_file_RjGxXlShLtSvbJsnXESj, "%s:%d:%s(): " fmt, __FILE__,     \
          __LINE__, __func__, __VA_ARGS__);
#else
#define MCMPI_VERBOSE(fmt, ...)
#endif

// void init_logging_RjGxXlShLtSvbJsnXESj(int world_rank) {
//   char filename[100];
//   sprintf(filename, "logs/%010d.txt", world_rank);
//   logging_file_RjGxXlShLtSvbJsnXESj = fopen(filename, "wb");
// }

// void stop_logging_RjGxXlShLtSvbJsnXESj() {
//   fclose(logging_file_RjGxXlShLtSvbJsnXESj);
// }

#define MCMPI_DEBUG_INIT(world_rank)                                           \
  char filename_RjGxXlShLtSvbJsnXESj[100];                                     \
  sprintf(filename_RjGxXlShLtSvbJsnXESj, "logs/%010d.txt", world_rank);        \
  logging_file_RjGxXlShLtSvbJsnXESj =                                          \
      fopen(filename_RjGxXlShLtSvbJsnXESj, "wb");
#define MCMPI_DEBUG_STOP() fclose(logging_file_RjGxXlShLtSvbJsnXESj);

#else
// * ========== * //
//   NO Logging   //
// * ========== * //
#define MCMPI_DEBUG(fmt, ...)
#define MCMPI_VERBOSE(fmt, ...)
#define MCMPI_DEBUG_INIT(world_rank)
#define MCMPI_DEBUG_STOP()
#endif