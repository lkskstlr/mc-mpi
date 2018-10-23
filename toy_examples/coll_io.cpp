#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char const *argv[]) {
  MPI_Init(NULL, NULL);
  int world_size, world_rank;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, "out.txt", MPI_MODE_WRONLY | MPI_MODE_CREATE,
                MPI_INFO_NULL, &file);

  char buf[6] = "hello";
  MPI_File_write_at(file, world_rank * 5, buf, 5, MPI_CHAR, MPI_STATUS_IGNORE);

  if (world_rank == 0) {
    usleep(1000000);
  }

  MPI_File_close(&file);
  printf("rank = %d\n", world_rank);

  MPI_Finalize();
  return 0;
}