#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUF_SIZE 100000

void stress() {
  volatile int a = 31;
  const int k = 17;
  for (int i = 0; i < 100000000; ++i) {
    a = (a * a + 3) % k;
  }
}

int main(int argc, char const *argv[]) {
  MPI_Init(NULL, NULL);
  int world_size, world_rank;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0 && world_size != 2) {
    fprintf(stderr, "This example requires two MPI processes.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  /* Non Blocking Send */
  int buf[BUF_SIZE] = {0};

  MPI_Barrier(MPI_COMM_WORLD);
  double time;
  if (world_rank == 1) {
    time = MPI_Wtime();
    MPI_Request request;
    MPI_Isend(buf, BUF_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
    stress();
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  } else {
    usleep(100);
    time = MPI_Wtime();
    MPI_Recv(buf, BUF_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  time = MPI_Wtime() - time;
  printf("rank = %d, time = %f sec\n", world_rank, time);
  MPI_Barrier(MPI_COMM_WORLD);

  usleep(100);
  if (world_rank == 0) {
    printf("---\n");
  }

  /* Non Blocking Receive */
  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 1) {
    usleep(100);
    time = MPI_Wtime();
    MPI_Send(buf, BUF_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD);
  } else {
    time = MPI_Wtime();
    MPI_Request request;
    MPI_Irecv(buf, BUF_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    stress();
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }
  time = MPI_Wtime() - time;
  printf("rank = %d, time = %f sec\n", world_rank, time);
  MPI_Barrier(MPI_COMM_WORLD);

  usleep(100);
  if (world_rank == 0) {
    printf("---\n");
  }

  /* Both NonBlocking */
  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 1) {
    time = MPI_Wtime();
    MPI_Request request;
    MPI_Isend(buf, BUF_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
    stress();
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  } else {
    usleep(100);
    time = MPI_Wtime();
    MPI_Request request;
    MPI_Irecv(buf, BUF_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }
  time = MPI_Wtime() - time;
  printf("rank = %d, time = %f sec\n", world_rank, time);
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}