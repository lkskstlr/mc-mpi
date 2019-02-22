#define _BSD_SOURCE

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
  int world_rank, world_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    if (argc != 2) {
      fprintf(stderr,
              "Must be called like: %s nb_int.\nWhere nb_int is the number of "
              "integers to be transfered.\n",
              argv[0]);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (world_size != 2) {
      fprintf(stderr, "Must be called with n=2 MPI processes.\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }
  int nb_int = atoi(argv[1]);

  MPI_Win win;
  int *ptr = NULL;

  if (world_rank == 0)
    MPI_Win_allocate(sizeof(int) * nb_int, sizeof(int), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &ptr, &win);
  else
    MPI_Win_allocate(0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &ptr, &win);

  MPI_Barrier(MPI_COMM_WORLD);
  printf("%d/%d ptr = %p\n", world_rank, world_size, ptr);

  if (world_rank == 1) {
    ptr = (int *)malloc(sizeof(int) * nb_int);
    for (int i = 0; i < nb_int; i++) ptr[i] = (i * i * i) % (31);

    double t0, t1, t2, t3;
    t0 = MPI_Wtime();
    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
    t1 = MPI_Wtime();
    MPI_Put(ptr, nb_int, MPI_INT, 0, 0, nb_int, MPI_INT, win);
    t2 = MPI_Wtime();
    MPI_Win_unlock(0, win);
    t3 = MPI_Wtime();

    printf(
        "Timing:\n\tLock  : %f ms\n\tPut   : %f ms\n\tUnlock: %f ms\n\tTotal : "
        "%f ms\n",
        1000.0 * (t1 - t0), 1000.0 * (t2 - t1), 1000.0 * (t3 - t2),
        1000.0 * (t3 - t0));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  usleep(10000 * (1 + world_rank));
  int sum = 0;
  for (int i = 0; i < nb_int; i++) sum += ptr[i];

  printf("%d/%d sum = %d\n", world_rank, world_size, sum);

  MPI_Barrier(MPI_COMM_WORLD);
  usleep(10000 * (1 + world_rank));
  if (world_rank == 0) {
    void *win_base;
    int *win_size, *win_disp_unit, flag;

    MPI_Win_get_attr(win, MPI_WIN_BASE, &win_base, &flag);
    printf("win_base      = %p,\tflag = %d\n", win_base, flag);

    int ret = MPI_Win_get_attr(win, MPI_WIN_SIZE, &win_size, &flag);
    if (ret != MPI_SUCCESS) {
      printf("WHUUUPPPS NO SUCCESS\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    printf("win_size      = %d,\t\tflag = %d\n", *win_size, flag);

    MPI_Win_get_attr(win, MPI_WIN_DISP_UNIT, &win_disp_unit, &flag);
    printf("win_disp_unit = %d,\t\tflag = %d\n", *win_disp_unit, flag);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  usleep(10000 * (1 + world_rank));
  if (world_rank == 1) {
    void *win_base;
    int *win_size, *win_disp_unit, flag;

    MPI_Win_get_attr(win, MPI_WIN_BASE, &win_base, &flag);
    printf("win_base      = %p,\tflag = %d\n", win_base, flag);

    int ret = MPI_Win_get_attr(win, MPI_WIN_SIZE, &win_size, &flag);
    if (ret != MPI_SUCCESS) {
      printf("WHUUUPPPS NO SUCCESS\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    printf("win_size      = %d,\t\tflag = %d\n", *win_size, flag);

    MPI_Win_get_attr(win, MPI_WIN_DISP_UNIT, &win_disp_unit, &flag);
    printf("win_disp_unit = %d,\t\tflag = %d\n", *win_disp_unit, flag);
  }

  MPI_Finalize();
  return 0;
}