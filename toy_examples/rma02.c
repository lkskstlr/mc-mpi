#define _BSD_SOURCE

#include <stdio.h>
#include <mpi.h>
#include <stdint.h>
#include <unistd.h>
#include <inttypes.h>

#define NBUFFER 4

int main(int argc, char **argv)
{
    int world_rank, world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int buffer_size = 16;
    MPI_Win buffer_wins[NBUFFER];
    int *buffers[NBUFFER] = {0};

    if (world_rank == 0)
    {
        for (int i = 0; i < NBUFFER; i++)
        {
            MPI_Win_allocate(buffer_size * sizeof(int),
                             sizeof(int), MPI_INFO_NULL,
                             MPI_COMM_WORLD,
                             &(buffers[i]),
                             &(buffer_wins[i]));
        }
    }
    else
    {
        for (int i = 0; i < NBUFFER; i++)
        {
            MPI_Win_allocate(0,
                             sizeof(int),
                             MPI_INFO_NULL,
                             MPI_COMM_WORLD,
                             &(buffers[i]),
                             &(buffer_wins[i]));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(10000 * world_rank);
    printf("%d/%d buffers = [", world_rank, world_size);
    for (int i = 0; i < NBUFFER; i++)
        printf("%p ", buffers[i]);
    printf("]\n");
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}