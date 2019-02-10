#define _BSD_SOURCE

#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    int world_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int color = world_rank / 2;
    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &new_comm);

    int new_rank, new_size;
    MPI_Comm_size(new_comm, &new_size);
    MPI_Comm_rank(new_comm, &new_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(10000 * world_rank);
    printf("WORLD RANK/SIZE: %d/%d \t NEW RANK/SIZE: %d/%d\n",
           world_rank, world_size, new_rank, new_size);
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 5 || world_rank == 6)
    {
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);

        MPI_Group new_group;
        int *ranks = (int *)malloc(sizeof(int) * 2);
        ranks[0] = 6;
        ranks[1] = 5;
        MPI_Group_incl(world_group, 2, ranks, &new_group);

        MPI_Comm small_comm;
        MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &small_comm);

        int small_rank, small_size;

        MPI_Comm_size(small_comm, &small_size);
        MPI_Comm_rank(small_comm, &small_rank);

        usleep(10000 * small_rank);
        printf("WORLD RANK/SIZE: %d/%d \t SMALL RANK/SIZE: %d/%d\n",
               world_rank, world_size, small_rank, small_size);
    }
    MPI_Finalize();
    return 0;
}