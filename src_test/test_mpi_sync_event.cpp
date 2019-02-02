#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>
#include "layer.hpp"

int main(int argc, char *argv[])
{
    int world_size, world_rank;

    //MPI initialization and basic function calls
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    srand(30061994 + 17 * world_rank);

    int disabled_own = 0, disabled_all, nb_particles = world_size * 2;
    int i = 0;
    while (true)
    {
        if ((disabled_own < nb_particles / world_size) && (rand() > RAND_MAX / 2))
            disabled_own++;

        MPI_Allreduce(&disabled_own, &disabled_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(100000 * world_rank);
        if (world_rank == 0)
            printf("\n\n========== %2d ==========\n", i);
        printf("rank = %2d, disabled_own = %2d, disabled_all = %3d\n", world_rank, disabled_own, disabled_all);
        MPI_Barrier(MPI_COMM_WORLD);
        if (disabled_all == nb_particles)
            break;
        usleep(1000000);
        i++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(100000 * world_rank);
    if (world_rank == 0)
        printf("\n\n========== %2d ==========\n", i + 1);
    printf("rank = %2d out\n", world_rank);
    MPI_Finalize();

    return 0;
}
