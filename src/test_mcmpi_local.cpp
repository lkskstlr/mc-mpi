#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

// From: http://www.cs.yorku.ca/~oz/hash.html
unsigned long hash(unsigned char *str)
{
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

void MCMPI_Local(int *const local_size, int *const local_rank)
{
    int world_rank;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char hostname[2048] = {0};
    gethostname(hostname, 2048);
    unsigned long my_hash = hash((unsigned char *)hostname);
    unsigned long *hashes = (unsigned long *)malloc(sizeof(unsigned long) * world_size);

    MPI_Allgather(&my_hash, 1, MPI_UNSIGNED_LONG, hashes, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    *local_size = 0;
    *local_rank = 0;

    for (int j = 0; j < world_size; j++)
    {
        if (hashes[j] == my_hash)
        {
            (*local_size)++;
            if (j < world_rank)
                (*local_rank)++;
        }
    }

    free(hashes);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int world_rank;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int local_size, local_rank;
    MCMPI_Local(&local_size, &local_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(100000 * world_rank);
    printf("world:%d/%d  ->  local:%d/%d\n", world_rank, world_size, local_rank, local_size);

    MPI_Finalize();
}