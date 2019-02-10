#define _BSD_SOURCE

#include <mpi.h>
#include <chrono>
#include <iostream>
#include <thread>
#include "rma_comm.hpp"
#include <unistd.h>

int main(int argc, char **argv)
{
    using std::cout;
    using std::endl;

    int world_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char hostname[100];
    gethostname(hostname, 100);
    cout << "Hi from " << hostname << " with rank" << world_rank << endl;

    RmaComm<int> comm;
    comm.init(65000);
    comm.mpi_data_t = MPI_INT;

    if (world_rank == 0)
        comm.subscribe(1);
    else
        comm.advertise(0);

    if (world_rank == 0)
        comm.print();

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 1)
    {
        std::vector<int> data(5000, 17);
        comm.send(data, 0);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0)
        comm.print();

    if (world_rank == 0)
    {
        std::vector<int> data;
        comm.recv(data, 1);
        comm.print();
    }

    MPI_Finalize();
    return 0;
}