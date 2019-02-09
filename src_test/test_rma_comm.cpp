#include <mpi.h>
#include <chrono>
#include <iostream>
#include <thread>
#include "rma_comm.hpp"

int main(int argc, char **argv)
{
    using std::cout;
    using std::endl;

    int world_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    cout << "Hi" << endl;

    RmaComm<int> comm;
    comm.init(100);
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
        std::vector<int> data = {1, 2, 3, 4};
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

        for (auto const &x : data)
        {
            cout << x << "\n";
        }
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}