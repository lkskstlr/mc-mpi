#include "mcmpi_options.hpp"
#include "worker.hpp"
#include "worker_async.hpp"
#include "worker_rma.hpp"
#include "worker_sync.hpp"
#include <iostream>
#include <mpi.h>
#include <set>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

std::pair<std::string, std::string> parse_input(int argc, char **argv,
                                                int world_rank)
{
  std::set<std::string> comm_modes = {"sync", "async", "rma"};

  if (argc != 3 || comm_modes.find(argv[2]) == comm_modes.end())
  {
    if (world_rank == 0)
    {
      fprintf(stderr,
              "Usage: mpirun -n nb_layers %s config_file_path comm_mode\n",
              argv[0]);
      std::string comm_modes_str;
      for (auto const &str : comm_modes)
      {
        comm_modes_str.append(str);
        comm_modes_str.append(", ");
      }
      if (comm_modes_str.size() >= 2)
      {
        comm_modes_str.pop_back();
        comm_modes_str.pop_back();
      }

      fprintf(stderr, "comm_mode must be in {%s}\n", comm_modes_str.c_str());
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  std::string filepath(argv[1]);
  std::string comm_mode(argv[2]);
  return std::make_pair(filepath, comm_mode);
}

int main(int argc, char **argv)
{
  // -- MPI Setup --
  MPI_Init(&argc, &argv);
  int world_rank, world_size;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // parse input
  std::pair<std::string, std::string> pair =
      parse_input(argc, argv, world_rank);

  MCMPIOptions options = options_from_config(pair.first, world_size);

  Worker *worker = NULL;
  if (pair.second.compare("sync") == 0)
  {
    worker = new WorkerSync(world_rank, options);
  }
  else if (pair.second.compare("async") == 0)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(10000 * world_rank);
    worker = new WorkerAsync(world_rank, options);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  else if (pair.second.compare("rma") == 0)
  {
    worker = new WorkerRma(world_rank, options);
  }

  if (worker == NULL && world_rank == 0)
  {
    fprintf(stderr, "No valid worker.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  double starttime = MPI_Wtime();
  worker->spin();
  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0)
    printf("%lf\n", MPI_Wtime() - starttime);
  worker->dump();

  MPI_Finalize();
  return 0;
}
