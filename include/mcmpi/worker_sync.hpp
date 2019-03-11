#ifndef WORKER_SYNC_HPP
#define WORKER_SYNC_HPP

#include "worker.hpp"

class WorkerSync : public Worker
{
public:
  /*!
   * \function Worker
   *
   * \brief Constructor
   *
   * \param[in] world_rank mpi world_rank of the processor
   * \param[in] MCMPIOptions global options
   */
  WorkerSync(int world_rank, MCMPIOptions &options);

  /*!
   * \function spin
   *
   * \brief Main function. Call spin() once in the main. The function will exit
   * once the simulation is over.
   */
  void spin() override;

private:
  MPI_Datatype mpi_particle_type;
  const int particle_tag = 0;
};

#endif