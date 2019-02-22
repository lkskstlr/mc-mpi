#ifndef WORKER_ASYNC_HPP
#define WORKER_ASYNC_HPP

#include "particle_comm.hpp"
#include "state_comm.hpp"
#include "worker.hpp"

class WorkerAsync : public Worker
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
    WorkerAsync(int world_rank, const MCMPIOptions &options);

    /*!
   * \function spin
   *
   * \brief Main function. Call spin() once in the main. The function will exit
   * once the simulation is over.
   */
    void spin() override;

    ParticleComm particle_comm;
    StateComm state_comm;
};

#endif