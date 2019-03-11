#ifndef WORKER_RMA_HPP
#define WORKER_RMA_HPP

#include "particle_rma_comm.hpp"
#include "state_comm.hpp"
#include "worker.hpp"

class WorkerRma : public Worker
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
  WorkerRma(int world_rank, MCMPIOptions &options);

  /*!
   * \function spin
   *
   * \brief Main function. Call spin() once in the main. The function will exit
   * once the simulation is over.
   */
  void spin() override;

  ParticleRmaComm particle_comm;
  StateComm state_comm;
};

#endif
