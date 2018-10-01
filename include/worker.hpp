#ifndef WORKER_HPP
#define WORKER_HPP

#include "layer.hpp"
#include "mcmpi_options.hpp"
#include "particle.hpp"
#include "particle_comm.hpp"
#include "random.hpp"
#include "timer.hpp"
#include "types.hpp"

/*!
 * \class Worker
 *
 * \brief Main class for the simulation. Each process has one worker on which
 * after setup spin() is called. All the control flow and data is handeled by
 * the worker.
 */
class Worker {
public:
  /*!
   * \function Worker
   *
   * \brief Constructor
   *
   * \param[in] world_rank mpi world_rank of the processor
   * \param[in] MCMPIOptions global options
   */
  Worker(int world_rank, const MCMPIOptions &options);

  /*!
   * \function spin
   *
   * \brief Main function. Call spin() once in the main. The function will exit
   * once the simulation is over.
   */
  void spin();

  /*!
   * \function weights_absorbed
   *
   * \brief Calculates the total absorbed weight in the simulation (at this
   * point)
   *
   * \return real_t total absorbed weight
   */
  std::vector<real_t> weights_absorbed();

  // private:
  const int world_rank;
  const MCMPIOptions options;
  UnifDist dist;
  Layer layer;
  ParticleComm particle_comm;
  std::vector<Particle> particles_left, particles_right, particles_disabled;
  AsyncComm<int> event_comm;

  Timer timer;
};
#endif