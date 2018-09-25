#ifndef WORKER_HPP
#define WORKER_HPP

#include "async_comm.hpp"
#include "layer.hpp"
#include "random.hpp"
#include "types.hpp"

#define MCMPI_PARTICLE_TAG 1
#define MCMPI_NB_DISABLED_TAG 2
#define MCMPI_FINISHED_TAG 3
#define MCMPI_WAIT_MS 1
#define MCMPI_NB_STEPS_PER_CYCLE 1000

/*!
 * \struct McOptions
 *
 * \brief Options for the Monte Carlo Simulation
 */
typedef struct mc_options_tag {
  int world_size; /** number of processes/layers in the simulation */
  real_t x_min, x_max,
      x_ini; /** x_min, x_max and x_ini of the global simulation */

  std::size_t buffer_size;  /** buffer size in bytes of the async_comm send
                               buffer. Higher is better, e.g. 1024*1024 */
  std::size_t nb_particles; /** total number of particles in the simulation */
} McOptions;

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
   * \param[in] McOptions global options
   */
  Worker(int world_rank, const McOptions &McOptions);

  /*!
   * \function spin
   *
   * \brief Main function. Call spin() once in the main. The function will exit
   * once the simulation is over.
   */
  void spin();

  /*!
   * \function weight_absorbed
   *
   * \brief Calculates the total absorbed weight in the simulation (at this
   * point)
   *
   * \return real_t total absorbed weight
   */
  real_t weight_absorbed();

private:
  const int world_rank;
  const McOptions options;
  UnifDist dist;
  Layer layer;
  AsyncComm<Particle> particle_comm;
  std::vector<Particle> particles_left, particles_right;
  AsyncComm<int> event_comm;
};
#endif