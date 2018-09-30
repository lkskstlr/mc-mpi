#ifndef WORKER_HPP
#define WORKER_HPP

#include "layer.hpp"
#include "particle_comm.hpp"
#include "random.hpp"
#include "timer.hpp"
#include "types.hpp"

#define MCMPI_PARTICLE_TAG 1
#define MCMPI_NB_DISABLED_TAG 2
#define MCMPI_FINISHED_TAG 3

#ifndef NDEBUG
#define MCMPI_WAIT_MS 1000
#define MCMPI_NB_STEPS_PER_CYCLE 1
#else
#define MCMPI_WAIT_MS 1
#define MCMPI_NB_STEPS_PER_CYCLE 10000
#endif

/*!
 * \struct McOptions
 *
 * \brief Options for the Monte Carlo Simulation
 */
typedef struct mc_options_tag {
  int world_size;         /** number of processes/layers in the simulation */
  int nb_cells_per_layer; /** number of cells in each layer */
  real_t x_min, x_max,
      x_ini; /** x_min, x_max and x_ini of the global simulation */

  real_t particle_min_weight; /** If weight < ... the particle is disabled */

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
  const McOptions options;
  UnifDist dist;
  Layer layer;
  ParticleComm particle_comm;
  std::vector<Particle> particles_left, particles_right, particles_disabled;
  AsyncComm<int> event_comm;

  Timer timer;
};
#endif