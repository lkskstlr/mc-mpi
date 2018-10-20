#ifndef WORKER_HPP
#define WORKER_HPP

#include "layer.hpp"
#include "mcmpi_options.hpp"
#include "particle.hpp"
#include "particle_comm.hpp"
#include "timer.hpp"
#include "types.hpp"
#include <string>

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
   * \function dump
   *
   * \brief Creates folder out and dumps times.csv and config.yaml there.
   */
  void dump();

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
  Layer layer;
  ParticleComm particle_comm;
  std::vector<Particle> particles_left, particles_right, particles_disabled;
  AsyncComm<int> event_comm;

  Timer timer;
  std::vector<Timer::State> timer_states;
  MPI_Datatype timer_state_mpi_t;

private:
  void gather_times(int *total_len, int **displs, Timer::State **states);
  void gather_weights_absorbed(int *total_len, int **displs, real_t **weights);
  void mkdir_out();
  void dump_config();
  void dump_times(int total_len, int const *displs, Timer::State const *states);
  void dump_recv_times();
  void dump_weights_absorbed(int total_len, int const *displs,
                             real_t const *weights);

  unsigned long unix_timestamp_start;

  std::vector<int> nb_recv;
  std::vector<double> dt_recv;

  int nb_mpi_send = 0;
  int nb_mpi_recv = 0;
};

Worker worker_from_config(std::string filepath, int world_size, int world_rank);

#endif