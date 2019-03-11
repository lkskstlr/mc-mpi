#ifndef WORKER_HPP
#define WORKER_HPP

#include "layer.hpp"
#include "mcmpi_options.hpp"
#include "particle.hpp"
#include "state_comm.hpp"
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
class Worker
{
public:
  typedef struct env_tag
  {
    int node_size;
    int nb_gpu;
  } Env;
  /*!
   * \function Worker
   *
   * \brief Constructor
   *
   * \param[in] world_rank mpi world_rank of the processor
   * \param[in] MCMPIOptions global options
   */
  Worker(int world_rank, MCMPIOptions &options);

  /*!
   * \function spin
   *
   * \brief Main function. Call spin() once in the main. The function will exit
   * once the simulation is over.
   */
  virtual void spin() = 0;

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
  bool use_gpu;

  std::vector<real_t> weights_absorbed();

  // private:
  const int world_rank;
  MCMPIOptions options;
  Layer layer;

  Timer timer;
  std::vector<Timer::State> timer_states;
  std::vector<Stats::State> stats_states;
  std::vector<int> cycle_states;

private:
  void write_file(char *filename);
  void gather_weights_absorbed(int *total_len, int **displs, real_t **weights);
  void mkdir_out();
  void dump_config();
  void dump_weights_absorbed(int total_len, int const *displs,
                             real_t const *weights);

  std::string foldername;
};

MCMPIOptions options_from_config(std::string filepath, int world_size);

#endif
