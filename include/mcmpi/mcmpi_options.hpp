#ifndef MCMPI_OPTIONS_HPP
#define MCMPI_OPTIONS_HPP

#include "types.hpp"

/*!
 * \struct MCMPIOptions
 *
 * \brief Options for the Monte Carlo Simulation
 */
typedef struct mcmpi_options_tag
{
    /*!
   * \enum Tag
   *
   * \brief MPI send/recv tags
   */
    enum Tag : int
    {
        Particle = 0,
        State,
        STATE_COUNT
    };

    /* Simulation properties */
    int world_size;         /** number of processes/layers in the simulation */
    int nb_cells_per_layer; /** number of cells in each layer */
    real_t x_min, x_max,
        x_ini; /** x_min, x_max and x_ini of the global simulation */

    real_t particle_min_weight; /** If weight < ... the particle is disabled */
    size_t nb_particles;        /** total number of particles in the simulation */

    /* Performance properties */
    size_t buffer_size;         /** buffer size in bytes of the async_comm send
                              buffer. Higher is better, e.g. 1024*1024 */
    double cycle_time;          /** Each worker tries to complete one cycle in cycle time
                         seconds */
    int nb_particles_per_cycle; /** Number particles simulated per cycle */
    int nthread;                /** Number of OMP threads. Set -1 for automatic */

    double statistics_cycle_time; /** Each worker will dump statistics after this
                                   cycle time*/
} MCMPIOptions;

#endif