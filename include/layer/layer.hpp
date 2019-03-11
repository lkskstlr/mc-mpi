#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include "particle.hpp"
#include "types.hpp"
#include <vector>

/*!
 * \class Layer
 *
 * \brief Stores one layer and the particles in it. Also does the
 * computations on the particles
 */
class Layer
{
public:
  /*!
   * \function Layer
   *
   * \brief Constructor
   *
   * \param[in] x_min min of layer, e.g. 0.2
   * \param[in] x_max max of layer, e.g. 0.4
   * \param[in] m number of cells in this layer
   */
  Layer(real_t x_min, real_t x_max, int index_start, int m,
        real_t particle_min_weight);

  /*!
   * \function create_particles
   *
   * \brief Creates the particles in this layer
   *
   * \param[in] x_ini Initial position of all particles (must be in (x_min,
   * x_max))
   * \param[in] wmc weight monte carlo
   * \param[in] n number of partciles to create
   * \param[in] seed used
   */
  void create_particles(real_t x_ini, real_t wmc, int n, seed_t seed);

  /*!
   * \function simulate
   *
   * \brief Simulates nb_particles until they leave the domain or become
   * inactive
   *
   * \param[in] nb_particles Number of particles to simulampi_decompose_domainte.
   * Use -1 to simulate until all particles have left or became inactive.
   * \param[in] nthread Number of OpenMP threads to use, optional (default =
   * -1). If -1 is selected the OpenMP runtime decides, e.g. through
   * OMP_NUM_THREADS.
   *
   * \return void
   */
  void simulate(int nb_particles, int nthread = -1, bool use_gpu = false);

  /*!
   * \function dump_WA
   *
   * \brief Dump weights file to disk
   *
   * \return void
   */
  void dump_WA();

  /*!
   * \function nb_active
   *
   * \brief Returns number of particles active in this layer
   *
   * \return int number of particles active in this layer
   */
  int nb_active() const;

  void create_particles(int n);
  int simulate_particle(Particle &particle,
                        std::vector<real_t> &weights_absorbed_local);
  int particle_step(Particle &particle,
                    std::vector<real_t> &weights_absorbed_local);
  void simulate_helper(int nb_particles, int nthread, bool use_gpu);

  // -- Data --
  real_t x_min, x_max;

  int m;
  int index_start;
  std::vector<real_t> weights_absorbed;
  std::vector<Particle> particles;
  std::vector<Particle> particles_left;
  std::vector<Particle> particles_right;
  const real_t dx;
  int nb_disabled = 0;

  const bool left_border, right_border;

  // -- physical properties --
  std::vector<real_t> sigs; // = exp(-0.5*(x_min+x_max))
  std::vector<real_t> absorption_rates;
  const real_t particle_min_weight;

  seed_t seed;
  real_t x_ini, wmc;
  int nb_particles_create = 0;
};

Layer decompose_domain(real_t x_min, real_t x_max, real_t x_ini, int world_size,
                       int world_rank, int nb_cells, int nb_particles,
                       real_t particle_min_weight);
#endif
