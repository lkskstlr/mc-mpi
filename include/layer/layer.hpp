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
class Layer {
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
  void create_particles(real_t x_ini, real_t wmc, int n, seed_t *seed);

  /*!
   * \function simulate
   *
   * \brief Simulates nb_steps steps of partcile movement choosing basically
   * random particles from the domain. One particle will be simulated until it
   * leaves the domain, then another one etc..
   *
   * \param[in] nb_steps Number of simulation steps
   * \param[in] particles_left reference to the vector at wich the particles
   * that leave the layer to the left (x_min) will be appended
   * \param[in] particles_right reference to the vector at wich the particles
   * that leave the layer to the right (x_max) will be appended
   * \param[in] particles_disabled reference to the vector at wich the particles
   * that fall below the min weight will be appended
   */
  void simulate(int nb_steps, std::vector<Particle> &particles_left,
                std::vector<Particle> &particles_right,
                std::vector<Particle> &particles_disabled);

  // -- Data --
  const real_t x_min, x_max;
  std::vector<Particle> particles;
  std::vector<real_t> weights_absorbed;

  int particle_step(Particle &particle);

  // private:
  // -- physical properties --
  const real_t dx;
  const int m;
  const int index_start;
  std::vector<real_t> sigs; // = exp(-0.5*(x_min+x_max))
  std::vector<real_t> absorption_rates;
  const real_t particle_min_weight;
};

Layer decompose_domain(real_t x_min, real_t x_max, real_t x_ini, int world_size,
                       int world_rank, int cells_per_layer, int nb_particles,
                       real_t particle_min_weight);
#endif