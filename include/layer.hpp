#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include "random.hpp"
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
  Layer(real_t x_min, real_t x_max, std::size_t index_start, std::size_t m,
        real_t particle_min_weight);

  /*!
   * \function create_particles
   *
   * \brief Creates the particles in this layer
   *
   * \param[in] dist UnifDist objext reference
   * \param[in] x_ini Initial position of all particles (must be in (x_min,
   * x_max))
   * \param[in] wmc weight monte carlo
   * \param[in] n number of partciles to create
   */
  void create_particles(UnifDist &dist, real_t x_ini, real_t wmc,
                        std::size_t n);

  /*!
   * \function simulate
   *
   * \brief Simulates nb_steps steps of partcile movement choosing basically
   * random particles from the domain. One particle will be simulated until it
   * leaves the domain, then another one etc..
   *
   * \param[in] dist UnifDist objext reference
   * \param[in] nb_steps Number of simulation steps
   * \param[in] particles_left reference to the vector at wich the particles
   * that leave the layer to the left (x_min) will be appended
   * \param[in] particles_right reference to the vector at wich the particles
   * that leave the layer to the right (x_max) will be appended
   * \param[in] particles_disabled reference to the vector at wich the particles
   * that fall below the min weight will be appended
   */
  void simulate(UnifDist &dist, std::size_t nb_steps,
                std::vector<Particle> &particles_left,
                std::vector<Particle> &particles_right,
                std::vector<Particle> &particles_disabled);

  // -- Data --
  const real_t x_min, x_max;
  std::vector<Particle> particles;
  std::vector<real_t> weights_absorbed;

private:
  int particle_step(UnifDist &dist, Particle &particle);

  // -- physical properties --
  const real_t dx;
  const std::size_t m;
  const std::size_t index_start;
  std::vector<real_t> sigs; // = exp(-0.5*(x_min+x_max))
  std::vector<real_t> absorption_rates;
  const real_t particle_min_weight;
};

Layer decompose_domain(UnifDist &dist, real_t x_min, real_t x_max, real_t x_ini,
                       int world_size, int world_rank,
                       std::size_t cells_per_layer, std::size_t nb_particles,
                       real_t particle_min_weight);
#endif
