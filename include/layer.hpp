#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include "random.hpp"
#include "types.hpp"
#include <vector>

class Layer {
public:
  Layer(real_t x_min, real_t x_max);
  void create_particles(UnifDist &dist, real_t x_ini, real_t wmc,
                        std::size_t n);
  int particle_step(UnifDist &dist, Particle &particle);

  // -- Data --
  const real_t x_min, x_max;
  std::vector<Particle> particles;

  // -- physical properties --
  const real_t sig; // = exp(-0.5*(x_min+x_max))
  real_t weight_absorbed = 0;

  /* magic numbers. interaction = 1 - absorption */
  static constexpr real_t absorption_rate = 0.5;
  static constexpr real_t interaction_rate = 1.0 - 0.5;

  /* derived quantities */
  const real_t sig_i; // = sig * interaction_rate
  const real_t sig_a; // = sig * absorption_rate
};

Layer decompose_domain(UnifDist &dist, real_t x_min, real_t x_max, real_t x_ini,
                       int world_size, int world_rank,
                       std::size_t nb_particles);
#endif
