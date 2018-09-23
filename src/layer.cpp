#include "layer.hpp"
#include <algorithm>

#ifndef VECTOR_RESERVE
#define VECTOR_RESERVE 10000
#endif

Layer::Layer(real_t x_min, real_t x_max)
    : x_min(x_min), x_max(x_max), sig(std::exp(-0.5 * (x_min + x_max))),
      sig_i(sig * interaction_rate), sig_a(sig * absorption_rate) {
  particles.reserve(VECTOR_RESERVE);
}

void Layer::create_particles(UnifDist &dist, real_t x_ini, std::size_t n) {
  if (x_ini > x_min && x_ini < x_max) {
    Particle particle = Particle();
    particle.x = x_ini;
    particle.wmc = 0.0;

    particles.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      particle.mu = 2.0 * dist() - 1.0;
      particles.push_back(particle);
    }
  }
}

int Layer::particle_step(UnifDist &dist, Particle &particle) {
  // -- Calculate distance to nearest edge --
  int index_new;
  real_t di_edge;
  real_t x_new_edge;

  if (particle.x - x_min > x_max - particle.x) {
    index_new = -1;
    di_edge = particle.x - x_min;
    x_new_edge = x_min;
  } else {
    index_new = 1;
    di_edge = x_max - particle.x;
    x_new_edge = x_max;
  }

  // -- Draw random traveled distance. If particle would exit set position to
  // exit point --
  const real_t h = 1.0 - dist(); // in (0, 1]

  real_t di = MAXREAL;

  if (sig_i > EPS_PRECISION) {
    di = -log(h) / sig_i;
  }

  if (di < di_edge) {
    /* move inside cell an draw new mu */
    index_new = 0;
    particle.x += di * particle.mu;
    particle.mu = 2 * dist() - 1;
  } else {
    /* set position to border */
    di = di_edge;
    particle.x = x_new_edge;
  }

  // -- Calculate amount of absorbed energy --
  const real_t dw = (1 - exp(-sig_a * di)) * particle.wmc;

  /* Weight removed from particle is added to the layer */
  particle.wmc -= dw;
  weight_absorbed += dw;

  return index_new;
}