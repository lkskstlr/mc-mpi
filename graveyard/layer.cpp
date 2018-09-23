#include "layer.hpp"
#include <algorithm>

#ifndef VECTOR_RESERVE
#define VECTOR_RESERVE 10000
#endif

Layer decompose_domain(UnifDist &dist, real_t x_min, real_t x_max, real_t x_ini,
                       int world_size, int world_rank,
                       std::size_t nb_particles) {
  assert(x_max > x_min);
  real_t dx = (x_max - x_min) / world_size;
  int nc_ini = (int)((x_ini - x_min) / dx);
  Layer layer(x_min + world_rank * dx, x_min + (1 + world_rank) * dx);
  if (world_rank == nc_ini) {
    layer.create_particles(dist, x_ini, 1.0 / nb_particles, nb_particles);
  }

  return layer;
}

Layer::Layer(real_t x_min, real_t x_max)
    : x_min(x_min), x_max(x_max), sig(std::exp(-0.5 * (x_min + x_max))),
      sig_i(sig * interaction_rate), sig_a(sig * absorption_rate) {
  particles.reserve(VECTOR_RESERVE);
}

void Layer::create_particles(UnifDist &dist, real_t x_ini, real_t wmc,
                             std::size_t n) {
  if (x_ini > x_min && x_ini < x_max) {
    Particle particle = Particle();
    particle.x = x_ini;
    particle.wmc = wmc;

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

  if (particle.mu < 0) {
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

bool Layer::simulate(UnifDist &dist, std::size_t nb_steps,
                     std::vector<Particle> &particles_left,
                     std::vector<Particle> &particles_right) {
  if (particles.empty()) {
    return false;
  }
  int index_new;
  Particle &particle = particles.back();

  for (std::size_t i = 0; i < nb_steps; ++i) {
    index_new = particle_step(dist, particle);

    switch (index_new) {
    case 0:
      break;
    case 1:
      particles_right.push_back(particle);
      particles.pop_back();
      if (particles.empty()) {
        return true;
      }
      particle = particles.back();
      break;
    case -1:
      particles_left.push_back(particle);
      particles.pop_back();
      if (particles.empty()) {
        return true;
      }
      particle = particles.back();
      break;
    }
  }
  return true;
}