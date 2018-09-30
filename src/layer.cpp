#include "layer.hpp"
#include <cmath>

#ifndef VECTOR_RESERVE
#define VECTOR_RESERVE 10000
#endif

Layer decompose_domain(UnifDist &dist, real_t x_min, real_t x_max, real_t x_ini,
                       int world_size, int world_rank, int cells_per_layer,
                       int nb_particles, real_t particle_min_weight) {
  assert(x_max > x_min);
  real_t dx = (x_max - x_min) / world_size;
  int nc_ini = (int)((x_ini - x_min) / dx);
  Layer layer(x_min + world_rank * dx, x_min + (1 + world_rank) * dx,
              world_rank * cells_per_layer, cells_per_layer,
              particle_min_weight);
  if (world_rank == nc_ini) {
    layer.create_particles(dist, x_ini, 1.0 / nb_particles, nb_particles);
  }

  return layer;
}

Layer::Layer(real_t x_min, real_t x_max, int index_start, int m,
             real_t particle_min_weight)
    : x_min(x_min), x_max(x_max), m(m), index_start(index_start),
      dx((x_max - x_min) / m), particle_min_weight(particle_min_weight) {

  // sigs
  sigs.reserve(m);
  real_t x_mid;
  for (int i = 0; i < m; ++i) {
    x_mid = (i * dx) + 0.5 * dx;
    sigs.push_back(exp(-0.5 * x_mid));
  }

  // absorption rates
  absorption_rates = std::vector<real_t>(m, 0.5);

  // weights absorbed
  weights_absorbed = std::vector<real_t>(m, 0.0);

  particles.reserve(VECTOR_RESERVE);
}

void Layer::create_particles(UnifDist &dist, real_t x_ini, real_t wmc, int n) {
  if (x_ini > x_min && x_ini < x_max) {
    Particle particle = Particle();
    particle.x = x_ini;
    particle.wmc = wmc;

    // compute index
    particle.index = static_cast<int>(x_ini / dx);

    particles.reserve(n);
    for (int i = 0; i < n; ++i) {
      particle.mu = 2.0 * dist() - 1.0;

      assert(particle.x >= x_min && particle.x <= x_max &&
             "Particle position should be in layer.");
      assert(particle.index >= 0 &&
             particle.index <= static_cast<int>(x_max / dx) &&
             "Particle index should be in layer.");
      assert(particle.wmc >= 0.0 && particle.wmc <= 1.0 &&
             "Particle weight must be in [0, 1]");
      particles.push_back(particle);
    }
  }
}

int Layer::particle_step(UnifDist &dist, Particle &particle) {
  assert(particle.x >= x_min && particle.x <= x_max &&
         "Particle position should be in layer at call.");
  assert(particle.index >= index_start && particle.index < index_start + m &&
         "Particle index should be in layer at call.");

  const int index_local = particle.index - index_start;

  const real_t interaction_rate = 1.0 - absorption_rates[index_local];
  const real_t sig_a = sigs[index_local] * absorption_rates[index_local];
  const real_t sig_i = sigs[index_local] * interaction_rate;

  // calculate theoretic movement
  const real_t h = 1.0 - dist(); // in (0, 1]
  real_t di = sig_i > EPS_PRECISION ? -std::log(h) / sig_i : MAXREAL;

  // -- possible new cell --
  int index_new;
  real_t x_new_edge;

  if (particle.mu < 0) {
    index_new = particle.index - 1;
    x_new_edge = particle.index * dx;
  } else {
    index_new = particle.index + 1;
    x_new_edge = (particle.index + 1) * dx;
  }
  const real_t di_edge = abs(x_new_edge - particle.x);

  if (abs(di * particle.mu) < di_edge) {
    /* move inside cell an draw new mu */
    index_new = particle.index;
    particle.x += di * particle.mu;
    particle.mu = 2.0 * dist() - 1.0;
  } else {
    /* set position to border */
    di = di_edge;
    particle.x = x_new_edge;
  }

  // -- Calculate amount of absorbed energy --
  const real_t dw = (1 - exp(-sig_a * di)) * particle.wmc;

  /* Weight removed from particle is added to the layer */
  particle.wmc -= dw;
  weights_absorbed[index_local] += dw;

  particle.index = index_new;

  assert(particle.x >= x_min - EPS_PRECISION &&
         particle.x <= x_max + EPS_PRECISION &&
         "Particle position should be in layer +/- eps at return.");
  assert(particle.index >= index_start - 1 &&
         particle.index <= index_start + m &&
         "Particle index should be in layer +/- 1 at return.");
  return index_new;
}

void Layer::simulate(UnifDist &dist, int nb_steps,
                     std::vector<Particle> &particles_left,
                     std::vector<Particle> &particles_right,
                     std::vector<Particle> &particles_disabled) {
  if (particles.empty()) {
    return;
  }

  int max_particles = std::max(static_cast<int>(particles.size()), nb_steps);
  particles_left.reserve(particles_left.size() + max_particles / 2);
  particles_right.reserve(particles_right.size() + max_particles / 2);
  particles_disabled.reserve(particles_disabled.size() + max_particles / 2);

  int index_new;
  for (int i = 0; i < nb_steps && !particles.empty(); ++i) {
    index_new = particle_step(dist, particles.back());

    if (particles.back().wmc < particle_min_weight) {
      /* disable */
      particles_disabled.push_back(particles.back());
      particles.pop_back();
    } else {
      if (index_new == index_start + m) {
        particles_right.push_back(particles.back());
        particles.pop_back();
      } else if (index_new == index_start - 1) {
        particles_left.push_back(particles.back());
        particles.pop_back();
      }
    }
  }
  particles_left.shrink_to_fit();
  particles_right.shrink_to_fit();
  particles_disabled.shrink_to_fit();
}