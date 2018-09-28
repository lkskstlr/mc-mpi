#include "layer.hpp"

#ifndef VECTOR_RESERVE
#define VECTOR_RESERVE 10000
#endif

Layer decompose_domain(UnifDist &dist, real_t x_min, real_t x_max, real_t x_ini,
                       int world_size, int world_rank,
                       std::size_t cells_per_layer, std::size_t nb_particles,
                       real_t particle_min_weight) {
  assert(x_max > x_min);
  real_t dx = (x_max - x_min) / world_size;
  int nc_ini = (int)((x_ini - x_min) / dx);
  Layer layer(x_min + world_rank * dx, x_min + (1 + world_rank) * dx,
              cells_per_layer, particle_min_weight);
  if (world_rank == nc_ini) {
    layer.create_particles(dist, x_ini, 1.0 / nb_particles, nb_particles);
  }

  return layer;
}

Layer::Layer(real_t x_min, real_t x_max, std::size_t m,
             real_t particle_min_weight)
    : x_min(x_min), x_max(x_max), m(m), dx((x_max - x_min) / m),
      particle_min_weight(particle_min_weight) {

  // sigs
  sigs.reserve(m);
  real_t x_mid;
  for (std::size_t i = 0; i < m; ++i) {
    x_mid = (i * dx) + 0.5 * dx;
    sigs.push_back(exp(-0.5 * x_mid));
  }

  // absorption rates
  absorption_rates = std::vector<real_t>(m, 0.5);

  // weights absorbed
  weights_absorbed = std::vector<real_t>(m, 0.0);

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
  // constants
  int index_clostest_edge = round((particle.x - x_min) / dx);
  // float problem: check if wrong cell by float eps
  int tmp_index = static_cast<int>((particle.x - x_min) / dx);
  if (abs(index_clostest_edge * dx + x_min - particle.x) < EPS_PRECISION) {
#ifndef NDEBUG
    printf("    ");
#endif
    if (particle.mu >= 0) {
      tmp_index = index_clostest_edge;
    } else {
      tmp_index = index_clostest_edge - 1;
    }
  }
  const int index = tmp_index;

#ifndef NDEBUG
  printf("step: (x_min, x_max) = (%f, %f), x_begin = %f, index = %d, mu = %f, ",
         x_min, x_max, particle.x, index, particle.mu);
#endif

  const real_t interaction_rate = 1.0 - absorption_rates[index];
  const real_t sig_a = sigs[index] * absorption_rates[index];
  const real_t sig_i = sigs[index] * interaction_rate;

  // const real_t sig_i =

  // -- Calculate distance to nearest edge --
  int index_new;
  real_t x_new_edge;

  if (particle.mu < -EPS_PRECISION || EPS_PRECISION < particle.mu) {
    if (particle.mu < 0) {
      index_new = index - 1;
      x_new_edge = index * dx + x_min;
    } else {
      index_new = index + 1;
      x_new_edge = (index + 1) * dx + x_min;
    }
  }
  const real_t di_edge = (x_new_edge - particle.x) / particle.mu;

  // -- Draw random traveled distance. If particle would exit set position to
  // exit point --
  const real_t h = 1.0 - dist(); // in (0, 1]

  real_t di = MAXREAL;

  if (sig_i > EPS_PRECISION) {
    di = -log(h) / sig_i;
  }

#ifndef NDEBUG
  printf("di*mu = %f, ", particle.mu * di);
#endif
  if (di < di_edge) {
    /* move inside cell an draw new mu */
    index_new = index;
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
  weights_absorbed[index] += dw;

#ifndef NDEBUG
  printf("x_end = %f, index_new = %d\n", particle.x, index_new);
#endif
  return index_new;
}

void Layer::simulate(UnifDist &dist, std::size_t nb_steps,
                     std::vector<Particle> &particles_left,
                     std::vector<Particle> &particles_right,
                     std::vector<Particle> &particles_disabled) {
  if (particles.empty()) {
    return;
  }

  std::size_t max_particles = std::max(particles.size(), nb_steps);
  particles_left.reserve(particles_left.size() + max_particles / 2);
  particles_right.reserve(particles_right.size() + max_particles / 2);
  particles_disabled.reserve(particles_disabled.size() + max_particles / 2);

  int index_new;
  for (std::size_t i = 0; i < nb_steps && !particles.empty(); ++i) {
    index_new = particle_step(dist, particles.back());

    if (particles.back().wmc < particle_min_weight) {
      /* disable */
      particles_disabled.push_back(particles.back());
      particles.pop_back();
    } else {
      if (index_new == m) {
        particles_right.push_back(particles.back());
        particles.pop_back();
      } else if (index_new == -1) {
        particles_left.push_back(particles.back());
        particles.pop_back();
      }
    }
  }
  particles_left.shrink_to_fit();
  particles_right.shrink_to_fit();
  particles_disabled.shrink_to_fit();
}