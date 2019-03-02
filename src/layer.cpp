#include "layer.hpp"
#include "random.hpp"
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define MAX_PARTICLES_VECTOR 1000000

Layer decompose_domain(real_t x_min, real_t x_max, real_t x_ini, int world_size,
                       int world_rank, int cells_per_layer, int nb_particles,
                       real_t particle_min_weight) {
  assert(x_max > x_min);
  real_t dx = (x_max - x_min) / world_size;
  int nc_ini = (int)((x_ini - x_min) / dx);
  Layer layer(x_min + world_rank * dx, x_min + (1 + world_rank) * dx,
              world_rank * cells_per_layer, cells_per_layer,
              particle_min_weight);
  if (world_rank == nc_ini) {
    seed_t seed = 5127801;
    layer.create_particles(x_ini, 1.0 / nb_particles, nb_particles, seed);
  }

  return layer;
}

Layer::Layer(real_t x_min, real_t x_max, int index_start, int m,
             real_t particle_min_weight)
    : x_min(x_min), x_max(x_max), m(m), index_start(index_start),
      dx((x_max - x_min) / m), left_border(fabs(x_min) < EPS_PRECISION),
      right_border(fabs(x_max - 1.0) < EPS_PRECISION),
      particle_min_weight(particle_min_weight) {
  constexpr int vector_reserve = 10000;

  // sigs
  sigs.reserve(m);
  real_t x_mid;
  for (int i = 0; i < m; ++i) {
    x_mid = x_min + (i * dx) + 0.5 * dx;
    sigs.push_back(exp(-x_mid));
  }

  // absorption rates
  absorption_rates = std::vector<real_t>(m, 0.5);

  // weights absorbed
  weights_absorbed = std::vector<real_t>(m, 0.0);

  particles.reserve(vector_reserve);
}

void Layer::create_particles(real_t x_ini, real_t wmc, int n, seed_t seed) {
  if (x_ini > x_min && x_ini < x_max) {
    this->x_ini = x_ini;
    this->wmc = wmc;
    this->nb_particles_create = n;
    this->seed = seed;

    this->create_particles(MAX_PARTICLES_VECTOR);
  }
}

int Layer::nb_active() const {
  return (int)particles.size() + nb_particles_create;
}

void Layer::create_particles(int n) {
  if (nb_particles_create <= 0)
    return;

  /* how many particles to create */
  int best_num_particles = MAX_PARTICLES_VECTOR - particles.size();
  n = MAX(best_num_particles,
          n); // only required to create at least n particles
  n = MIN(nb_particles_create, n); // cannot create more than the layer offers
  nb_particles_create -= n;

  Particle particle = Particle();
  particle.x = x_ini;
  particle.wmc = wmc;

  // compute index
  particle.index = static_cast<int>(x_ini / dx);

  particles.reserve(particles.size() + n);
  for (int i = 0; i < n; ++i) {
    particle.seed = rnd_seed(&seed);
    particle.mu = 2 * rnd_real(&particle.seed) - 1;

    assert(particle.index >= 0 &&
           particle.index <= static_cast<int>(x_max / dx) &&
           "Particle index should be in layer.");
    assert(particle.wmc >= 0.0 && particle.wmc <= 1.0 &&
           "Particle weight must be in [0, 1]");
    particles.push_back(particle);
  }
}

int Layer::particle_step(Particle &particle,
                         std::vector<real_t> &weights_absorbed_local) {
  assert(particle.index >= index_start && particle.index < index_start + m &&
         "Particle index should be in layer at call.");

  const int index_local = particle.index - index_start;

  const real_t interaction_rate = 1.0 - absorption_rates[index_local];
  const real_t sig_a = sigs[index_local] * absorption_rates[index_local];
  const real_t sig_i = sigs[index_local] * interaction_rate;

  // calculate theoretic movement
  const real_t h = rnd_real(&particle.seed);
  real_t di = sig_i > EPS_PRECISION ? -log(h) / sig_i : MAXREAL;

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

  real_t di_edge = MAXREAL;
  if (particle.mu < -EPS_PRECISION || EPS_PRECISION < particle.mu) {
    di_edge = (x_new_edge - particle.x) / particle.mu;
  }

  if (di < di_edge) {
    /* move inside cell an draw new mu */
    index_new = particle.index;
    particle.x += di * particle.mu;
    particle.mu = 2 * rnd_real(&particle.seed) - 1;
  } else {
    /* set position to border */
    di = di_edge;
    particle.x = x_new_edge;
  }

  // -- Calculate amount of absorbed energy --
  const real_t dw = (1 - expf(-sig_a * di)) * particle.wmc;

  /* Weight removed from particle is added to the layer */
  particle.wmc -= dw;
  weights_absorbed_local[index_local] += dw;

  particle.index = index_new;

  assert(particle.x >= x_min - EPS_PRECISION &&
         particle.x <= x_max + EPS_PRECISION &&
         "Particle position should be in layer +/- eps at return.");
  assert(particle.index >= index_start - 1 &&
         particle.index <= index_start + m &&
         "Particle index should be in layer +/- 1 at return.");
  return index_new;
}

int Layer::simulate_particle(Particle &particle,
                             std::vector<real_t> &weights_absorbed_local) {
  while ((particle.wmc >= particle_min_weight) &&
         (particle.index < index_start + m) &&
         (particle.index >= index_start)) {
    particle_step(particle, weights_absorbed_local);
  }

  if (particle.index == index_start - 1) {
    return -1;
  }

  if (particle.index == index_start + m) {
    return +1;
  }

  if (particle.wmc < particle_min_weight) {
    return 0;
  }

  return -1;
}

void Layer::simulate_helper(int nb_particles, int nthread) {
  if (nb_particles == -1) {
    while ((particles.size() > 0) || (nb_particles_create > 0)) {
      simulate(MAX_PARTICLES_VECTOR, nthread);
    }
  }

  while ((nb_particles > 0) &&
         (particles.size() > 0 || nb_particles_create > 0)) {
    int nb_particles_this_call = MIN(nb_particles, MAX_PARTICLES_VECTOR);
    simulate(nb_particles_this_call, nthread);
    nb_particles -= nb_particles_this_call;
  }
}

void Layer::simulate(int nb_particles, int nthread) {
  if ((nb_particles == -1) || (nb_particles > MAX_PARTICLES_VECTOR))
    simulate_helper(nb_particles, nthread);

  if (((int)particles.size() < nb_particles) && nb_particles_create > 0)
    create_particles(nb_particles);

  nb_particles = MIN((int)particles.size(), nb_particles);

  if (nb_particles <= 0) {
    return;
  }

  int *const result = (int *)malloc(sizeof(int) * nb_particles);
  const int particles_size = particles.size();

#ifdef _OPENMP
  if (nthread > 0)
    omp_set_num_threads(nthread);
#endif

#pragma omp parallel default(none)                                             \
    firstprivate(result, particles_size, nb_particles)
  {
    std::vector<real_t> weights_absorbed_local;
    weights_absorbed_local.resize(weights_absorbed.size(), 0.0);
#pragma omp for schedule(static)
    for (int i = 0; i < nb_particles; i++) {
      result[i] = simulate_particle(particles[particles_size - 1 - i],
                                    weights_absorbed_local);
    }

#pragma omp critical
    {
      for (int j = 0; j < (int)weights_absorbed.size(); j++) {
        weights_absorbed[j] += weights_absorbed_local[j];
      }
    }
  }

  for (int i = 0; i < nb_particles; i++) {
    switch (result[i]) {
    case -1:
      particles_left.push_back(particles[particles_size - 1 - i]);
      break;
    case 1:
      particles_right.push_back(particles[particles_size - 1 - i]);
      break;
    case 0:
      nb_disabled++;
      break;
    }
  }

  particles.resize(particles.size() - nb_particles);

  if (left_border) {
    nb_disabled += particles_left.size();
    particles_left.clear();
  }

  if (right_border) {
    nb_disabled += particles_right.size();
    particles_right.clear();
  }
}

void Layer::dump_WA() {
  FILE *file;
  file = fopen("WA.out", "w");
  if (!file) {
    fprintf(stderr, "Couldn't open file WA.out for writing.\n");
    exit(1);
  }

  for (int i = 0; i < (int)weights_absorbed.size(); ++i) {
    fprintf(file, "%.4e %.3e\n", x_min + (i * dx) + 0.5 * dx,
            weights_absorbed[i] / dx);
  }

  fclose(file);
}
