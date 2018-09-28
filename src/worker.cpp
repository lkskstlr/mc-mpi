#include "worker.hpp"
#include <chrono>
#include <thread>

using std::size_t;

Worker::Worker(int world_rank, const McOptions &options)
    : world_rank(world_rank), options(options), dist(SOME_SEED + world_rank),
      layer(decompose_domain(dist, options.x_min, options.x_max, options.x_ini,
                             options.world_size, world_rank,
                             options.nb_cells_per_layer, options.nb_particles,
                             options.particle_min_weight)) {
  /* Particle as MPI Type */
  MPI_Datatype mpi_particle_type;
  constexpr int nitems = 3;
  int blocklengths[nitems] = {1, 1, 1};
  MPI_Datatype types[nitems] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
  MPI_Aint offsets[nitems];

  offsets[0] = offsetof(Particle, x);
  offsets[1] = offsetof(Particle, mu);
  offsets[2] = offsetof(Particle, wmc);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                         &mpi_particle_type);
  MPI_Type_commit(&mpi_particle_type);

  particle_comm.init(world_rank, mpi_particle_type, options.buffer_size);

  /* Event as int */
  event_comm.init(world_rank, MPI_INT, options.buffer_size);
}

void Worker::spin() {
  using std::chrono::high_resolution_clock;

  auto start = high_resolution_clock::now();
  auto finish = high_resolution_clock::now();
  std::vector<int> vec_nb_particles_disabled;
  if (world_rank == 0) {
    vec_nb_particles_disabled = std::vector<int>(options.world_size, 0);
  }

  bool flag = true;
  while (flag) {
    start = high_resolution_clock::now();
    // simulate
    layer.simulate(dist, MCMPI_NB_STEPS_PER_CYCLE, particles_left,
                   particles_right, particles_disabled);

    /* Send & Recv of Particles */
    {
      if (world_rank == 0) {

        particle_comm.send(particles_right, world_rank + 1, MCMPI_PARTICLE_TAG);
        particles_right.clear();
      } else if (world_rank == options.world_size - 1) {

        particle_comm.send(particles_left, world_rank - 1, MCMPI_PARTICLE_TAG);
        particles_left.clear();
      } else {

        particle_comm.send(particles_left, world_rank - 1, MCMPI_PARTICLE_TAG);
        particles_left.clear();
        particle_comm.send(particles_right, world_rank + 1, MCMPI_PARTICLE_TAG);
        particles_right.clear();
      }

      // receive
      particle_comm.recv(layer.particles, MPI_ANY_SOURCE, MCMPI_PARTICLE_TAG);
    }

    /* Send and recv of events */
    {

      if (world_rank == 0) {
        // master
        vec_nb_particles_disabled[0] = particles_left.size();

        std::vector<int> tmp_vec;
        for (int source = 1; source < options.world_size; ++source) {
          tmp_vec.clear();
          if (event_comm.recv(tmp_vec, source, MCMPI_NB_DISABLED_TAG) &&
              !tmp_vec.empty()) {
            vec_nb_particles_disabled[source] = tmp_vec[0];
          }
        }

        int nb_total_disabled = 0;
        for (auto nb : vec_nb_particles_disabled) {
          nb_total_disabled += nb;
        }

        if (nb_total_disabled == options.nb_particles) {
          // end of simulation
          for (int i = 1; i < options.world_size; ++i) {
            event_comm.send(1, i, MCMPI_FINISHED_TAG);
          }
          flag = false;
        }
      }

      if (world_rank > 0) {
        // Send number of disabled particles to master (rank = 0)
        int nb_particles_disabled = particles_disabled.size();
        if (world_rank == options.world_size - 1) {
          nb_particles_disabled += particles_right.size();
        }
        if (nb_particles_disabled > 0) {
          event_comm.send(nb_particles_disabled, 0, MCMPI_NB_DISABLED_TAG);
        }

        std::vector<int> finished_vec;
        if (event_comm.recv(finished_vec, 0, MCMPI_FINISHED_TAG)) {
          // end of simulation
          flag = false;
        }
      }
    }

    // timing
    finish = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;

#ifndef NDEBUG
    printf("[%3d/%3d]: n = %8zu, t = %f ms\n", world_rank, options.world_size,
           layer.particles.size(), elapsed.count());
#endif
    if (elapsed.count() < MCMPI_WAIT_MS) {
      std::this_thread::sleep_for(
          std::chrono::duration<double, std::milli>(MCMPI_WAIT_MS) - elapsed);
    }
  }
}

std::vector<real_t> Worker::weights_absorbed() {
  return layer.weights_absorbed;
}
