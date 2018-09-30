#include "worker.hpp"
#include <chrono>
#include <thread>

using std::size_t;

Worker::Worker(int world_rank, const McOptions &options)
    : world_rank(world_rank), options(options), dist(SOME_SEED + world_rank),
      layer(decompose_domain(dist, options.x_min, options.x_max, options.x_ini,
                             options.world_size, world_rank,
                             options.nb_cells_per_layer, options.nb_particles,
                             options.particle_min_weight)),
      particle_comm(world_rank, options.buffer_size) {
  /* Event as int */
  event_comm.init(world_rank, MPI_INT, options.buffer_size);
}

void Worker::spin() {
  using std::chrono::high_resolution_clock;

  auto timestamp = timer.start(Timer::Tag::Idle);

  auto start = high_resolution_clock::now();
  auto finish = high_resolution_clock::now();
  std::vector<int> vec_nb_particles_disabled;
  if (world_rank == 0) {
    vec_nb_particles_disabled = std::vector<int>(options.world_size, 0);
  }

  bool flag = true;
  while (flag) {
    start = high_resolution_clock::now();

    /* Simulate Particles */
    timer.change(timestamp, Timer::Tag::Computation);
    {
      layer.simulate(dist, MCMPI_NB_STEPS_PER_CYCLE, particles_left,
                     particles_right, particles_disabled);
    }

    /* Sending Particles */
    timer.change(timestamp, Timer::Tag::Send);
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
    }

    /* Receive Particles */
    timer.change(timestamp, Timer::Tag::Recv);
    {
      // recv
      particle_comm.recv(layer.particles, MPI_ANY_SOURCE, MCMPI_PARTICLE_TAG);
    }

    /* Receive Events */
    timer.change(timestamp, Timer::Tag::Recv);
    {
      if (world_rank == 0) {
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
          flag = false;
        }
      } else {
        std::vector<int> finished_vec;
        if (event_comm.recv(finished_vec, 0, MCMPI_FINISHED_TAG)) {
          // end of simulation
          flag = false;
        }
      }
    }

    /* Send Events */
    timer.change(timestamp, Timer::Tag::Send);
    {
      if (world_rank == 0) {
        if (!flag) {
          for (int i = 1; i < options.world_size; ++i) {
            event_comm.send(1, i, MCMPI_FINISHED_TAG);
          }
        }
      } else {
        int nb_particles_disabled = particles_disabled.size();
        if (world_rank == options.world_size - 1) {
          nb_particles_disabled += particles_right.size();
        }
        if (nb_particles_disabled > 0) {
          event_comm.send(nb_particles_disabled, 0, MCMPI_NB_DISABLED_TAG);
        }
      }
    }

    /* Idle */
    timer.change(timestamp, Timer::Tag::Idle);
    {
      finish = high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = finish - start;
      if (elapsed.count() < MCMPI_WAIT_MS) {
        std::this_thread::sleep_for(
            std::chrono::duration<double, std::milli>(MCMPI_WAIT_MS) - elapsed);
      }
    }
  }

  timer.stop(timestamp);
  return;
}

std::vector<real_t> Worker::weights_absorbed() {
  return layer.weights_absorbed;
}
