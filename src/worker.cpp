#include "worker.hpp"
#include <chrono>
#include <thread>

using std::size_t;

Worker::Worker(int world_rank, const MCMPIOptions &options)
    : world_rank(world_rank), options(options), dist(SOME_SEED + world_rank),
      layer(decompose_domain(dist, options.x_min, options.x_max, options.x_ini,
                             options.world_size, world_rank,
                             options.nb_cells_per_layer, options.nb_particles,
                             options.particle_min_weight)),
      particle_comm(world_rank, options.buffer_size) {

  /* reserve */
  timer_states.reserve(100);

  /* Commit Timer::State Type */
  timer_state_mpi_t = Timer::State::mpi_t();
  MPI_Type_commit(&timer_state_mpi_t);

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
      layer.simulate(dist, options.cycle_nb_steps, particles_left,
                     particles_right, particles_disabled);
    }

    /* Sending Particles */
    timer.change(timestamp, Timer::Tag::Send);
    {
      if (world_rank == 0) {

        particle_comm.send(particles_right, world_rank + 1,
                           MCMPIOptions::Tag::Particle);
        particles_right.clear();
      } else if (world_rank == options.world_size - 1) {

        particle_comm.send(particles_left, world_rank - 1,
                           MCMPIOptions::Tag::Particle);
        particles_left.clear();
      } else {

        particle_comm.send(particles_left, world_rank - 1,
                           MCMPIOptions::Tag::Particle);
        particles_left.clear();
        particle_comm.send(particles_right, world_rank + 1,
                           MCMPIOptions::Tag::Particle);
        particles_right.clear();
      }
    }

    /* Receive Particles */
    timer.change(timestamp, Timer::Tag::Recv);
    {
      // recv
      particle_comm.recv(layer.particles, MPI_ANY_SOURCE,
                         MCMPIOptions::Tag::Particle);
    }

    /* Receive Events */
    timer.change(timestamp, Timer::Tag::Recv);
    {
      if (world_rank == 0) {
        vec_nb_particles_disabled[0] = particles_left.size();

        std::vector<int> tmp_vec;
        for (int source = 1; source < options.world_size; ++source) {
          tmp_vec.clear();
          if (event_comm.recv(tmp_vec, source, MCMPIOptions::Tag::Disable) &&
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
        if (event_comm.recv(finished_vec, 0, MCMPIOptions::Tag::Finish)) {
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
            event_comm.send(1, i, MCMPIOptions::Tag::Finish);
          }
        }
      } else {
        int nb_particles_disabled = particles_disabled.size();
        if (world_rank == options.world_size - 1) {
          nb_particles_disabled += particles_right.size();
        }
        if (nb_particles_disabled > 0) {
          event_comm.send(nb_particles_disabled, 0, MCMPIOptions::Tag::Disable);
        }
      }
    }

    /* Timer State */
    if (timer.time() > timer.starttime() + options.statistics_cycle_time) {
      // dump
      timer_states.push_back(timer.restart(timestamp, Timer::Tag::Idle));
    }

    /* Idle */
    timer.change(timestamp, Timer::Tag::Idle);
    {
      finish = high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = finish - start;
      if (elapsed.count() < options.cycle_time * 1e3) {
        auto sleep_length = std::chrono::duration<double, std::milli>(
                                options.cycle_time * 1e3) -
                            elapsed;
        std::this_thread::sleep_for(sleep_length);
      }
    }
  }

  timer_states.push_back(timer.stop(timestamp));
  return;
}

void Worker::gather_timings() {
  constexpr int root = 0;
  int *recvcounts = NULL;

  /* Only root has the received data */
  if (world_rank == root) {
    recvcounts = (int *)malloc(options.world_size * sizeof(int));
  }

  int my_len = static_cast<int>(timer_states.size());
  MPI_Gather(&my_len, 1, MPI_INT, recvcounts, 1, MPI_INT, root, MPI_COMM_WORLD);

  int total_len = 0;
  int *displs = NULL;
  Timer::State *total_states = NULL;

  if (world_rank == root) {
    displs = (int *)malloc(options.world_size * sizeof(int));

    displs[0] = 0;
    total_len += recvcounts[0];

    for (int i = 1; i < options.world_size; i++) {
      total_len += recvcounts[i];
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    /* allocate */
    total_states = (Timer::State *)malloc(total_len * sizeof(Timer::State));
  }

  /*
   * Now we have the receive buffer, counts, and displacements, and
   * can gather the strings
   */

  MPI_Gatherv(timer_states.data(), my_len, timer_state_mpi_t, total_states,
              recvcounts, displs, timer_state_mpi_t, root, MPI_COMM_WORLD);

  if (world_rank == root) {

    for (int i = 0; i < total_len; ++i) {
      printf("%f, ", total_states[i].starttime);
    }
    printf("\n");
  }
}

std::vector<real_t> Worker::weights_absorbed() {
  return layer.weights_absorbed;
}
