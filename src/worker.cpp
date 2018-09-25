#include "worker.hpp"
#include <chrono>
#include <thread>

using std::size_t;

Worker::Worker(int world_rank, const McOptions &options)
    : world_rank(world_rank), options(options), dist(SOME_SEED + world_rank),
      layer(decompose_domain(dist, options.x_min, options.x_max, options.x_ini,
                             options.world_size, world_rank,
                             options.nb_particles)) {
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

  bool flag = true;
  while (flag) {
    start = high_resolution_clock::now();
    // simulate
    layer.simulate(dist, 100'000'000, particles_left, particles_right);

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
      if (world_rank == options.world_size - 1) {
        // Send number particles to master (rank = 0)
        std::vector<int> nb_particles_disabled;
        nb_particles_disabled.push_back(particles_right.size());
        printf("BLUBBBBB dis = %d\n", nb_particles_disabled.back());
        event_comm.send(nb_particles_disabled, 0, MCMPI_NB_DISABLED_TAG);
      }

      if (world_rank == 0) {
        // master
        std::vector<int> nb_particles_disabled_right_vec;
        bool recv_dis =
            event_comm.recv(nb_particles_disabled_right_vec,
                            options.world_size - 1, MCMPI_NB_DISABLED_TAG);
        printf("-----MASTER n = %ld\n", nb_particles_disabled_right_vec.size());
        if (recv_dis && !nb_particles_disabled_right_vec.empty()) {
          size_t nb_disabled_right =
              static_cast<size_t>(nb_particles_disabled_right_vec[0]);
          printf("=====MASTER nb_disabled_right = %ld\n", nb_disabled_right);
          if (particles_left.size() + nb_disabled_right ==
              options.nb_particles) {
            // end of simulation
            for (int i = 1; i < options.world_size; ++i) {
              std::vector<int> finished_vec;
              finished_vec.push_back(1);
              event_comm.send(finished_vec, i, MCMPI_FINISHED_TAG);
            }
            printf("Master: Simulation finished!\n");
            flag = false;
          }
        }
      }

      if (world_rank > 0) {
        std::vector<int> finished_vec;
        bool did_receive = event_comm.recv(finished_vec, 0, MCMPI_FINISHED_TAG);
        printf("[%03d/%03d]: n_recv_event = %ld\n", world_rank,
               options.world_size, finished_vec.size());
        if (did_receive) {
          // end of simulation
          printf("Process [%d,%d]: Simulation finished.\n", world_rank,
                 options.world_size);
          flag = false;
        }
      }
    }

    // timing
    finish = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;

    printf("[%03d/%03d]: n = %08ld t = %fms\n", world_rank, options.world_size,
           layer.particles.size(), elapsed.count());
    if (elapsed.count() < MC_MPI_WAIT_MS) {
      std::this_thread::sleep_for(
          std::chrono::duration<double, std::milli>(MC_MPI_WAIT_MS) - elapsed);
    }
  }
}
