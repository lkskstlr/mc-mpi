#include "layer.hpp"
#include "particle_comm.hpp"
#include "timer.hpp"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <thread>

int main(int argc, char const *argv[]) {
  using std::chrono::high_resolution_clock;

  // -- MPI Setup --
  MPI_Init(NULL, NULL);
  int world_rank, world_size;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  constexpr double cycle_time = 1e-3;
  constexpr int num_cycles = 10'000;
  int buffer_size = 100'000'000;

  Timer timer;
  /* Commit Timer::State Type */
  MPI_Datatype timer_state_mpi_t = Timer::State::mpi_t();
  MPI_Type_commit(&timer_state_mpi_t);
  std::vector<Timer::State> timer_states;

  std::vector<Particle> particles;
  Particle p{5127801, 0.17, 123.12, -1231.1, 21};
  particles.push_back(p);

  ParticleComm particle_comm(world_rank, buffer_size);

  if (world_rank == 0) {
    auto timestamp = timer.start(Timer::Tag::Recv);
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 1'000; ++i) {
      particle_comm.recv(particles, MPI_ANY_SOURCE, 0);
    }

    timer.stop(timestamp);
    auto finish = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    std::cout << timer << std::endl;
    std::cout << "Elapsed = " << elapsed.count() << " ms" << std::endl;
    std::cout << "This means " << elapsed.count() / 1'000
              << " ms per empty receive" << std::endl;
    timer.reset();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 1) {
    auto timestamp = timer.start(Timer::Tag::Send);
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 1'000; ++i) {
      particle_comm.send(particles, 0, 0);
    }

    timer.stop(timestamp);
    auto finish = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    std::cout << timer << std::endl;
    std::cout << "Elapsed = " << elapsed.count() << " ms" << std::endl;
    std::cout << "This means " << elapsed.count() / 1'000
              << " ms per 1 Particle send" << std::endl;
    timer.reset();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 0) {
    auto timestamp = timer.start(Timer::Tag::Recv);
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 1'000; ++i) {
      particle_comm.recv(particles, MPI_ANY_SOURCE, 0);
    }

    timer.stop(timestamp);
    auto finish = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    std::cout << timer << std::endl;
    std::cout << "Elapsed = " << elapsed.count() << " ms" << std::endl;
    std::cout << "This means " << elapsed.count() / 1'000
              << " ms per 1 Particle recv" << std::endl;
    timer.reset();
    particles.clear();
    particles.push_back(p);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 1) {
    particle_comm.send(particles, 0, 0);
    auto timestamp = timer.start(Timer::Tag::Send);
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 1'000; ++i) {
      particle_comm.free();
    }

    timer.stop(timestamp);
    auto finish = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    std::cout << timer << std::endl;
    std::cout << "Elapsed = " << elapsed.count() << " ms" << std::endl;
    std::cout << "This means " << elapsed.count() / 1'000
              << " ms per 1 Partcile free" << std::endl;
    timer.reset();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 0) {
    particle_comm.recv(particles, MPI_ANY_SOURCE, 0);
    particles.clear();
    particles.push_back(p);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  {
    auto timestamp = timer.start(Timer::Tag::Idle);

    auto start = high_resolution_clock::now();
    auto finish = high_resolution_clock::now();

    for (int i = 0; i < num_cycles; ++i) {
      start = high_resolution_clock::now();

      /* Sending */
      timer.change(timestamp, Timer::Tag::Send);
      {
        particle_comm.send(particles, (world_rank + 1) % world_size, 0);
        particles.clear();
      }

      /* Receive Particles */
      timer.change(timestamp, Timer::Tag::Recv);
      {
        // recv
        particle_comm.recv(particles, MPI_ANY_SOURCE, 0);
      }

      /* Idle */
      timer.change(timestamp, Timer::Tag::Idle);
      {
        finish = high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = finish - start;
        if (elapsed.count() < cycle_time * 1e3) {
          auto sleep_length =
              std::chrono::duration<double, std::milli>(cycle_time * 1e3) -
              elapsed;
          std::this_thread::sleep_for(sleep_length);
        }
      }

      timer_states.push_back(timer.restart(timestamp, Timer::Tag::Idle));
    }

    timer_states.push_back(timer.stop(timestamp));
  }

  // GATHER TIMES
  int total_len = 0;
  int *displs = NULL;
  Timer::State *states = NULL;
  {
    int *recvcounts = NULL;

    if (world_rank == 0) {
      recvcounts = (int *)malloc(world_size * sizeof(int));
    }

    int my_len = static_cast<int>(timer_states.size());
    MPI_Gather(&my_len, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
      displs = (int *)malloc(world_size * sizeof(int));

      displs[0] = 0;
      total_len += recvcounts[0];

      for (int i = 1; i < world_size; i++) {
        total_len += recvcounts[i];
        displs[i] = displs[i - 1] + recvcounts[i - 1];
      }

      states = (Timer::State *)malloc(total_len * sizeof(Timer::State));
    }

    MPI_Gatherv(timer_states.data(), my_len, timer_state_mpi_t, states,
                recvcounts, displs, timer_state_mpi_t, 0, MPI_COMM_WORLD);
  }

  // dump
  if (world_rank == 0) {
    FILE *file;
    file = fopen("times.csv", "w");
    if (!file) {
      fprintf(stderr, "Couldn't open file times.csv for writing.\n");
      exit(1);
    }

    // printf("total_len = %d\n", total_len);
    // printf("displs = (");
    // for (int i = 0; i < world_size; ++i) {
    //   printf("%d,", displs[i]);
    // }
    // printf(")\n");

    fprintf(file, "proc, starttime, endtime, state_comp, state_send, "
                  "state_recv, state_idle\n");
    int proc = 0;
    for (int i = 0; i < total_len; ++i) {
      if (proc < world_size - 1 && displs[proc + 1] == i) {
        proc++;
      }
      fprintf(file, "%d, %.18e, %.18e, %.18e, %.18e, %.18e, %.18e\n", proc,
              states[i].starttime, states[i].endtime,
              states[i].cumm_times[Timer::Tag::Computation],
              states[i].cumm_times[Timer::Tag::Send],
              states[i].cumm_times[Timer::Tag::Recv],
              states[i].cumm_times[Timer::Tag::Idle]);
    }

    fclose(file);
  }

  MPI_Finalize();
  return 0;
}