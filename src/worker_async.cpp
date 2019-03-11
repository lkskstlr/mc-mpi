#include "worker_async.hpp"
#include <numeric>
#include <chrono>
#include <thread>

WorkerAsync::WorkerAsync(int world_rank, MCMPIOptions &options)
    : Worker(world_rank, options),
      particle_comm(world_rank, options.buffer_size),
      state_comm(options.world_size, world_rank, MCMPIOptions::Tag::State,
                 [nb_particles = options.nb_particles](std::vector<int> msgs) {
                   int sum = std::accumulate(msgs.begin(), msgs.end(), 0);
                   if (sum == (int)nb_particles)
                   {
                     return StateComm::State::Finished;
                   }
                   return StateComm::State::Running;
                 }) {}

void WorkerAsync::spin()
{
  using std::chrono::high_resolution_clock;
  auto timestamp = timer.start(Timer::Tag::Idle);

  int nb_cycles_stats = 0;

  auto start = high_resolution_clock::now();
  while (true)
  {
    start = high_resolution_clock::now();

    /* Simulate Particles */
    timer.change(timestamp, Timer::Tag::Computation);
    {
      layer.simulate(options.nb_particles_per_cycle, options.nthread);
    }

    /* Sending Particles */
    timer.change(timestamp, Timer::Tag::Send);
    {
      particle_comm.send(layer.particles_left, world_rank - 1,
                         MCMPIOptions::Tag::Particle);
      particle_comm.send(layer.particles_right, world_rank + 1,
                         MCMPIOptions::Tag::Particle);
    }

    /* Receive Particles */
    timer.change(timestamp, Timer::Tag::Recv);
    {
      // recv
      particle_comm.recv(layer.particles, MPI_ANY_SOURCE,
                         MCMPIOptions::Tag::Particle);
    }

    /* Send Events */
    timer.change(timestamp, Timer::Tag::Send);
    {
      state_comm.send_msg(layer.nb_disabled);
      state_comm.send_state();
    }

    /* Receive Events */
    timer.change(timestamp, Timer::Tag::Recv);
    {
      if (state_comm.recv_state() == StateComm::State::Finished)
      {
        break;
      }
    }

    /* Timer State */
    if (timer.time() > timer.starttime() + options.statistics_cycle_time)
    {
      timer_states.push_back(timer.restart(timestamp, Timer::Tag::Computation));
      stats_states.push_back(particle_comm.reset_stats() +
                             state_comm.reset_stats());
      cycle_states.push_back(nb_cycles_stats);
      nb_cycles_stats = 0;
    }

    /* Idle */
    timer.change(timestamp, Timer::Tag::Idle);
    {
      std::chrono::duration<double, std::milli> elapsed = high_resolution_clock::now() - start;
      if (elapsed.count() < options.cycle_time * 1e3)
      {
        using std::chrono::high_resolution_clock;
        auto sleep_length = std::chrono::duration<double, std::milli>(
                                options.cycle_time * 1e3) -
                            elapsed;
        std::this_thread::sleep_for(sleep_length);
      }
    }

    nb_cycles_stats++;
  }

  timer_states.push_back(timer.stop(timestamp));
  stats_states.push_back(particle_comm.reset_stats() +
                         state_comm.reset_stats());
  cycle_states.push_back(nb_cycles_stats);
  return;
}
