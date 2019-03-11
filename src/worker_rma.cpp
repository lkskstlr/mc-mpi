#include "worker_rma.hpp"
#include <numeric>

WorkerRma::WorkerRma(int world_rank, MCMPIOptions &options)
    : Worker(world_rank, options), particle_comm(world_rank, -1),
      state_comm(options.world_size, world_rank, MCMPIOptions::Tag::State,
                 [nb_particles = options.nb_particles](std::vector<int> msgs) {
                   int sum = std::accumulate(msgs.begin(), msgs.end(), 0);
                   if (sum == (int)nb_particles)
                   {
                     return StateComm::State::Finished;
                   }
                   return StateComm::State::Running;
                 }) {}

void WorkerRma::spin()
{
  auto timestamp = timer.start(Timer::Tag::Idle);

  int nb_cycles_stats = 0;
  while (true)
  {
    /* Simulate Particles */
    timer.change(timestamp, Timer::Tag::Computation);
    {
      layer.simulate(options.nb_particles_per_cycle, options.nthread);
    }

    /* Sending Particles */
    timer.change(timestamp, Timer::Tag::Send);
    {
      particle_comm.send(layer.particles_left, world_rank - 1);
      particle_comm.send(layer.particles_right, world_rank + 1);
    }

    /* Receive Particles */
    timer.change(timestamp, Timer::Tag::Recv);
    {
      // recv
      particle_comm.recv(layer.particles, world_rank - 1);
      particle_comm.recv(layer.particles, world_rank + 1);
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

      cycle_states.push_back(nb_cycles_stats);
      nb_cycles_stats = 0;
    }

    nb_cycles_stats++;
  }

  timer_states.push_back(timer.stop(timestamp));
  cycle_states.push_back(nb_cycles_stats);
  return;
}
