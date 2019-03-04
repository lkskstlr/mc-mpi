#include "worker_sync.hpp"
#include <stddef.h>

WorkerSync::WorkerSync(int world_rank, const MCMPIOptions &options)
    : Worker(world_rank, options)
{
  constexpr int nitems = 5;
  int blocklengths[nitems] = {1, 1, 1, 1, 1};
  MPI_Datatype types[nitems] = {MPI_UNSIGNED_LONG_LONG, MPI_FLOAT, MPI_FLOAT,
                                MPI_FLOAT, MPI_INT};
  MPI_Aint offsets[nitems];

  offsets[0] = offsetof(Particle, seed);
  offsets[1] = offsetof(Particle, x);
  offsets[2] = offsetof(Particle, mu);
  offsets[3] = offsetof(Particle, wmc);
  offsets[4] = offsetof(Particle, index);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                         &mpi_particle_type);
  MPI_Type_commit(&mpi_particle_type);
}

void WorkerSync::spin()
{
  auto timestamp = timer.start(Timer::Tag::Idle);
  int nb_cycles_stats = 0;

  MPI_Status status;

  int world_size = options.world_size;

  while (true)
  {
    /* Simulate Particles */
    timer.change(timestamp, Timer::Tag::Computation);
    {
      layer.simulate(options.nb_particles_per_cycle, options.nthread);
    }

    /* SendRecv Particles */
    timer.change(timestamp, Timer::Tag::Send);
    {
      int old_size, recv_count;
      // Prepare the particles vector for the extra ones from the left OR right,
      // which are AT MOST options.nb_particles_per_cycle
      old_size = layer.particles.size();
      layer.particles.resize(layer.particles.size() +
                             options.nb_particles_per_cycle);
      /************************** [ODDS] <==> [EVENS]
       * ****************************/
      recv_count = 0;
      if ((world_rank % 2) && (world_rank + 1 < world_size))
      {
        // ODD
        MPI_Sendrecv(layer.particles_right.data(), layer.particles_right.size(),
                     mpi_particle_type, world_rank + 1, particle_tag,
                     layer.particles.data() + old_size,
                     options.nb_particles_per_cycle, mpi_particle_type,
                     world_rank + 1, particle_tag, MPI_COMM_WORLD, &status);

        layer.particles_right.clear();
        MPI_Get_count(&status, mpi_particle_type, &recv_count);
      }
      if (!(world_rank % 2) && (world_rank > 0))
      {
        // EVEN
        MPI_Sendrecv(layer.particles_left.data(), layer.particles_left.size(),
                     mpi_particle_type, world_rank - 1, particle_tag,
                     layer.particles.data() + old_size,
                     options.nb_particles_per_cycle, mpi_particle_type,
                     world_rank - 1, particle_tag, MPI_COMM_WORLD, &status);
        layer.particles_left.clear();
        MPI_Get_count(&status, mpi_particle_type, &recv_count);
      }

      layer.particles.resize(old_size + recv_count +
                             options.nb_particles_per_cycle);
      old_size += recv_count;

      MPI_Barrier(MPI_COMM_WORLD);
      /************************** [EVENS] <==> [ODDS]
       * ****************************/
      recv_count = 0;
      if (!(world_rank % 2) && (world_rank + 1 < world_size))
      {
        // EVEN
        MPI_Sendrecv(layer.particles_right.data(), layer.particles_right.size(),
                     mpi_particle_type, world_rank + 1, particle_tag,
                     layer.particles.data() + old_size,
                     options.nb_particles_per_cycle, mpi_particle_type,
                     world_rank + 1, particle_tag, MPI_COMM_WORLD, &status);
        layer.particles_right.clear();
        MPI_Get_count(&status, mpi_particle_type, &recv_count);
      }
      if (world_rank % 2)
      {
        // ODD
        MPI_Sendrecv(layer.particles_left.data(), layer.particles_left.size(),
                     mpi_particle_type, world_rank - 1, particle_tag,
                     layer.particles.data() + old_size,
                     options.nb_particles_per_cycle, mpi_particle_type,
                     world_rank - 1, particle_tag, MPI_COMM_WORLD, &status);
        layer.particles_left.clear();
        MPI_Get_count(&status, mpi_particle_type, &recv_count);
      }

      layer.particles.resize(old_size + recv_count);
    }

    /* SendRecv Events */
    timer.change(timestamp, Timer::Tag::Recv);
    {
      int disabled_own, disabled_all;
      disabled_own = layer.nb_disabled;
      MPI_Allreduce(&disabled_own, &disabled_all, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      if (disabled_all == (int)options.nb_particles)
        break;
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
