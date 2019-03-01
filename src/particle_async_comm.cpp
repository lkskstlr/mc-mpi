#include "particle_async_comm.hpp"
#include <stddef.h>

ParticleAsyncComm::ParticleAsyncComm(int world_rank, size_t max_buffer_size) {
  /* Particle as MPI Type */
  MPI_Datatype mpi_particle_type;
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

  init(world_rank, mpi_particle_type, max_buffer_size);
}
