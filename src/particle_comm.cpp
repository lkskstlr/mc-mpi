#include "particle_comm.hpp"

ParticleComm::ParticleComm(int world_rank, std::size_t max_buffer_size) {

  /* Particle as MPI Type */
  MPI_Datatype mpi_particle_type;
  constexpr int nitems = 4;
  int blocklengths[nitems] = {1, 1, 1, 1};
  MPI_Datatype types[nitems] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT};
  MPI_Aint offsets[nitems];

  offsets[0] = offsetof(Particle, x);
  offsets[1] = offsetof(Particle, mu);
  offsets[2] = offsetof(Particle, wmc);
  offsets[3] = offsetof(Particle, index);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                         &mpi_particle_type);
  MPI_Type_commit(&mpi_particle_type);

  init(world_rank, mpi_particle_type, max_buffer_size);
}