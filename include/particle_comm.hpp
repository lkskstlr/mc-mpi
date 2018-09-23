#ifndef PARTICLE_COMM_HPP
#define PARTICLE_COMM_HPP

#include "types.hpp"
#include <mpi.h>
#include <vector>

typedef struct send_info_tag {
  Particle *buf;
  std::size_t bytes;
  MPI_Request request;
} SendInfo;

class ParticleComm {
public:
  ParticleComm(int world_size, int world_rank, std::size_t max_buffer_size);

  // TODO avoid the copy being made of the memory of vector
  void send_particles(std::vector<Particle> const &particles, int rel_dest);
  bool receive_particles(std::vector<Particle> &particles);

  const int world_size;
  const int world_rank;

  std::size_t curr_buffer_size = 0;
  const std::size_t max_buffer_size;
  std::size_t max_buffer_size_attained = 0;

  MPI_Datatype mcmpi_particle_type;
  std::vector<SendInfo> send_infos;

private:
  // frees up memory
  void free();
};
#endif