#include "particle_comm.hpp"
#include "stdio.h"
#include <string.h>

#define MCMPI_PARTICLE_TAG 1

// Implementation relies on std::vector being contignuous in memory which is
// guranteed by the standard
// https://stackoverflow.com/questions/849168/are-stdvector-elements-guaranteed-to-be-contiguous

ParticleComm::ParticleComm(int world_size, int world_rank,
                           std::size_t max_buffer_size)
    : world_size(world_size), world_rank(world_rank),
      max_buffer_size(max_buffer_size) {

  // Particle as MPI Type
  constexpr int nitems = 3;
  int blocklengths[nitems] = {1, 1, 1};
  MPI_Datatype types[nitems] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
  MPI_Aint offsets[nitems];

  offsets[0] = offsetof(Particle, x);
  offsets[1] = offsetof(Particle, mu);
  offsets[2] = offsetof(Particle, wmc);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                         &mcmpi_particle_type);
  MPI_Type_commit(&mcmpi_particle_type);

  send_infos.reserve(1024);
}

void ParticleComm::send_particles(std::vector<Particle> const &particles,
                                  int rel_dest) {
  if (particles.empty()) {
    return;
  }
  int dest = world_rank + rel_dest;
  assert((dest >= 0) && (dest < world_size));

  SendInfo send_info;
  send_info.bytes = sizeof(Particle) * particles.size();

  // Use only limited space
  assert(send_info.bytes <= max_buffer_size);
  do {
    this->free();
  } while (curr_buffer_size + send_info.bytes > max_buffer_size);

  send_info.buf = (Particle *)malloc(send_info.bytes);
  memcpy(send_info.buf, particles.data(), send_info.bytes);

  MPI_Issend(send_info.buf, particles.size(), mcmpi_particle_type, dest,
             MCMPI_PARTICLE_TAG, MPI_COMM_WORLD, &send_info.request);

  send_infos.push_back(send_info);
}

void ParticleComm::free() {
  MPI_Status status;
  int flag = 0;

  // only valid with erase:
  // https://en.cppreference.com/w/cpp/container/vector/erase
  for (auto iter = send_infos.begin(); iter != send_infos.end();) {
    // MPI
    flag = 0;
    MPI_Test(&(iter->request), &flag, &status);

    if (flag) {
      /* the send is through */
      ::free(iter->buf); // from std lib
      curr_buffer_size -= iter->bytes;
      iter = send_infos.erase(iter);
    } else {
      ++iter;
    }
  }
}

bool ParticleComm::receive_particles(std::vector<Particle> &particles) {
  // TODO speedup by one big buffer

  MPI_Status status;
  int flag;
  int number_particles = -1;
  do {
    MPI_Iprobe(MPI_ANY_SOURCE, MCMPI_PARTICLE_TAG, MPI_COMM_WORLD, &flag,
               &status);
    if (flag) {
      MPI_Get_count(&status, mcmpi_particle_type, &number_particles);
      Particle *buf = (Particle *)malloc(sizeof(Particle) * number_particles);
      MPI_Recv(buf, number_particles, mcmpi_particle_type, MPI_ANY_SOURCE,
               MCMPI_PARTICLE_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

      particles.reserve(particles.size() + number_particles);
      for (std::size_t i = 0; i < number_particles; ++i) {
        particles.push_back(buf[i]);
      }

      ::free(buf);
    }
  } while (flag);

  return (number_particles != -1);
}