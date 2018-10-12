#ifndef PARTICLE_COMM
#define PARTICLE_COMM

#include "async_comm.hpp"
#include "particle.hpp"

class ParticleComm : public AsyncComm<Particle> {
public:
  ParticleComm(int world_rank, std::size_t max_buffer_size);
};
#endif