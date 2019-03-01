#ifndef PARTICLE_ASYNC_COMM
#define PARTICLE_ASYNC_COMM

#include "async_comm.hpp"
#include "particle.hpp"

class ParticleAsyncComm : public AsyncComm<Particle> {
public:
  ParticleAsyncComm(int world_rank, size_t max_buffer_size);
};
#endif
