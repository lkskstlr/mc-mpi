#ifndef PARTICLE_RMA_COMM
#define PARTICLE_RMA_COMM

#include "particle.hpp"
#include "rma_comm.hpp"

class ParticleRmaComm : public RmaComm<Particle> {
public:
  ParticleRmaComm(int world_rank, int buffer_size);
};
#endif
