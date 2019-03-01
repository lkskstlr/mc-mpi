#include "particle_rma_comm.hpp"
#include "types.hpp"
#include <iostream>
#include <mpi.h>

int hash(char const *data, int len) {
  unsigned int hash = 0x811c9dc5;
  unsigned int prime = 16777619;
  for (int i = 0; i < len; ++i) {
    hash = hash ^ data[i];
    hash = hash * prime;
  }

  int res = *((int *)(&hash));
  return res;
}

int main(int argc, char const *argv[]) {
  // -- MPI Setup --
  MPI_Init(NULL, NULL);
  int world_rank, world_size;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  ParticleRmaComm comm(world_rank, -1);
  if (world_rank == 0)
    comm.connect(1);
  else
    comm.connect(0);

  Particle p{5127801, 0.172634512365276, 123.12342351, -1231.123486123,
             2134512};
  const int hash1 = hash((char *)&p, sizeof(p));

  bool flag_success = false;
  if (world_rank == 1) {
    std::vector<Particle> tmp = {p};
    comm.send(tmp, 0);
  } else {
    std::vector<Particle> recv_particles;

    do {
      comm.recv(recv_particles, 1);
    } while (recv_particles.size() == 0);

    Particle p2 = recv_particles.back();
    const int hash2 = hash((char *)&p2, sizeof(p2));

    if (hash1 == hash2) {
      flag_success = true;
    } else {
      flag_success = false;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0) {
    if (!flag_success)
      exit(EXIT_FAILURE);
  }

  MPI_Finalize();
  return 0;
}
