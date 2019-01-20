#include <mpi.h>
#include <iostream>
#include "particle_comm.hpp"
#include "types.hpp"

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

  size_t max_buffer_size = 10 * sizeof(Particle);
  ParticleComm comm(world_rank, max_buffer_size);

  Particle p{5127801, 0.172634512365276, 123.12342351, -1231.123486123,
             2134512};
  const int hash1 = hash((char *)&p, sizeof(p));

  bool flag_success;
  if (world_rank == 1) {
    comm.send(p, 0, 0);
  } else {
    std::vector<Particle> recv_particles;

    do {
      comm.recv(recv_particles, MPI_ANY_SOURCE, MPI_ANY_TAG);
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
    if (flag_success) {
      std::cout << "CORRECT" << std::endl;
    } else {
      std::cout << "ERROR: hash values are not identical. Serialization does "
                   "not work!"
                << std::endl;
    }
  }

  MPI_Finalize();
  return 0;
}
