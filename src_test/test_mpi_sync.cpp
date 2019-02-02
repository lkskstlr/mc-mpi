#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>
#include "layer.hpp"

int main(int argc, char *argv[])
{
  int world_size;
  int world_rank;

  constexpr real_t x_min = 0.0;
  constexpr real_t x_max = 1.0;
  const real_t x_ini = sqrtf(2.0) / 2.0;
  constexpr int nb_cells_per_layer = 100;
  constexpr int nb_particles = 1000;
  constexpr real_t particle_min_weight = 0.25 / nb_particles;

  //MPI initialization and basic function calls
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  /****************************** MPI_PARTICLE_TYPE **************************/
  MPI_Datatype mpi_particle_type;
  constexpr int nitems = 5;
  int blocklengths[nitems] = {1, 1, 1, 1, 1};
  MPI_Datatype types[nitems] = {MPI_UNSIGNED_LONG_LONG, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT};
  MPI_Aint offsets[nitems];

  offsets[0] = offsetof(Particle, seed);
  offsets[1] = offsetof(Particle, x);
  offsets[2] = offsetof(Particle, mu);
  offsets[3] = offsetof(Particle, wmc);
  offsets[4] = offsetof(Particle, index);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_particle_type);
  MPI_Type_commit(&mpi_particle_type);
  /***************************************************************************/

  MPI_Status status;

  // Decompose the domain into as many pieces as there are processes
  Layer layer(decompose_domain(x_min, x_max, x_ini, world_size, world_rank,
                               nb_cells_per_layer, nb_particles,
                               particle_min_weight));

  // Generate vectors for the particles that exit through the left, right, and the ones that get disabled
  std::vector<Particle> particles_send_left;
  std::vector<Particle> particles_send_right;
  std::vector<Particle> particles_disabled;
  std::vector<Particle> particles_recv_right;
  std::vector<Particle> particles_recv_left;

  //Simulate until there are no particles left (a lot of steps will make it run until that happens, probably)
  layer.simulate(100000000, particles_send_left, particles_send_right, particles_disabled);

  int size_send_right = particles_send_right.size();
  int size_send_left = particles_send_left.size();
  int size_disabled = particles_disabled.size();
  int size_recv_right, size_recv_left;
  /************************** [EVENS] <== [ODDS] *****************************/
  //SIZE
  if (world_rank % 2)
  {
    MPI_Ssend(&size_send_left, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD);
  }
  else if (world_rank + 1 < world_size)
  {
    MPI_Recv(&size_recv_right, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  //DATA
  if (world_rank % 2)
  {
    if (size_send_left)
      MPI_Ssend(&particles_send_left[0], size_send_left, mpi_particle_type, world_rank - 1, 0, MPI_COMM_WORLD);
  }
  else if ((world_rank + 1 < world_size) && size_recv_right)
  {
    particles_recv_right.resize(size_recv_right);
    MPI_Recv(&particles_recv_right[0], particles_recv_right.size(), mpi_particle_type, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  /************************** [ODDS] <== [EVENS] *****************************/
  //SIZE
  if (!(world_rank % 2))
  {
    if (world_rank)
      MPI_Ssend(&size_send_left, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD);
  }
  else if (world_rank + 1 < world_size)
  {
    MPI_Recv(&size_recv_right, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  //DATA
  if (!(world_rank % 2))
  {
    if (world_rank && size_send_left)
      MPI_Ssend(&particles_send_left[0], size_send_left, mpi_particle_type, world_rank - 1, 0, MPI_COMM_WORLD);
  }
  else if ((world_rank + 1 < world_size) && size_recv_right)
  {
    particles_recv_right.resize(size_recv_right);
    MPI_Recv(&particles_recv_right[0], particles_recv_right.size(), mpi_particle_type, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  /************************** [ODDS] ==> [EVENS] *****************************/
  //SIZE
  if (world_rank % 2)
  {
    if (world_rank + 1 < world_size)
      MPI_Ssend(&size_send_right, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD);
  }
  else if (world_rank)
  {
    MPI_Recv(&size_recv_left, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  //DATA
  if (world_rank % 2)
  {
    if ((world_rank + 1 < world_size) && size_send_right)
      MPI_Ssend(&particles_send_right[0], size_send_right, mpi_particle_type, world_rank + 1, 0, MPI_COMM_WORLD);
  }
  else if (world_rank && size_recv_left)
  {
    particles_recv_left.resize(size_recv_left);
    MPI_Recv(&particles_recv_left[0], particles_recv_left.size(), mpi_particle_type, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  /************************** [EVENS] ==> [ODDS] *****************************/
  //SIZE
  if (!(world_rank % 2))
  {
    if ((world_rank + 1 < world_size))
      MPI_Ssend(&size_send_right, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Recv(&size_recv_left, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  //DATA
  if (!(world_rank % 2))
  {
    if ((world_rank + 1 < world_size) && size_send_right)
      MPI_Ssend(&particles_send_right[0], size_send_right, mpi_particle_type, world_rank + 1, 0, MPI_COMM_WORLD);
  }
  else if (size_recv_left)
  {
    particles_recv_left.resize(size_recv_left);
    MPI_Recv(&particles_recv_left[0], particles_recv_left.size(), mpi_particle_type, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // if (world_rank == 0)
  // {
  //   MPI_Ssend(&size_send_right, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD);
  // }
  // else
  // {
  //   MPI_Recv(&size_recv_left, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // }

  // if (world_rank == 1)
  // {
  //   MPI_Ssend(&size_send_left, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD);
  // }
  // else
  // {
  //   MPI_Recv(&size_recv_right, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // }

  // if (world_rank == 0)
  // {
  //   MPI_Ssend(&particles_right[0], size_send_right, mpi_particle_type, world_rank + 1, 100, MPI_COMM_WORLD);
  // }
  // else
  // {
  //   particles_recv_left.resize(size_recv_left);
  //   MPI_Recv(&particles_recv_left[0], size_recv_left, mpi_particle_type, world_rank - 1, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // }

  // if (world_rank == 1)
  // {
  //   MPI_Ssend(&particles_left[0], size_send_left, mpi_particle_type, world_rank - 1, 100, MPI_COMM_WORLD);
  // }
  // else
  // {
  //   particles_recv_right.resize(size_recv_right);
  //   MPI_Recv(&particles_recv_right[0], size_recv_right, mpi_particle_type, world_rank + 1, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // }
  MPI_Barrier(MPI_COMM_WORLD);
  usleep(10000 * world_rank);
  printf("Rank: %d/%d\n\t%d particles EXITED my domain to the LEFT\n\t%d particles EXITED my domain to the RIGHT\n\t%d particles got DISABLED in my domain\n\t%d particles ENTERED my domain through the LEFT\n\t%d particles ENTERED my domain through the RIGHT\n ",
         world_rank, world_size - 1, size_send_left, size_send_right, size_disabled, size_recv_left, size_recv_right);

  // Particle m_particle = {0, 0, 0, 0, world_rank};
  // Particle l_particle = {0, 0, 0, 0, 10};
  // Particle r_particle = {0, 0, 0, 0, 10};

  // if (world_rank == 0)
  // {
  //   MPI_Send(&m_particle, 1, mpi_particle_type, 1, 0, MPI_COMM_WORLD);
  // }
  // else
  // {
  //   MPI_Recv(&l_particle, 1, mpi_particle_type, 0, 0, MPI_COMM_WORLD, &status);
  // }

  // if (world_rank == 1)
  // {
  //   MPI_Ssend(&m_particle, 1, mpi_particle_type, world_rank - 1, 0, MPI_COMM_WORLD);
  // }
  // else
  // {
  //   MPI_Recv(&r_particle, 1, mpi_particle_type, world_rank + 1, 0, MPI_COMM_WORLD, &status);
  // }

  // printf("I am rank %d/%d and my particles are:\n\tMy particle: %d\n\tParticle from the left: %d\n\tParticle from the right: %d\n ",
  //        world_rank, world_size - 1, m_particle.index, l_particle.index, r_particle.index);

  MPI_Finalize();

  // // MPI_Ssend ......
  // // particles_left
  // // &particles_left[0] pointer to 0-th element
  // // https://en.cppreference.com/w/cpp/container/vector/clear: particles_left.clear()
  // layer.dump_WA();
  return 0;
}
