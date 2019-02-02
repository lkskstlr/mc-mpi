#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>
#include "layer.hpp"

void dump_weights_absorbed(int world_rank, int world_size, Layer &layer, int total_len, int const *displs,
                           real_t const *weights)
{
  FILE *file;
  file = fopen("weights.csv", "w");
  if (!file)
  {
    fprintf(stderr, "Couldn't open file weights.csv for writing.\n");
    exit(1);
  }

  fprintf(file, "proc, x, weight\n");
  int proc = 0;
  real_t x_pos = layer.dx;
  for (int i = 0; i < total_len; ++i)
  {
    if (proc < world_size - 1 && displs[proc + 1] == i)
    {
      proc++;
    }
    fprintf(file, "%d, %.18e, %.18e\n", proc, layer.dx * (i + 0.5),
            weights[i] / layer.dx);
  }

  fclose(file);
}

void gather_weights_absorbed(int world_rank, int world_size, Layer &layer, int *total_len, int **displs,
                             real_t **weights)
{
  int *recvcounts = NULL;

  if (world_rank == 0)
  {
    recvcounts = (int *)malloc(world_size * sizeof(int));
  }

  int my_len = static_cast<int>(layer.weights_absorbed.size());
  MPI_Gather(&my_len, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  *total_len = 0;

  if (world_rank == 0)
  {
    *displs = (int *)malloc(world_size * sizeof(int));

    (*displs)[0] = 0;
    *total_len += recvcounts[0];

    for (int i = 1; i < world_size; i++)
    {
      *total_len += recvcounts[i];
      (*displs)[i] = (*displs)[i - 1] + recvcounts[i - 1];
    }

    *weights = (real_t *)malloc((*total_len) * sizeof(real_t));
  }

  MPI_Gatherv(layer.weights_absorbed.data(), my_len, MCMPI_REAL_T, *weights,
              recvcounts, *displs, MCMPI_REAL_T, 0, MPI_COMM_WORLD);
}

void dump(int world_rank, int world_size, Layer &layer)
{
  int total_len = 0;
  int *displs = NULL;
  real_t *weights = NULL;

  gather_weights_absorbed(world_rank, world_size, layer, &total_len, &displs, &weights);

  if (world_rank == 0)
  {
    dump_weights_absorbed(world_rank, world_size, layer, total_len, displs, weights);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
  int world_size;
  int world_rank;

  constexpr real_t x_min = 0.0;
  constexpr real_t x_max = 1.0;
  const real_t x_ini = sqrtf(2.0) / 2.0;
  constexpr int nb_cells_per_layer = 100;
  constexpr int nb_particles = 1000000;
  constexpr real_t particle_min_weight = 1e-12;

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

  int size_send_right;
  int size_send_left;
  int size_recv_right, size_recv_left, i = 0;
  int unfinished_flag = 1;
  int disabled_own, disabled_all;

  while (unfinished_flag)
  {
    layer.simulate(100000000, particles_send_left, particles_send_right, particles_disabled);
    size_send_right = particles_send_right.size();
    size_send_left = particles_send_left.size();
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
      {
        MPI_Ssend(&particles_send_left[0], size_send_left, mpi_particle_type, world_rank - 1, 0, MPI_COMM_WORLD);
        particles_send_left.clear();
      }
    }
    else if ((world_rank + 1 < world_size) && size_recv_right)
    {
      particles_recv_right.resize(size_recv_right);
      MPI_Recv(&particles_recv_right[0], particles_recv_right.size(), mpi_particle_type, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      layer.particles.insert(layer.particles.end(), particles_recv_right.begin(), particles_recv_right.end());
      particles_recv_right.clear();
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
      {
        MPI_Ssend(&particles_send_left[0], size_send_left, mpi_particle_type, world_rank - 1, 0, MPI_COMM_WORLD);
        particles_send_left.clear();
      }
    }
    else if ((world_rank + 1 < world_size) && size_recv_right)
    {
      particles_recv_right.resize(size_recv_right);
      MPI_Recv(&particles_recv_right[0], particles_recv_right.size(), mpi_particle_type, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      layer.particles.insert(layer.particles.end(), particles_recv_right.begin(), particles_recv_right.end());
      particles_recv_right.clear();
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
      {
        MPI_Ssend(&particles_send_right[0], size_send_right, mpi_particle_type, world_rank + 1, 0, MPI_COMM_WORLD);
        particles_send_right.clear();
      }
    }
    else if (world_rank && size_recv_left)
    {
      particles_recv_left.resize(size_recv_left);
      MPI_Recv(&particles_recv_left[0], particles_recv_left.size(), mpi_particle_type, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      layer.particles.insert(layer.particles.end(), particles_recv_left.begin(), particles_recv_left.end());
      particles_recv_left.clear();
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
      {
        MPI_Ssend(&particles_send_right[0], size_send_right, mpi_particle_type, world_rank + 1, 0, MPI_COMM_WORLD);
        particles_send_right.clear();
      }
    }
    else if (size_recv_left)
    {
      particles_recv_left.resize(size_recv_left);
      MPI_Recv(&particles_recv_left[0], particles_recv_left.size(), mpi_particle_type, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      layer.particles.insert(layer.particles.end(), particles_recv_left.begin(), particles_recv_left.end());
      particles_recv_left.clear();
    }

    disabled_own = particles_disabled.size();
    if (world_rank == 0)
      disabled_own += size_send_left;
    else if (world_rank + 1 == world_size)
      disabled_own += size_send_right;
    MPI_Allreduce(&disabled_own, &disabled_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (disabled_all == nb_particles)
      unfinished_flag = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(10000 * world_rank);
    if (world_rank == 0)
      printf("\n\n========== %2d ==========\n", i);
    printf("rank = %2d, disabled_own = %6d, disabled_all = %6d\n", world_rank, disabled_own, disabled_all);
    usleep(100000);
    i++;
  }

  dump(world_rank, world_size, layer);
  // MPI_Barrier(MPI_COMM_WORLD);
  // usleep(10000 * world_rank);
  // printf("Rank: %d/%d\n\t%d particles EXITED my domain to the LEFT\n\t%d particles EXITED my domain to the RIGHT\n\t%d particles got DISABLED in my domain\n\t%d particles ENTERED my domain through the LEFT\n\t%d particles ENTERED my domain through the RIGHT\n ",
  //        world_rank, world_size - 1, size_send_left, size_send_right, particles_disabled.size(), size_recv_left, size_recv_right);

  MPI_Finalize();

  // layer.dump_WA();
  return 0;
}
