#include "layer.hpp"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void dump_weights_absorbed(int world_rank, int world_size, Layer &layer,
                           int total_len, int const *displs,
                           real_t const *weights) {
  FILE *file;
  file = fopen("weights.csv", "w");
  if (!file) {
    fprintf(stderr, "Couldn't open file weights.csv for writing.\n");
    exit(1);
  }

  fprintf(file, "proc, x, weight\n");
  int proc = 0;
  real_t x_pos = layer.dx;
  for (int i = 0; i < total_len; ++i) {
    if (proc < world_size - 1 && displs[proc + 1] == i) {
      proc++;
    }
    fprintf(file, "%d, %.18e, %.18e\n", proc, layer.dx * (i + 0.5),
            weights[i] / layer.dx);
  }

  fclose(file);
}

void gather_weights_absorbed(int world_rank, int world_size, Layer &layer,
                             int *total_len, int **displs, real_t **weights) {
  int *recvcounts = NULL;

  if (world_rank == 0) {
    recvcounts = (int *)malloc(world_size * sizeof(int));
  }

  int my_len = static_cast<int>(layer.weights_absorbed.size());
  MPI_Gather(&my_len, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  *total_len = 0;

  if (world_rank == 0) {
    *displs = (int *)malloc(world_size * sizeof(int));

    (*displs)[0] = 0;
    *total_len += recvcounts[0];

    for (int i = 1; i < world_size; i++) {
      *total_len += recvcounts[i];
      (*displs)[i] = (*displs)[i - 1] + recvcounts[i - 1];
    }

    *weights = (real_t *)malloc((*total_len) * sizeof(real_t));
  }

  MPI_Gatherv(layer.weights_absorbed.data(), my_len, MCMPI_REAL_T, *weights,
              recvcounts, *displs, MCMPI_REAL_T, 0, MPI_COMM_WORLD);
}

void dump(int world_rank, int world_size, Layer &layer) {
  int total_len = 0;
  int *displs = NULL;
  real_t *weights = NULL;

  gather_weights_absorbed(world_rank, world_size, layer, &total_len, &displs,
                          &weights);

  if (world_rank == 0) {
    dump_weights_absorbed(world_rank, world_size, layer, total_len, displs,
                          weights);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
  int world_size;
  int world_rank;
  // Change later

  int particle_tag = 0;

  constexpr real_t x_min = 0.0;
  constexpr real_t x_max = 1.0;
  const real_t x_ini = sqrtf(2.0) / 2.0;
  constexpr int nb_cells_per_layer = 1000;
  constexpr int nb_particles = 10000000;
  constexpr real_t particle_min_weight = 1e-12;

  int nb_particles_per_cycle = nb_particles / 10;

  // MPI initialization and basic function calls
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  /****************************** MPI_PARTICLE_TYPE **************************/
  MPI_Datatype mpi_particle_type;
  constexpr int nitems = 5;
  int blocklengths[nitems] = {1, 1, 1, 1, 1};
  MPI_Datatype types[nitems] = {MPI_UNSIGNED_LONG_LONG, MPI_FLOAT, MPI_FLOAT,
                                MPI_FLOAT, MPI_INT};
  MPI_Aint offsets[nitems];

  offsets[0] = offsetof(Particle, seed);
  offsets[1] = offsetof(Particle, x);
  offsets[2] = offsetof(Particle, mu);
  offsets[3] = offsetof(Particle, wmc);
  offsets[4] = offsetof(Particle, index);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                         &mpi_particle_type);
  MPI_Type_commit(&mpi_particle_type);
  /***************************************************************************/

  MPI_Status status;

  // Decompose the domain into as many pieces as there are processes
  Layer layer(decompose_domain(x_min, x_max, x_ini, world_size, world_rank,
                               nb_cells_per_layer, nb_particles,
                               particle_min_weight));

  int unfinished_flag = 1;
  int disabled_own, disabled_all;
  int old_size, recv_count, temp = 0, i = 0;

  double starttime = MPI_Wtime();

  while (unfinished_flag) {
    layer.simulate(nb_particles_per_cycle);

    // Prepare the particles vector for the extra ones from the left OR right,
    // which are AT MOST nb_particles_per_cycle
    old_size = layer.particles.size();
    layer.particles.resize(layer.particles.size() + nb_particles_per_cycle);
    /************************** [ODDS] <==> [EVENS]
     * ****************************/
    recv_count = 0;
    if ((world_rank % 2) && (world_rank + 1 < world_size)) {
      // ODD
      MPI_Sendrecv(layer.particles_right.data(), layer.particles_right.size(),
                   mpi_particle_type, world_rank + 1, particle_tag,
                   layer.particles.data() + old_size, nb_particles_per_cycle,
                   mpi_particle_type, world_rank + 1, particle_tag,
                   MPI_COMM_WORLD, &status);
      temp = layer.particles_right.size();
      layer.particles_right.clear();
      MPI_Get_count(&status, mpi_particle_type, &recv_count);
    }
    if (!(world_rank % 2) && (world_rank > 0)) {
      // EVEN
      MPI_Sendrecv(layer.particles_left.data(), layer.particles_left.size(),
                   mpi_particle_type, world_rank - 1, particle_tag,
                   layer.particles.data() + old_size, nb_particles_per_cycle,
                   mpi_particle_type, world_rank - 1, particle_tag,
                   MPI_COMM_WORLD, &status);
      temp = layer.particles_left.size();
      layer.particles_left.clear();
      MPI_Get_count(&status, mpi_particle_type, &recv_count);
    }

    layer.particles.resize(old_size + recv_count + nb_particles_per_cycle);
    old_size += recv_count;

    MPI_Barrier(MPI_COMM_WORLD);
    /************************** [EVENS] <==> [ODDS]
     * ****************************/
    recv_count = 0;
    if (!(world_rank % 2) && (world_rank + 1 < world_size)) {
      // EVEN
      MPI_Sendrecv(layer.particles_right.data(), layer.particles_right.size(),
                   mpi_particle_type, world_rank + 1, particle_tag,
                   layer.particles.data() + old_size, nb_particles_per_cycle,
                   mpi_particle_type, world_rank + 1, particle_tag,
                   MPI_COMM_WORLD, &status);
      temp = layer.particles_right.size();
      layer.particles_right.clear();
      MPI_Get_count(&status, mpi_particle_type, &recv_count);
    }
    if (world_rank % 2) {
      // ODD
      MPI_Sendrecv(layer.particles_left.data(), layer.particles_left.size(),
                   mpi_particle_type, world_rank - 1, particle_tag,
                   layer.particles.data() + old_size, nb_particles_per_cycle,
                   mpi_particle_type, world_rank - 1, particle_tag,
                   MPI_COMM_WORLD, &status);
      temp = layer.particles_left.size();
      layer.particles_left.clear();
      MPI_Get_count(&status, mpi_particle_type, &recv_count);
    }

    layer.particles.resize(old_size + recv_count);

    disabled_own = layer.nb_disabled;
    MPI_Allreduce(&disabled_own, &disabled_all, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (disabled_all == nb_particles)
      unfinished_flag = 0;

    // MPI_Barrier(MPI_COMM_WORLD);
    // usleep(10000 * world_rank);
    // if (world_rank == 0)
    //   printf("\n\n========== %2d ==========\n", i);
    // printf("rank = %2d, active = %7d, disabled_own = %6d, disabled_all =
    // %6d\n", world_rank, (int)layer.particles.size(), disabled_own,
    // disabled_all); usleep(100000);
    i++;
  }

  double endtime = MPI_Wtime();

  printf("%d, time = %f\n", world_rank, endtime - starttime);

  dump(world_rank, world_size, layer);
  // MPI_Barrier(MPI_COMM_WORLD);
  // usleep(10000 * world_rank);
  // printf("Rank: %d/%d\n\t%d particles EXITED my domain to the LEFT\n\t%d
  // particles EXITED my domain to the RIGHT\n\t%d particles got DISABLED in my
  // domain\n\t%d particles ENTERED my domain through the LEFT\n\t%d particles
  // ENTERED my domain through the RIGHT\n ",
  //        world_rank, world_size - 1, size_send_left, size_send_right,
  //        particles_disabled.size(), size_recv_left, size_recv_right);

  MPI_Finalize();

  // layer.dump_WA();
  return 0;
}
