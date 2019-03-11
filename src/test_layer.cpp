#include "layer.hpp"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FLOAT_EPS (1e-6)

void compareFiles(FILE *fp1, FILE *fp2)
{
  char ch1 = getc(fp1);
  char ch2 = getc(fp2);

  // iterate loop till end of file
  while (ch1 != EOF && ch2 != EOF)
  {
    if (ch1 != ch2)
    {
      fclose(fp1);
      fclose(fp2);
      printf("ERROR: Output files are not identical!\n");
      exit(EXIT_FAILURE);
    }

    // fetching character until end of file
    ch1 = getc(fp1);
    ch2 = getc(fp2);
  }

  if (!(ch1 == EOF && ch2 == EOF))
  {
    fclose(fp1);
    fclose(fp2);
    printf("ERROR: Output files are not identical!\n");
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char const *argv[])
{
  constexpr real_t x_min = 0.0;
  constexpr real_t x_max = 1.0;
  const real_t x_ini = sqrtf(2.0) / 2.0;
  constexpr int world_size = 1;
  constexpr int world_rank = 0;
  constexpr int nb_cells_per_layer = 100;
  constexpr int nb_particles = 100;
  constexpr real_t particle_min_weight = 0.0;

  Layer layer(decompose_domain(x_min, x_max, x_ini, world_size, world_rank,
                               nb_cells_per_layer, nb_particles,
                               particle_min_weight));

  layer.simulate(-1); // simulate until end
  layer.dump_WA();

  // Compare
  FILE *fp1 = fopen("../data/test_layer_target_WA.out", "r");
  FILE *fp2 = fopen("WA.out", "r");

  if (fp1 == NULL || fp2 == NULL)
  {
    printf("Error : Couldn't open files\n");
    remove("WA.out");
    exit(1);
  }

  compareFiles(fp1, fp2);

  // closing both file
  fclose(fp1);
  fclose(fp2);
  remove("WA.out");

#ifdef CUDA_ENABLED
  /* Also test CUDA version */
  Layer layer_cuda(decompose_domain(x_min, x_max, x_ini, world_size, world_rank,
                                    nb_cells_per_layer, nb_particles,
                                    particle_min_weight));

  layer_cuda.simulate(-1, -1, true);
  float max_diff = 0.0;
  for (int j = 0; j < layer.weights_absorbed.size(); j++)
  {
    float curr_diff = fabs(layer.weights_absorbed[j] - layer_cuda.weights_absorbed[j]);
    if (curr_diff > max_diff)
      max_diff = curr_diff;
  }
  printf("DIFF = %.10f\n", max_diff);
  if (max_diff >= FLOAT_EPS)
    exit(EXIT_FAILURE);
#endif

  return 0;
}
