#include "layer.hpp"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int compareFiles(FILE *fp1, FILE *fp2) {
  char ch1 = getc(fp1);
  char ch2 = getc(fp2);

  // iterate loop till end of file
  while (ch1 != EOF && ch2 != EOF) {
    if (ch1 != ch2) {
      fclose(fp1);
      fclose(fp2);
      printf("ERROR: Output files are not identical!\n");
      return 1;
    }

    // fetching character until end of file
    ch1 = getc(fp1);
    ch2 = getc(fp2);
  }

  if (!(ch1 == EOF && ch2 == EOF)) {
    fclose(fp1);
    fclose(fp2);
    printf("ERROR: Output files are not identical!\n");
    return 1;
  }

  return 0;
}

int main(int argc, char const *argv[]) {
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

  if (fp1 == NULL || fp2 == NULL) {
    printf("Error : Couldn't open files\n");
    remove("WA.out");
    exit(1);
  }

  int result = compareFiles(fp1, fp2);

  // closing both file
  fclose(fp1);
  fclose(fp2);
  remove("WA.out");

  return result;
}