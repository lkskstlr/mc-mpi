#include <stdio.h>
#include <types.hpp>

typedef struct particle_tag {
public:
  seed_t seed;
  real_t x;
  real_t mu;
  // real_t wmc;
  int index; /** Cell index of the particle. This must be inside the data
                structure. If x \approx y, where y is the boundary between two
                cells, it is hard to tell in which cell the particle is based on
                floating point inaccuracies. */
} Particle;

int main(int argc, char const *argv[]) {
  printf("sizeof(Particle) = %zu\n", sizeof(Particle));
  printf("sizes = 12 + %zu\n", sizeof(seed_t));
  printf("offsetof(index) = %zu\n", offsetof(Particle, index));
  return 0;
}