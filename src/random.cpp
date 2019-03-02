/**
 * INF560 - MC
 */
#include "random.hpp"

/* REAL RNG */

static const seed_t RNG_G = (seed_t)(6364136223846793005ull);
static const seed_t RNG_C = (seed_t)(1442695040888963407ull);
static const seed_t RNG_P = (seed_t)(1) << 63;

real_t rnd_real(seed_t *seed) {
  real_t inv_RNG_P = (real_t)(1) / (real_t)(RNG_P);
  *seed = (RNG_G * *seed + RNG_C) % RNG_P;
  return (real_t)(*seed) * inv_RNG_P;
}

/* SEED RNG */

static const seed_t RNGS_G = (seed_t)(5177284530976225183ull);
static const seed_t RNGS_C = (seed_t)(2096348467109453893ull);
static const seed_t RNGS_P = (seed_t)(1) << 63;

seed_t rnd_seed(seed_t *seed) {
  // unused
  // real_t inv_RNGS_P = (real_t)(1) / (real_t)(RNGS_P);
  *seed = (RNGS_G * *seed + RNGS_C) % RNGS_P;
  return *seed;
}
