#include "random.hpp"

#define SEED 922987996
#define N_DISCARD 10000

UnifDist::UnifDist() {
  // generator
  std::minstd_rand0 lc_generator(SEED);
  std::uint_least32_t seed_data[std::mt19937::state_size];

  std::generate_n(seed_data, std::mt19937::state_size, std::ref(lc_generator));
  std::seed_seq q(std::begin(seed_data), std::end(seed_data));

  std::mt19937 gen{q};
  gen.discard(N_DISCARD);

  // distribution
  dis = std::uniform_real_distribution<real_t>(0.0, 1.0);
}

real_t UnifDist::operator()() { return dis(gen); }
