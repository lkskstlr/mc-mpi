#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "types.hpp"
#include <random>

#define SOME_SEED 922987996

class UnifDist {
public:
  UnifDist(std::uint_fast32_t seed);
  real_t operator()();

private:
  std::uniform_real_distribution<real_t> dis;
  std::mt19937 gen;
};

#endif
