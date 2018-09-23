#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "types.hpp"
#include <random>

class UnifDist {
public:
  UnifDist();
  real_t operator()();

private:
  std::uniform_real_distribution<real_t> dis;
  std::mt19937 gen;
};

#endif
