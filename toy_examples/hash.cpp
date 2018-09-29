#include "types.hpp"
#include <functional>
#include <iostream>

int hash(char const *data, int len) {
  unsigned int hash = 0x811c9dc5;
  unsigned int prime = 16777619;
  for (int i = 0; i < len; ++i) {
    hash = hash ^ data[i];
    hash = hash * prime;
  }

  int res = *((int *)(&hash));
  return res;
}

int main() {

  Particle p = {0.0, 1.0, 0.5, 2};
  Particle p2 = {0.0, 1.0, 0.5, 3};
  Particle p3;
  std::cout << "hash = " << hash((char *)&p, sizeof(p)) << std::endl;
  std::cout << "hash = " << hash((char *)&p2, sizeof(p2)) << std::endl;
  std::cout << "hash = " << hash((char *)&p3, sizeof(p3)) << std::endl;
}