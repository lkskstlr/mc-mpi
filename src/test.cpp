#include "geometry.hpp"
#include "random.hpp"
#include <iostream>

using std::cout, std::endl;

int main(int argc, char const *argv[]) {

  // Test Random
  UnifDist dist = UnifDist();
  cout << dist() << ", " << dist() << ", " << dist() << endl;
  cout << "sizeof(UnifDist) = " << sizeof(dist) << endl;

  // Test Layer
  Layer layer(0.0, 1.0);
  cout << "sizeof(Particle) = " << sizeof(Particle) << endl;
  cout << "Layer: n = " << layer.n << ", x_min = " << layer.x_min
       << ", x_max = " << layer.x_max << endl;
  // cout << dist() << ", " << dist() << ", " << dist() << endl;
  layer.create_particles(dist, 0.5, 20);
  cout << dist() << ", " << dist() << ", " << dist() << endl;

  cout << "Particles:" << endl;
  for (auto const &particle : layer.particles) {
    cout << "  x = " << particle.x << ", mu = " << particle.mu << endl;
  }
  return 0;
}
