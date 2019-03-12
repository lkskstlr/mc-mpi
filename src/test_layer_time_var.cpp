#include "layer.hpp"
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char const *argv[])
{
    using std::chrono::high_resolution_clock;

    int nb_particles = argc >= 2 ? atoi(argv[1]) : 10000;

    int nthread = -1;
    int nb_repeats = 100;

    constexpr real_t x_min = 0.0;
    constexpr real_t x_max = 1.0;
    const real_t x_ini = sqrtf(2.0) / 2.0;
    constexpr int world_size = 1;
    constexpr int world_rank = 0;
    constexpr int nb_cells_per_layer = 1000;

    constexpr real_t particle_min_weight = 0.0;

    for (int j = 0; j < nb_repeats; j++)
    {
        Layer layer(decompose_domain(x_min, x_max, x_ini, world_size, world_rank,
                                     nb_cells_per_layer, nb_particles,
                                     particle_min_weight, 33 * j + 17));
        auto start = high_resolution_clock::now();
        layer.simulate(-1, nthread);
        auto finish = high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = finish - start;
        printf("%f\n", elapsed.count() / 1e3);
    }

    return 0;
}
