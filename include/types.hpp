#ifndef TYPES_HPP
#define TYPES_HPP

#include <limits>

#if !defined MC_SIMPLE_PRECISION && !defined MC_DOUBLE_PRECISION
#define MC_SIMPLE_PRECIS
#endif

#ifdef MC_SIMPLE_PRECIS

typedef float real_t;
constexpr real_t MAXREAL = std::numeric_limits<real_t>::max();
#define EPS_PRECISION 1e-4F

#endif

#ifdef MC_DOUBLE_PRECISION
#error "MC_DOUBLE_PRECISION NOT IMPLEMENTED"
#endif

/* Is POD to be easily send over MPI */
typedef struct particle_tag {
public:
  real_t x;
  real_t mu;
  real_t wmc;
  int index; /** Cell index of the particle. This must be inside the data
                structure. If x \approx y, where y is the boundary between two
                cells, it is hard to tell in which cell the particle is based on
                floating point inaccuracies. */
} Particle;

#endif
