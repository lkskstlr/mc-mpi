#ifndef TYPES_HPP
#define TYPES_HPP

#include <limits>

#if !defined MC_SIMPLE_PRECISION && !defined MC_DOUBLE_PRECISION
#define MC_SIMPLE_PRECIS
#endif

#ifdef MC_SIMPLE_PRECIS

typedef float real_t;
// #define MAXREAL MAXFLOAT
constexpr real_t MAXREAL = std::numeric_limits<real_t>::max();
#define EPS_PRECISION 1e-4F

#endif

#ifdef MC_DOUBLE_PRECISION
#error "MC_DOUBLE_PRECISION NOT IMPLEMENTED"
#endif

#endif