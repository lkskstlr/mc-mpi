#ifndef TYPES_HPP
#define TYPES_HPP

#include <limits>
#include <mpi.h>

#if !defined MC_SIMPLE_PRECISION && !defined MC_DOUBLE_PRECISION
#define MC_SIMPLE_PRECIS
#endif

#ifdef MC_SIMPLE_PRECIS
#define MCMPI_REAL_T MPI_FLOAT
typedef float real_t;
constexpr real_t MAXREAL = std::numeric_limits<real_t>::max();
#define EPS_PRECISION 1e-4F

#endif

#ifdef MC_DOUBLE_PRECISION
#error "MC_DOUBLE_PRECISION NOT IMPLEMENTED"
#endif

#endif
