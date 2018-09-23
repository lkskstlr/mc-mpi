/**
 * INF560 - MC
 */
#ifndef __TYPES_MC_H
#define __TYPES_MC_H

#include <float.h>

#define bool int
#define true 1
#define false 0

typedef unsigned long long seed_t;


#define MC_SIMPLE_PRECIS
//#define MC_DOUBLE_PRECIS

#ifdef MC_SIMPLE_PRECIS

  typedef float real_t;
  // #define MAXREAL MAXFLOAT
  #define MAXREAL FLT_MAX
  #define EPS_PRECIS 1e-4F

#endif

#ifdef MC_DOUBLE_PRECIS
  #error "MC_DOUBLE_PRECIS A IMPLEMENTER"
#endif

#endif

