/**
 * INF560 - MC
 */
#ifndef __ENS_PARTIC_H
#define __ENS_PARTIC_H

#include <stdlib.h>

#include "types_mc.h"

struct ens_partic_t
{
  real_t xini; /* Initial position of all particles */
  int nb_partics; /* Total number of particles */
};

typedef struct ens_partic_t ens_partic_t ;

/* In init_domaine.c */

/* real_t */
void alloc_array_particles_real_t(ens_partic_t *ens_partic, real_t ** p_arr ) ;
void free_array_particles_real_t( real_t ** p_arr ) ;
/* int */
void alloc_array_particles_int(ens_partic_t *ens_partic, int ** p_arr ) ;
void free_array_particles_int( int ** p_arr ) ;
/* seed_t */
void alloc_array_particles_seed_t(ens_partic_t *ens_partic, seed_t** p_arr ) ;
void free_array_particles_seed_t( seed_t** p_arr ) ;
/* bool */
void alloc_array_particles_bool(ens_partic_t *ens_partic, bool ** p_arr ) ;
void free_array_particles_bool( bool ** p_arr ) ;

#endif

