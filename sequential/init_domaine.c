/**
 * INF560 - MC
 */
#include "domaine.h"
#include "types_mc.h"
#include "ens_partic.h"

void init_domaine(
  /* IN */
  real_t     xmin,
  real_t     xmax,
  int        nb_couches,
  real_t     xini,
  /* OUT */
  domaine_t *domaine
)
{
  domaine->xmin = xmin;
  domaine->xmax = xmax;

  domaine->nb_couches = nb_couches;

  domaine->coeff_coord = (xmax - xmin)/nb_couches;

  domaine->nc_ini = (int)((xini - xmin) / domaine->coeff_coord);

  domaine->absorption_rate = 0.5;
  domaine->interaction_rate = 1 - domaine->absorption_rate;
}

real_t calc_coord_face(domaine_t *dom, int i_f)
{
  return dom->xmin + i_f * dom->coeff_coord;
}

void alloc_array_couches_real_t( domaine_t * domaine, real_t ** c_arr )
{
  *c_arr = (real_t *)malloc( domaine->nb_couches * sizeof( real_t ) ) ;
}

void free_array_couches_real_t( real_t ** c_arr )
{
  free( *c_arr ) ;
  *c_arr = NULL ;
}

/* real_t */
void alloc_array_particles_real_t(ens_partic_t *ens_partic, real_t ** p_arr )
{
  *p_arr = (real_t *)malloc( ens_partic->nb_partics * sizeof( real_t ) ) ;
}

void free_array_particles_real_t( real_t ** p_arr )
{
  free( *p_arr ) ;
  *p_arr = NULL ;
}

/* int */
void alloc_array_particles_int(ens_partic_t *ens_partic, int ** p_arr )
{
  *p_arr = (int *)malloc( ens_partic->nb_partics * sizeof( int ) ) ;
}

void free_array_particles_int( int ** p_arr )
{
  free( *p_arr ) ;
  *p_arr = NULL ;
}

/* seed_t */
void alloc_array_particles_seed_t(ens_partic_t *ens_partic, seed_t** p_arr )
{
  *p_arr = (seed_t *)malloc( ens_partic->nb_partics * sizeof( seed_t ) ) ;
}

void free_array_particles_seed_t( seed_t** p_arr )
{
  free( *p_arr ) ;
  *p_arr = NULL ;
}

/* bool */
void alloc_array_particles_bool(ens_partic_t *ens_partic, bool ** p_arr )
{
  *p_arr = (bool *)malloc( ens_partic->nb_partics * sizeof( bool ) ) ;
}

void free_array_particles_bool( bool ** p_arr )
{
  free( *p_arr ) ;
  *p_arr = NULL ;
}
