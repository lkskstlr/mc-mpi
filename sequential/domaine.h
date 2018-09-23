/**
 * INF560 - MC
 */
#ifndef __DOMAINE_H
#define __DOMAINE_H

#include <stdlib.h>

#include "types_mc.h"


struct domaine_t
{
  real_t xmin;
  real_t xmax;

  int nb_couches;
  int nc_ini;

  real_t absorption_rate;
  real_t interaction_rate;


  /* implementation */
  real_t coeff_coord;
};

typedef struct domaine_t domaine_t ;

void init_domaine(
  /* IN */
  real_t     xmin,
  real_t     xmax,
  int        nb_couches,
  real_t     xini,
  /* OUT */
  domaine_t *domaine
);

real_t calc_coord_face(domaine_t *dom, int i_f);

void alloc_array_couches_real_t( domaine_t * domaine, real_t ** c_arr ) ;
void free_array_couches_real_t( real_t ** c_arr ) ;

#endif
