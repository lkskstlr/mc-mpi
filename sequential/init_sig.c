/**
 * INF560 - MC
 */
#include <math.h>

#include "domaine.h"

void init_sig(
  domaine_t *domaine,
  /* OUT */
  real_t *c_sig
)
{
  int ic ;
  for(ic = 0 ; ic < domaine->nb_couches ; ic++)
  {
    double x = calc_coord_face(domaine, ic) + 0.5*domaine->coeff_coord;
    c_sig[ic] = exp(-x);
  }
}

