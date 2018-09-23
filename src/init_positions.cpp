/**
 * INF560 - MC
 */
#include "types_mc.h"
#include "domaine.h"
#include "ens_partic.h"

void init_positions(
  ens_partic_t *ens_partic,
  domaine_t *domaine,
  /* OUT */
  real_t *p_x,
  int    *p_nc
)
{
  int ip ;
  for(ip = 0 ; ip < ens_partic->nb_partics ; ip++)
  {
    // Initially, one source of particles

    p_x[ip]  = ens_partic->xini;
    p_nc[ip] = domaine->nc_ini;
  }
}

