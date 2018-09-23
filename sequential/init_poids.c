/**
 * INF560 - MC
 */
#include "types_mc.h"
#include "ens_partic.h"

void init_poids(
  ens_partic_t *ens_partic,
  /* OUT */
  real_t *p_wmc
)
{
  int ip ;
  for(ip = 0 ; ip < ens_partic->nb_partics ; ip++)
  {
    p_wmc[ip] = (real_t)(1)/(real_t)(ens_partic->nb_partics);
  }
}

