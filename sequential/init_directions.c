/**
 * INF560 - MC
 */
#include "types_mc.h"
#include "random.h"
#include "ens_partic.h"

void init_directions(
  ens_partic_t *ens_partic,
  /* INOUT */
  seed_t *p_sd,
  /* OUT */
  real_t *p_mu
)
{
  int ip ;
  for(ip = 0 ; ip < ens_partic->nb_partics ; ip++)
  {
    /* We randomly generate code(theta) in [-1; 1]
     * It is equivalent to generate a direction
     */

    p_mu[ip] = 2*rnd_real(&p_sd[ip]) - 1;
  }
}

