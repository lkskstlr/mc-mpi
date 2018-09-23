/**
 * INF560 - MC
 */
#include "types_mc.h"
#include "random.h"
#include "ens_partic.h"

void init_graines(
  ens_partic_t *ens_partic,
  /* OUT */
  seed_t *p_sd
)
{
  seed_t seed = 5127801;

  int ip ;
  for(ip = 0 ; ip < ens_partic->nb_partics ; ip++)
  {
    p_sd[ip] = rnd_seed(&seed);
  }
}

