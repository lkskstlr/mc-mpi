/**
 * INF560 - MC
 */
#include "types_mc.h"
#include "ens_partic.h"

void enable_all_partics(
  ens_partic_t *ens_partic,
  /* OUT */
  bool *p_enable
)
{
  int ip ;
  for(ip = 0 ; ip < ens_partic->nb_partics ; ip++)
  {
    p_enable[ip] = true;
  }
}

