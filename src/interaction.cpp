/**
 * INF560 - MC
 */
#include "types_mc.h"
#include "ens_partic.h"
#include "random.h"

void interaction(
  ens_partic_t *ens_partic,
  /* IN */
  bool   *p_enable,
  real_t *p_di,
  int    *p_ev,
  /* INOUT */
  seed_t *p_sd,
  real_t *p_x,
  real_t *p_mu
)
{
  int ip ;
  int max = ens_partic->nb_partics ;

  for(ip = 0 ; ip < max ; ip++)
  {
    if (p_enable[ip])
    {
      if (p_ev[ip] == -1 /* code for interaction event */)
      {
        /* Move particle inside the cell */
        p_x[ip] += p_mu[ip] * p_di[ip];

        /* Choose a new direction */
        p_mu[ip] = 2*rnd_real(&p_sd[ip]) - 1;
      }
    }
  }
}

