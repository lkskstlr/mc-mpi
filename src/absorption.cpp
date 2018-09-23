/**
 * INF560 - MC
 */
#include <math.h>

#include "types_mc.h"
#include "domaine.h"
#include "ens_partic.h"

void absorption(
  domaine_t    *domaine,
  ens_partic_t *ens_partic,
  /* IN */
  real_t *c_sig,
  bool   *p_enable,
  int    *p_nc,
  real_t *p_di,
  /* INOUT */
  real_t *p_wmc,
  /* OUT */
  real_t *c_wa
)
{
  const real_t a_rate = domaine->absorption_rate;
  int ip ;
  int max = ens_partic->nb_partics ;

  for(ip = 0 ; ip < max ; ip++)
  {
    if (p_enable[ip])
    {
      const int ic = p_nc[ip]; // Cell index
      const real_t sig_a = a_rate*c_sig[ic];

      const real_t wmc = p_wmc[ip];
      const real_t di  = p_di[ip];

      const real_t dw = (1 - exp(-sig_a*di)) * wmc;

      /* Weight removed from particle is added to
       * the cell in which the particle is located
       */

      p_wmc[ip] -= dw;
      c_wa[ic]  += dw;
    }
  }
}

