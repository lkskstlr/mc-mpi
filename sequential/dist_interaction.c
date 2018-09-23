/**
 * INF560 - MC
 */
#include <math.h>
#include <stdio.h>

#include "domaine.h"
#include "ens_partic.h"
#include "random.h"
#include "types_mc.h"

void dist_interaction(domaine_t *domaine, ens_partic_t *ens_partic,
                      /* IN */
                      real_t *c_sig, bool *p_enable, int *p_nc,
                      /* INOUT */
                      seed_t *p_sd,
                      /* OUT */
                      real_t *p_di, int *p_ev) {
  const real_t i_rate = domaine->interaction_rate;
  int ip;
  int max = ens_partic->nb_partics;

  for (ip = 0; ip < ens_partic->nb_partics; ip++) {
    if (p_enable[ip]) {
      const real_t h = rnd_real(&p_sd[ip]); // h app. a [0, 1]

      const int ic = p_nc[ip]; /* Cell index */
      const real_t sig_i = i_rate * c_sig[ic];
      real_t di = MAXREAL;

      if (sig_i > EPS_PRECIS) {
        di = -log(h) / sig_i;
      }
      // printf(" %f, ", di);
      /* If (dist. interaction < dist. sortie couche)
       * Then event is interaction
       */

      if (di < p_di[ip]) {
        p_di[ip] = di;
        p_ev[ip] = -1;
      }
    }
  }
}
