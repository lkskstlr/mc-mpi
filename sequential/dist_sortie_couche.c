/**
 * INF560 - MC
 */
#include "domaine.h"
#include "ens_partic.h"
#include "types_mc.h"
#include <stdio.h>

void dist_sortie_couche(domaine_t *domaine, ens_partic_t *ens_partic,
                        bool *p_enable, real_t *p_x, real_t *p_mu, int *p_nc,
                        real_t *p_di, int *p_ev) {
  int ip;
  int max = ens_partic->nb_partics;
  for (ip = 0; ip < max; ip++) {
    if (p_enable[ip]) {
      // cos(theta) in [-1, +1]
      const real_t mu = p_mu[ip];

      real_t di = MAXREAL;
      int ev = 0;

      if (mu < -EPS_PRECIS || EPS_PRECIS < mu) {
        // i_f: index of exit edge
        const int i_f = p_nc[ip] + (mu < 0 ? 0 : 1);

        const real_t xf = calc_coord_face(domaine, i_f);

        di = (xf - p_x[ip]) / mu;
        ev = i_f;
      }

      printf(" (%f) ", di);

      p_di[ip] = di;
      p_ev[ip] = ev;
    }
  }
}
