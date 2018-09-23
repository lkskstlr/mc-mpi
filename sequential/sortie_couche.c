/**
 * INF560 - MC
 */
#include "types_mc.h"
#include "domaine.h"
#include "ens_partic.h"

void sortie_couche(
  domaine_t    *domaine,
  ens_partic_t *ens_partic,
  /* IN */
  int    *p_ev,
  /* INOUT */
  bool   *p_enable,
  int    *p_nc,
  /* OUT */
  real_t *p_x,
  int    *nb_disable
)
{

  int ip ;
  int max = ens_partic->nb_partics ;
  int reduc = 0 ;

  *nb_disable = 0;

  for(ip = 0 ; ip < max ; ip++)
  {
    if (p_enable[ip])
    {
      if (p_ev[ip] >= 0 /* condition for cell exit */)
      {
        /* Index of the cell edge */
        const int i_f = p_ev[ip];

        /* We put the particle exactly on the cell border */
        p_x[ip] = calc_coord_face(domaine, i_f);

        /* We determine the index of new cell
         * Be careful about mesh borders.
         * Particle is lost when reaching the frontiers */

        p_enable[ip] = !(i_f == 0 || i_f == domaine->nb_couches);

        if (p_enable[ip])
        {
          const int old_nc = p_nc[ip];
          p_nc[ip] = (i_f == old_nc ? old_nc-1 : old_nc+1);
        }

        reduc += (int)(!p_enable[ip]);
      }
    }
  }

  *nb_disable = reduc ;

}

