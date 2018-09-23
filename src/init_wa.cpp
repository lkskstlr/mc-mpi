/**
 * INF560 - MC
 */
#include "domaine.h"

void init_wa(
        domaine_t *domaine,
        /* OUT */
        real_t *c_wa
        )
{
    int ic ;
    for(ic = 0 ; ic < domaine->nb_couches ; ic++)
    {
        c_wa[ic] = 0;
    }
}

