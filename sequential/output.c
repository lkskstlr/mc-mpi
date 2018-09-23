/**
 * INF560 - MC
 */
#include <stdio.h>

#include "domaine.h"
#include "ens_partic.h"
#include "random.h"

void output_domaine(
        domaine_t *domaine,
        /* IN */
        real_t *c_var,
        const char *var_file
        )
{
    FILE *fd = fopen(var_file, "w");

    int ic ;
    for(ic = 0 ; ic < domaine->nb_couches ; ic++)
    {
        double x = calc_coord_face(domaine, ic) + 0.5*domaine->coeff_coord;
        fprintf(fd, "%.4e %.4e\n", x, c_var[ic]);
    }

    fclose(fd);
}
