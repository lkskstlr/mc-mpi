/**
 * INF560 - MC
 */
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#include "domaine.h"
#include "ens_partic.h"
#include "prototypes.h"
#include "random.h"
#include "types_mc.h"

#define DEFAULT_NB_OF_PARTICLES 1000000
#define DEFAULT_NB_OF_LAYERS 1000000
int main(int argc, char **argv) {
  /* DECLARATIONS */
  struct timeval t1, t2;
  double duration;
  domaine_t domaine;       /* Mesh */
  ens_partic_t ens_partic; /* Information about particles */

  real_t *p_x;    /* Position of particles */
  real_t *p_mu;   /* Direction of each particle */
  int *p_nc;      /* Cell where are located particle */
  real_t *p_di;   /* Distance of next event */
  real_t *p_wmc;  /* Weight of particle */
  seed_t *p_sd;   /* Current seed of particles */
  int *p_ev;      /* Next event */
  bool *p_enable; /* Are particles still alive? */

  real_t *c_sig; /* section efficace */
  real_t *c_wa;  /* Weight of each cell (output) */

  // was DEFAULT_NB_OF_CELLS before. Didn't compile
  int nb_couches = DEFAULT_NB_OF_LAYERS;
  int nb_partics = DEFAULT_NB_OF_PARTICLES;

  /* ARGUMENTS */
  if (argc != 3) {
    fprintf(stderr, "Usage: %s nb_cells nb_particles\n", argv[0]);
    exit(1);
  }

  nb_couches = atoi(argv[1]);
  nb_partics = atoi(argv[2]);

  if (nb_couches <= 0) {
    fprintf(stderr, "Wrong number of cells: %d\n", nb_couches);
    exit(1);
  }
  if (nb_partics <= 0) {
    fprintf(stderr, "Wrong number of particles: %d\n", nb_partics);
    exit(1);
  }

  printf("MC: Running on 1D-mesh of size %d w/ %d particles\n", nb_couches,
         nb_partics);

  /* STRUCTURE INITIALIZATION */
  ens_partic.xini = sqrt(2.) / 2;
  ens_partic.nb_partics = nb_partics;

  init_domaine(0., 1., nb_couches, ens_partic.xini, &domaine);

  /* ALLOCATIONS */
  alloc_array_particles_real_t(&ens_partic, &p_x);
  alloc_array_particles_real_t(&ens_partic, &p_mu);
  alloc_array_particles_int(&ens_partic, &p_nc);
  alloc_array_particles_real_t(&ens_partic, &p_di);
  alloc_array_particles_real_t(&ens_partic, &p_wmc);
  alloc_array_particles_seed_t(&ens_partic, &p_sd);
  alloc_array_particles_int(&ens_partic, &p_ev);
  alloc_array_particles_bool(&ens_partic, &p_enable);

  alloc_array_couches_real_t(&domaine, &c_sig);
  alloc_array_couches_real_t(&domaine, &c_wa);

  /* DOMAIN & PARTICLES INITIALIZATIONS */
  init_sig(&domaine, c_sig);
  init_wa(&domaine, c_wa);

  init_graines(&ens_partic, p_sd);
  init_positions(&ens_partic, &domaine, p_x, p_nc);
  init_poids(&ens_partic, p_wmc);
  init_directions(&ens_partic, p_sd, p_mu);
  enable_all_partics(&ens_partic, p_enable);

  /* PARTICLE TRACKING */
  int nb_partics_disb = 0;
  int nb_partics_enbl = ens_partic.nb_partics;
  int iter = 0;

  /* Timer start */
  gettimeofday(&t1, NULL);

  while (nb_partics_enbl > 0) {

    for (int i = 0; i < nb_partics; ++i) {
      printf("%d, ", p_nc[i]);
    }
    printf("\n");

    dist_sortie_couche(&domaine, &ens_partic, p_enable, p_x, p_mu, p_nc, p_di,
                       p_ev);

    dist_interaction(&domaine, &ens_partic, c_sig, p_enable, p_nc, p_sd, p_di,
                     p_ev);

    absorption(&domaine, &ens_partic, c_sig, p_enable, p_nc, p_di, p_wmc, c_wa);

    sortie_couche(&domaine, &ens_partic, p_ev, p_enable, p_nc, p_x,
                  &nb_partics_disb);

    interaction(&ens_partic, p_enable, p_di, p_ev, p_sd, p_x, p_mu);

    nb_partics_enbl -= nb_partics_disb;

#if 0
    if(iter%100==0)
    {
      printf("Nb partics enabled: %d\n", nb_partics_enbl);
    }
#endif

    iter++;
  }

  /* Timer stop */
  gettimeofday(&t2, NULL);

  duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

  printf("MC: simulation done in %g s with %d iteration(s)\n", duration, iter);

  real_t wa_tot = 0;
  int ic;
  for (ic = 0; ic < domaine.nb_couches; ic++) {
    wa_tot += c_wa[ic];
    c_wa[ic] /= domaine.coeff_coord;
  }
  printf("MC: Wa_tot = %g\n", wa_tot);

  /* GRAPHIC OUTPUT */
#ifdef GRAPHICS_ON
  printf("MC: Dumping results in WA.out...\n");
  output_domaine(&domaine, c_wa, "WA.out");
  printf("MC:      Dumping done\n");
#endif

  /* MEMORY DEALLOCATION */
  free_array_particles_real_t(&p_x);
  free_array_particles_real_t(&p_mu);
  free_array_particles_bool(&p_nc);
  free_array_particles_real_t(&p_di);
  free_array_particles_real_t(&p_wmc);
  free_array_particles_seed_t(&p_sd);
  free_array_particles_int(&p_ev);
  free_array_particles_bool(&p_enable);

  free_array_couches_real_t(&c_sig);
  free_array_couches_real_t(&c_wa);

  return 0;
}
