/**
 * INF560 - MC
 */
#ifndef _PROTOTYPES_H
#define _PROTOTYPES_H

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
);

void dist_interaction(
  domaine_t    *domaine,
  ens_partic_t *ens_partic,
  /* IN */
  real_t *c_sig,
  bool   *p_enable,
  int    *p_nc,
  /* INOUT */
  seed_t *p_sd,
  /* OUT */
  real_t *p_di,
  int    *p_ev
);

void dist_sortie_couche(
  domaine_t    *domaine,
  ens_partic_t *ens_partic,
  bool   *p_enable,
  real_t *p_x,
  real_t *p_mu,
  int    *p_nc,
  real_t *p_di,
  int    *p_ev
);

void enable_all_partics(
  ens_partic_t *ens_partic,
  /* OUT */
  bool *p_enable
);

void init_directions(
  ens_partic_t *ens_partic,
  /* INOUT */
  seed_t *p_sd,
  /* OUT */
  real_t *p_mu
);

void init_graines(
  ens_partic_t *ens_partic,
  /* OUT */
  seed_t *p_sd
);

void init_poids(
  ens_partic_t *ens_partic,
  /* OUT */
  real_t *p_wmc
);

void init_positions(
  ens_partic_t *ens_partic,
  domaine_t *domaine,
  /* OUT */
  real_t *p_x,
  int    *p_nc
);

void init_sig(
  domaine_t *domaine,
  /* OUT */
  real_t *c_sig
);

void init_wa(
  domaine_t *domaine,
  /* OUT */
  real_t *c_wa
);

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
);

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
);

void output_domaine(
  domaine_t *domaine,
  /* IN */
  real_t *c_var,
  const char *var_file
);

#endif

