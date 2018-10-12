/**
 * INF560 - MC
 */
#ifndef __RANDOM_H
#define __RANDOM_H

#include "types.hpp"

/*!
 * \function rnd_real
 *
 * \brief Randomly choose a real between 0.0 and 1.0
 *
 * \param[inout] seed Current seed for RNG
 *
 * \return Real between 0.0 and 1.0
 */
real_t rnd_real(seed_t *seed);

/*!
 * \function rnd_seed
 *
 * \brief Generate random seed
 *
 * \param[inout] seed Current seed used by RNG
 *
 * \return Seed
 */
seed_t rnd_seed(seed_t *seed);

#endif
