#include "logging.h"

int main(int argc, char const *argv[]) {
  int world_rank = 1;
  MCMPI_DEBUG_INIT(world_rank)
  MCMPI_DEBUG("First Debug print %d", world_rank)
  MCMPI_DEBUG_STOP()
  return world_rank;
}