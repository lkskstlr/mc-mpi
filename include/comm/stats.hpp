#ifndef STATS_HPP
#define STATS_HPP

// I use the fact that structs with methods are still POD in memory which is
// guranteed by the standard. See:
// https://stackoverflow.com/questions/422830/structure-of-a-c-object-in-memory-vs-a-struct

#include <mpi.h>
#include <stdio.h>

class Stats {
public:
  enum MPI : int { Send = 0, Recv, STATE_COUNT };

  typedef struct state_tag {
    int nb_mpi[MPI::STATE_COUNT] = {0};

    friend state_tag operator+(state_tag const &lhs, state_tag const &rhs);
    int sprintf(char *str);
    static int sprintf_header(char *str);
    static int sprintf_max_len();
    static MPI_Datatype mpi_t();
  } State;

  State reset();

  void increment(MPI mpi);

private:
  State state;
};

#endif