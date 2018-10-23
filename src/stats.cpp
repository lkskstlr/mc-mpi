#include "stats.hpp"
#include <mpi.h>
#include <string>

MPI_Datatype Stats::State::mpi_t() {
  constexpr int nitems = 1;
  MPI_Datatype mpi_state_type;
  int blocklengths[nitems] = {MPI::STATE_COUNT};
  MPI_Datatype types[nitems] = {MPI_INT};
  MPI_Aint offsets[nitems];

  offsets[0] = offsetof(State, nb_mpi);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_state_type);
  return mpi_state_type;
}

Stats::State operator+(Stats::State const &lhs, Stats::State const &rhs) {
  Stats::State res;
  for (int i = 0; i < Stats::MPI::STATE_COUNT; ++i) {
    res.nb_mpi[i] = lhs.nb_mpi[i] + rhs.nb_mpi[i];
  }

  return res;
}

int Stats::State::sprintf(char *str) {
  return ::sprintf(str, "%d, %d, ", nb_mpi[MPI::Send], nb_mpi[MPI::Recv]);
}

int Stats::State::sprintf_header(char *str) {
  return ::sprintf(str, "%s, %s, ", "stats_nb_send", "stats_nb_recv");
}

int Stats::State::sprintf_max_len() { return 2 * 20; }

Stats::State Stats::reset() {
  State res = state;
  state = State();
  return res;
}

void Stats::increment(MPI mpi) { state.nb_mpi[mpi]++; }
