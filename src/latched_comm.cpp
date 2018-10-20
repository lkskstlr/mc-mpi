#include "latched_comm.hpp"

using std::vector;

LatchedComm::LatchedComm(int world_size, int world_rank, int tag, int init_msg)
    : world_size(world_size), tag(tag), last_msg(init_msg), my_msg(init_msg),
      vec_msgs(world_size * static_cast<int>(!world_rank), init_msg) {

  init(world_rank, MPI_INT, 8192);
}

bool LatchedComm::send(int msg) {
  if (msg == last_msg) {
    return false;
  }

  last_msg = msg;
  AsyncComm<int>::send(msg, 0, tag);

  return true;
}

bool LatchedComm::bcast(int msg) {
  if (msg == last_msg) {
    return false;
  }

  last_msg = msg;
  for (int i = 1; i < world_size; ++i) {
    AsyncComm<int>::send(msg, i, tag);
  }

  return true;
}

int LatchedComm::msg() {
  MPI_Status status;
  int flag;

  do {
    // Probe
    MPI_Iprobe(0, tag, MPI_COMM_WORLD, &flag, &status);

    if (flag) {
      // Recv
      MPI_Recv(&my_msg, 1, mpi_t, status.MPI_SOURCE, tag, MPI_COMM_WORLD,
               MPI_STATUSES_IGNORE);
    }
  } while (flag);

  return my_msg;
}

vector<int> const &LatchedComm::msgs() {
  MPI_Status status;
  int flag;

  do {
    // Probe
    MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &status);

    if (flag) {
      // Recv
      MPI_Recv(&(vec_msgs[status.MPI_SOURCE]), 1, mpi_t, status.MPI_SOURCE, tag,
               MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
    }
  } while (flag);

  return vec_msgs;
}