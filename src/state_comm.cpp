#include "state_comm.hpp"

using std::vector;

StateComm::StateComm(int world_size, int world_rank, int tag,
                     std::function<State(vector<int>)> state_lambda,
                     State init_state, int init_msg)
    : world_size(world_size), tag(tag), state(init_state), last_msg(init_msg),
      vec_msgs(world_size * static_cast<int>(!world_rank), init_msg),
      state_lambda(state_lambda)
{
  init(world_rank, MPI_INT, 8192);
}

bool StateComm::send_msg(int msg)
{
  if (msg == last_msg)
  {
    return false;
  }

  last_msg = msg;
  if (world_rank == 0)
  {
    vec_msgs[0] = msg;
  }
  else
  {
    AsyncComm<int>::send(msg, 0, tag);
  }

  return true;
}

bool StateComm::send_state()
{
  if (world_rank != 0)
  {
    return false;
  }

  // receive current messages
  recv_msgs();
  // printf("vec_msgs = (");
  // for (int i = 0; i < world_size; ++i) {
  //   printf("%d, ", vec_msgs[i]);
  // }
  // printf(")\n");

  // invoke lambda
  State new_state = state_lambda(vec_msgs);
  if (new_state == state)
  {
    // state didn't change
    return false;
  }
  state = new_state;

  for (int i = 1; i < world_size; ++i)
  {
    AsyncComm<int>::send(state, i, tag);
  }

  return true;
}

StateComm::State StateComm::recv_state()
{
  if (world_rank == 0)
  {
    return state;
  }
  MPI_Status status;
  int flag;

  do
  {
    // Probe
    MPI_Iprobe(0, tag, MPI_COMM_WORLD, &flag, &status);

    if (flag)
    {
      // Recv
      MPI_Recv(&state, 1, mpi_t, status.MPI_SOURCE, tag, MPI_COMM_WORLD,
               MPI_STATUSES_IGNORE);
      stats.increment(Stats::MPI::Recv);
    }
  } while (flag);

  return state;
}

bool StateComm::recv_msgs()
{
  if (world_rank != 0)
  {
    return false;
  }
  MPI_Status status;
  int flag;
  bool recv_something = false;

  do
  {
    // Probe
    MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &status);

    if (flag)
    {
      // Recv
      MPI_Recv(&(vec_msgs[status.MPI_SOURCE]), 1, mpi_t, status.MPI_SOURCE, tag,
               MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      stats.increment(Stats::MPI::Recv);
      recv_something = true;
    }
  } while (flag);

  return recv_something;
}

Stats::State StateComm::reset_stats() { return AsyncComm<int>::reset_stats(); }
