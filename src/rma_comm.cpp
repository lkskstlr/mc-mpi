#include "rma_comm.hpp"
#include <assert.h>
#include <inttypes.h>
#include <iostream>

#define NBUFFER 4
#define NSHIFT 16 // sizeof(state_t) / NBUFFER

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define SET_MASK(i, n) (((state_t)n) << (16 * i))
#define GET_SIZE(p_state, i) ((int)(((*(p_state) >> (16 * i)) & 0xffff)))
#define SET_SIZE(p_state, i, n) *p_state |= SET_MASK(i, n);

#define EVEN(n) ((n % 2) == 0)
#define ODD(n) ((n % 2) == 1)

template <typename T> void RmaComm<T>::init(int buffer_size) {
  if (buffer_size <= 0) {
    // use max buffer size
    buffer_size = (1 << 16) - 1;
  }
  if ((buffer_size >> 16) != 0) {
    fprintf(stderr, "Buffersize too larger. Aborting.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  this->buffer_size = buffer_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}

template <typename T> void RmaComm<T>::init_1d(int buffer_size) {
  this->init(buffer_size);

  /* Even <-> Odd */
  if (EVEN(world_rank) && (world_rank + 1 < world_size))
    connect(world_rank + 1);
  if (ODD(world_rank))
    connect(world_rank - 1);

  /* Odd <-> Even*/
  if (ODD(world_rank) && (world_rank + 1 < world_size))
    connect(world_rank + 1);
  if (EVEN(world_rank) && (world_rank > 0))
    connect(world_rank - 1);
}

template <typename T> void RmaComm<T>::advertise(int target_rank) {
  /* Creating Communicator */
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  // 0 = recv
  int ranks[2] = {target_rank, world_rank};

  MPI_Group new_group;
  MPI_Group_incl(world_group, 2, ranks, &new_group);

  MPI_Comm new_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &new_comm);

  comms[target_rank] = new_comm;

  /* Creating RMA buffer */
  BufferInfo buffer = {0};
  buffer.size = buffer_size;

  // Malloc arrays
  buffer.wins = (MPI_Win *)calloc(NBUFFER, sizeof(MPI_Win));
  buffer.lines = (T **)calloc(NBUFFER, sizeof(T *));

  // State
  MPI_Win_allocate(0, sizeof(state_t), MPI_INFO_NULL, new_comm, &buffer.p_state,
                   &buffer.win_state);

  // Buffers
  for (int i = 0; i < NBUFFER; i++) {
    MPI_Win_allocate(0, sizeof(T), MPI_INFO_NULL, new_comm, &buffer.lines[i],
                     &buffer.wins[i]);
  }

  send_buffer_infos[target_rank] = buffer;
}

template <typename T> void RmaComm<T>::subscribe(int source_rank) {
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  // 0 = recv
  int ranks[2] = {world_rank, source_rank};

  MPI_Group new_group;
  MPI_Group_incl(world_group, 2, ranks, &new_group);

  MPI_Comm new_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &new_comm);

  /* Creating RMA buffer */
  BufferInfo buffer = {0};
  buffer.size = buffer_size;

  // Malloc arrays
  buffer.wins = (MPI_Win *)calloc(NBUFFER, sizeof(MPI_Win));
  buffer.lines = (T **)calloc(NBUFFER, sizeof(T *));

  // State
  MPI_Win_allocate(sizeof(state_t), sizeof(state_t), MPI_INFO_NULL, new_comm,
                   &buffer.p_state, &buffer.win_state);
  assert(buffer.p_state != NULL);
  *buffer.p_state = 0;

  // Buffers
  for (int i = 0; i < NBUFFER; i++) {
    MPI_Win_allocate(sizeof(T) * buffer.size, sizeof(T), MPI_INFO_NULL,
                     new_comm, &buffer.lines[i], &buffer.wins[i]);
    assert(buffer.lines[i] != NULL);
  }

  recv_buffer_infos[source_rank] = buffer;
}

template <typename T> void RmaComm<T>::connect(int target_rank) {
  if (world_rank < target_rank) {
    subscribe(target_rank);
    advertise(target_rank);
  } else {
    advertise(target_rank);
    subscribe(target_rank);
  }
}

template <typename T> void RmaComm<T>::send(std::vector<T> &data, int dest) {
  if (data.size() == 0)
    return;

  /* Find Buffer Info */
  auto iter = send_buffer_infos.find(dest);
  if (iter == send_buffer_infos.end()) {
    fprintf(stderr, "dest (%d) not found on %d, size = %zu\n", dest, world_rank,
            data.size());
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  BufferInfo buffer = iter->second;

  /* Find free state */
  state_t state;
  MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, buffer.win_state);
  MPI_Fetch_and_op(NULL, &state, mpi_state_t, 0, 0, MPI_NO_OP,
                   buffer.win_state);
  MPI_Win_unlock(0, buffer.win_state);

  int free_buffer = -1;
  for (int i = 0; i < NBUFFER; i++) {
    if (GET_SIZE(&state, i) == 0) {
      free_buffer = i;
      break;
    }
  }
  if (free_buffer == -1)
    return;

  /* Amount of data to send and pointer */
  int n_send = static_cast<int>(MIN((int)data.size(), buffer.size));
  size_t n_after = data.size() - n_send;
  T const *p_data = data.data() + n_after;

  /* Put data */
  MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, // MPI_MODE_NO_CHECK is the problem
               buffer.wins[free_buffer]);
  MPI_Put(p_data, n_send, mpi_data_t, 0, 0, n_send, mpi_data_t,
          buffer.wins[free_buffer]);
  MPI_Win_unlock(0, buffer.wins[free_buffer]);

  data.resize(n_after);

  /* Set state */
  state_t mask = SET_MASK(free_buffer, n_send);
  MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, buffer.win_state);
  MPI_Fetch_and_op(&mask, &state, mpi_state_t, 0, 0, MPI_BOR, buffer.win_state);
  MPI_Win_unlock(0, buffer.win_state);

  assert(GET_SIZE(&state, free_buffer) == 0 &&
         "This method of setting the size only works if it was 0 before. Now "
         "the state is corrupted.");
}

template <typename T> bool RmaComm<T>::recv(std::vector<T> &data, int source) {
  /* Find Buffer Info */
  auto iter = recv_buffer_infos.find(source);
  if (iter == recv_buffer_infos.end()) {
    return false;
  }
  BufferInfo buffer = iter->second;

  /* Get state */
  state_t state;
  state_t mask = 0;
  MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, buffer.win_state);
  MPI_Fetch_and_op(NULL, &state, mpi_state_t, 0, 0, MPI_NO_OP,
                   buffer.win_state);
  MPI_Win_unlock(0, buffer.win_state);
  bool result = (state != 0);

  /* Collect data */
  for (int i = 0; i < NBUFFER; i++) {
    if (GET_SIZE(&state, i) > 0) {
      int add_size = GET_SIZE(&state, i);
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, buffer.wins[i]);
      data.insert(data.end(), buffer.lines[i], buffer.lines[i] + add_size);
      MPI_Win_unlock(0, buffer.wins[i]);

      mask |= ((state_t)0xffff) << (16 * i);
    }
  }
  mask = ~mask;

  /* Set state */
  MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, buffer.win_state);
  MPI_Fetch_and_op(&mask, &state, mpi_state_t, 0, 0, MPI_BAND,
                   buffer.win_state);
  MPI_Win_unlock(0, buffer.win_state);

  return result;
}
