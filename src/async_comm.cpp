#include "async_comm.hpp"
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <mpi.h>

template <typename T>
void AsyncComm<T>::init(int world_rank, MPI_Datatype const mpi_t,
                        size_t max_buffer_size) {
  constexpr int send_info_initial_reserve = 256;
  this->world_rank = world_rank;
  this->mpi_t = mpi_t;
  this->max_buffer_size = max_buffer_size;
  send_infos.reserve(send_info_initial_reserve);
}

template <typename T>
void AsyncComm<T>::send(std::vector<T> &data, int dest, int tag) {
  if (data.empty()) {
    return;
  }

  SendInfo send_info;
  send_info.bytes = sizeof(T) * data.size();

  // Use only limited space
  if (send_info.bytes > max_buffer_size) {
    fprintf(stderr,
            "Abort in AsyncComm. Trying to send a message > "
            "max_buffer_size. world_rank = %d, message size = %zu "
            "bytes, max_bufer_size = %zu, dest = %d, tag = %d\n",
            world_rank, send_info.bytes, max_buffer_size, dest, tag);
    assert(send_info.bytes <=
           max_buffer_size); // gives nicer error message, but only in DEBUG
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // potentially free memory. This will lock if the buffer is
  // small and some sends have not completed.
  while (curr_buffer_size + send_info.bytes > max_buffer_size) {
    this->free();
  }

  send_info.buf = (T *)malloc(send_info.bytes);
  memcpy(send_info.buf, data.data(), send_info.bytes);

  MPI_Isend(send_info.buf, data.size(), mpi_t, dest, tag, MPI_COMM_WORLD,
            &send_info.request);
  stats.increment(Stats::MPI::Send);

  curr_buffer_size += send_info.bytes;
  stats.buffer_size(curr_buffer_size);
  send_infos.push_back(send_info);

  data.clear();
}

template <typename T>
void AsyncComm<T>::send(T const &instance, int dest, int tag) {

  SendInfo send_info;
  send_info.bytes = sizeof(T);

  // Use only limited space
  if (send_info.bytes > max_buffer_size) {
    fprintf(stderr,
            "Abort in AsyncComm. Trying to send a message > "
            "max_buffer_size. world_rank = %d, message size = %zu "
            "bytes, max_bufer_size = %zu, dest = %d, tag = %d\n",
            world_rank, send_info.bytes, max_buffer_size, dest, tag);
    assert(send_info.bytes <=
           max_buffer_size); // gives nicer error message, but only in DEBUG
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // potentially (at least once) free memory. This will lock if the buffer is
  // small and some sends have not completed.
  while (curr_buffer_size + send_info.bytes > max_buffer_size) {
    this->free();
  }

  send_info.buf = (T *)malloc(send_info.bytes);
  memcpy(send_info.buf, &instance, send_info.bytes);

  MPI_Isend(send_info.buf, 1, mpi_t, dest, tag, MPI_COMM_WORLD,
            &send_info.request);
  stats.increment(Stats::MPI::Send);

  curr_buffer_size += send_info.bytes;
  stats.buffer_size(curr_buffer_size);
  send_infos.push_back(send_info);
}

template <typename T>
bool AsyncComm<T>::recv(std::vector<T> &data, int source, int tag) {
  MPI_Status status;
  int flag;
  int nb_data = -1;
  int factor = 2;
  char *buf = NULL;
  size_t buf_size = 0;
  size_t buf_used = 0;
  size_t new_bytes = 0;

  do {
    // Probe
    MPI_Iprobe(source, tag, MPI_COMM_WORLD, &flag, &status);

    if (flag) {
      // Get Count
      MPI_Get_count(&status, mpi_t, &nb_data);

      // Buffer
      new_bytes = sizeof(T) * nb_data;
      if (!buf) {
        buf = (char *)malloc(new_bytes);
        buf_size = new_bytes;
      } else {
        if (buf_used + new_bytes > buf_size) {
          // enlarge
          size_t new_size =
              std::max(factor * buf_size, buf_size + factor * new_bytes);
          buf = (char *)realloc(buf, new_size);
          if (!buf) {
            fprintf(stderr, "Couldn't allocate!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
          }
          buf_size = new_size;
        }
      }

      // Recv
      MPI_Recv(buf + buf_used, nb_data, mpi_t, status.MPI_SOURCE, tag,
               MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      stats.increment(Stats::MPI::Recv);
      buf_used += new_bytes;
    }
  } while (flag);

  data.reserve(data.size() + buf_used / sizeof(T));
  for (size_t i = 0; i < buf_used / sizeof(T); ++i) {
    data.push_back(((T *)buf)[i]);
  }

  ::free(buf);

  return (buf_used > 0);
}

template <typename T> void AsyncComm<T>::free() {
  MPI_Status status;
  int flag = 0;

  // erase: https://en.cppreference.com/w/cpp/container/vector/erase
  for (auto iter = send_infos.begin(); iter != send_infos.end();) {
    flag = 0;
    MPI_Test(&(iter->request), &flag, &status);

    if (flag) {
      /* the send is through */
      ::free(iter->buf); // from std lib
      curr_buffer_size -= iter->bytes;
      iter = send_infos.erase(iter);
    } else {
      ++iter;
    }
  }
}

template <typename T> Stats::State AsyncComm<T>::reset_stats() {
  return stats.reset();
}

template <typename T> size_t AsyncComm<T>::get_max_used_buffer() const {
  return stats.get_max_buffer_size();
}
