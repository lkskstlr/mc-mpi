#ifndef ASYNC_COMM_HPP
#define ASYNC_COMM_HPP

#include <mpi.h>
#include <stdio.h>
#include <vector>

#define SEND_INFO_INITIAL_RESERVE 256
#define NDEBUG

template <typename T> class AsyncComm {

  /* Only needed inside class */
  typedef struct send_info_tag {
    void *buf;
    std::size_t bytes;
    MPI_Request request;
  } SendInfo;

public:
  /*!
   * \function init
   *
   * \brief Has to be called before usage! Initializes the asyncchronous
   * communicator. mpi_t has to be the mpi datatype for T, the template
   * parameter of the class
   *
   * \param[in] world_rank mpi world rank
   * \param[in] mpi_t the mpi datatype for T, the template parameter of the
   * class
   * \param[in] max_buffer_size maximum size of internal buffer in bytes
   */
  void init(int world_rank, MPI_Datatype const mpi_t,
            std::size_t max_buffer_size) {
    this->world_rank = world_rank;
    this->mpi_t = mpi_t;
    this->max_buffer_size = max_buffer_size;
    send_infos.reserve(SEND_INFO_INITIAL_RESERVE);
  }

  /*!
   * \function send
   *
   * \brief Stores data in buffer and sends it when possible
   *
   * \param[in] data Reference of vector with data to be sent
   * \param[in] dest Destination
   * \param[in] tag tag, can be MPI_ANY_TAG
   *
   * \return void
   */
  void send(std::vector<T> const &data, int dest, int tag) {
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

    // potentially (at least once) free memory. This will lock if the buffer is
    // small and some sends have not completed.
    do {
      this->free();
    } while (curr_buffer_size + send_info.bytes > max_buffer_size);

    send_info.buf = (T *)malloc(send_info.bytes);
    memcpy(send_info.buf, data.data(), send_info.bytes);

    MPI_Issend(send_info.buf, data.size(), mpi_t, dest, tag, MPI_COMM_WORLD,
               &send_info.request);

#ifndef NDEBUG
    printf("AsynCommSend: %4d -> %4d (%5d) : %10zu bytes, buff = (%8zu/%8zu)\n",
           world_rank, dest, tag, send_info.bytes, curr_buffer_size,
           max_buffer_size);
#endif
    curr_buffer_size += send_info.bytes;
    send_infos.push_back(send_info);
  };

  /*!
   * \function send
   *
   * \brief Stores data in buffer and sends it when possible
   *
   * \param[in] instance One instance of T to be sent
   * \param[in] dest Destination
   * \param[in] tag tag, can be MPI_ANY_TAG
   *
   * \return void
   */
  void send(T const &instance, int dest, int tag) {

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
    do {
      this->free();
    } while (curr_buffer_size + send_info.bytes > max_buffer_size);

    send_info.buf = (T *)malloc(send_info.bytes);
    memcpy(send_info.buf, &instance, send_info.bytes);

    MPI_Issend(send_info.buf, 1, mpi_t, dest, tag, MPI_COMM_WORLD,
               &send_info.request);

#ifndef NDEBUG
    printf("AsynCommSend: %4d -> %4d (%5d) : %10zu bytes, buff = (%8zu/%8zu)\n",
           world_rank, dest, tag, send_info.bytes, curr_buffer_size,
           max_buffer_size);
#endif
    curr_buffer_size += send_info.bytes;
    send_infos.push_back(send_info);
  };

  /*!
   * \function recv
   *
   * \brief Will receive any data that is waiting to be sent.
   *
   * \param[in] data Reference of vector at wich the received data will be
   * appended at the end
   * \param[in] source source of the data, can be MPI_ANY_SOURCE
   * \param[in] tag tag, can be MPI_ANY_TAG
   *
   * \return bool true if data was received, false otherwise
   */
  bool recv(std::vector<T> &data, int source, int tag) {
    MPI_Status status;
    int flag;
    int nb_data = -1;
    do {
      MPI_Iprobe(source, tag, MPI_COMM_WORLD, &flag, &status);
      if (flag) {
        MPI_Get_count(&status, mpi_t, &nb_data);
        T *buf = (T *)malloc(sizeof(T) * nb_data);
        MPI_Recv(buf, nb_data, mpi_t, status.MPI_SOURCE, tag, MPI_COMM_WORLD,
                 MPI_STATUSES_IGNORE);
#ifndef NDEBUG
        printf("AsynCommRecv: %4d -> %4d (%5d) : %10d bytes\n",
               status.MPI_SOURCE, world_rank, tag, nb_data);
#endif
        data.reserve(data.size() + nb_data);
        for (std::size_t i = 0; i < nb_data; ++i) {
          data.push_back(buf[i]);
        }

        ::free(buf);
      }
    } while (flag);

    return (nb_data != -1);
  };

  // frees up memory in buffer
  void free() {
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
  };

private:
  int world_rank;
  std::size_t curr_buffer_size = 0;
  std::size_t max_buffer_size;

  MPI_Datatype mpi_t;
  std::vector<SendInfo> send_infos;
};
#endif