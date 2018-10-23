#ifndef ASYNC_COMM_HPP
#define ASYNC_COMM_HPP

#include "stats.hpp"
#include <mpi.h>
#include <stdio.h>
#include <vector>

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
            std::size_t max_buffer_size);

  /*!
   * \function send
   *
   * \brief Stores data in buffer and sends it when possible
   *
   * \param[in] data Reference of vector with data to be sent. WILL BE CLEARED!
   * \param[in] dest Destination
   * \param[in] tag tag, can be MPI_ANY_TAG
   *
   * \return void
   */
  void send(std::vector<T> &data, int dest, int tag);

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
  void send(T const &instance, int dest, int tag);

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
  bool recv(std::vector<T> &data, int source, int tag);

  /*!
   * \function reset_stats
   *
   * \brief Will reset the internal statistics and return the old ones
   *
   * \return Stats::State returns the old Stats::State
   */
  Stats::State reset_stats();

  // frees up memory in buffer
  void free();

protected:
  int world_rank;
  std::size_t curr_buffer_size = 0;
  std::size_t max_buffer_size;

  MPI_Datatype mpi_t;
  std::vector<SendInfo> send_infos;

  Stats stats;
};
#endif