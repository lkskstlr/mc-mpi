#ifndef RMA_COMM
#define RMA_COMM

// #include "async_comm.hpp"
#include <mpi.h>
#include <stdint.h>
#include <map>
#include <vector>

template <typename T>
class RmaComm {
 public:
  typedef uint64_t state_t;  // Has to match mpi_state_t
  typedef struct buffer_info_tag {
    MPI_Win win_state;
    state_t *p_state;

    MPI_Win *wins;
    T **lines;
    int size;
  } BufferInfo;

 public:
  /*!
   * \function init
   *
   * \brief Initializes the communicator
   *
   * \param[in] buffer_size Size of one buffer line used.
   * Max number of elements that can be send in one message. buffer_size <= 2^16
   * - 1 = 65535
   *
   * \return void
   */
  void init(int buffer_size);

  /*!
   * \function init_1d
   *
   * \brief Initializes the communicator. Also builds all connections for 1d
   * mesh.
   *
   * \param[in] buffer_size Size of one buffer line used.
   * Max number of elements that can be send in one message. buffer_size <= 2^16
   * - 1 = 65535
   *
   * \return void
   */
  void init_1d(int buffer_size);

  /*!
   * \function advertise
   *
   * \brief Advertises (i.e. later send) a connection to target_rank
   * Each advertise call MUST be met by a subscribe call on process target_rank
   * Prefer connect function.
   *
   * \param[in] target_rank Rank of the process (in MPI_COMM_WORLD) to later
   * send data to
   *
   * \return void
   */
  void advertise(int target_rank);

  /*!
   * \function subscribe
   *
   * \brief Subscribe (i.e. later recv) a connection from source_rank
   * Each subscribe call MUST be met by an advertise call on process source_rank
   * Prefer connect function.
   *
   * \param[in] source_rank Rank of the process (in MPI_COMM_WORLD) to later
   * recv data from
   *
   * \return void
   */
  void subscribe(int source_rank);

  /*!
   * \function connect
   *
   * \brief Connect (i.e. later send and recv) with target_rank
   *
   * \param[in] target_rank Rank of the process (in MPI_COMM_WORLD) to later
   * send data to and recv data from
   *
   * \return void
   */
  void connect(int target_rank);

  /*!
   * \function send
   *
   * \brief Put data into remote buffer.
   * Must have used advertise (or connect) before. Otherwise undefined behavior.
   *
   * \param[in] data Reference of vector with data to be sent. WILL BE CLEARED!
   * \param[in] dest Destination
   *
   * \return void
   */
  void send(std::vector<T> &data, int dest);

  /*!
   * \function recv
   *
   * \brief Recv all data from internal buffer
   * Must have used subscribe (or connect) before. Otherwise undefined behavior.
   *
   * \param[in] data Reference of vector at wich the received data will be
   * appended at the end
   * \param[in] source source of the data
   *
   * \return bool true if data was received, false otherwise
   */
  bool recv(std::vector<T> &data, int source);

  /*!
   * \function print
   *
   * \brief Print current state of buffers to std::cout
   *
   * \return void
   */
  void print();

 public:
  int world_size, world_rank;
  int buffer_size;
  std::map<int, MPI_Comm> comms;
  std::map<int, BufferInfo> recv_buffer_infos;
  std::map<int, BufferInfo> send_buffer_infos;

 public:
  MPI_Datatype mpi_data_t;
  const MPI_Datatype mpi_state_t = MPI_UINT64_T;  // Has to match state_t
};
#endif