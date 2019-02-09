#ifndef RMA_COMM
#define RMA_COMM

// #include "async_comm.hpp"
#include <mpi.h>
#include <stdint.h>
#include <map>
#include <vector>

template <typename T>
class RmaComm
// class RmaComm : public AsyncComm<T>
{
  protected:
    typedef uint64_t state_t; // Has to match mpi_state_t
    typedef struct buffer_info_tag
    {
        MPI_Win win_state;
        state_t *p_state;

        MPI_Win *wins;
        T **lines;
        int size;
    } BufferInfo;

  public:
    void init(int buffer_size);
    void advertise(int target_rank);
    void subscribe(int source_rank);

    /*!
   * \function send
   *
   * \brief Stores data in buffer and sends it when possible
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
   * \brief Will receive any data that is waiting to be sent.
   *
   * \param[in] data Reference of vector at wich the received data will be
   * appended at the end
   * \param[in] source source of the data, can be MPI_ANY_SOURCE
   *
   * \return bool true if data was received, false otherwise
   */
    bool recv(std::vector<T> &data, int source);

    void print();

  private:
    int world_rank;
    int buffer_size;
    std::map<int, MPI_Comm> comms;
    std::map<int, BufferInfo> recv_buffer_infos;
    std::map<int, BufferInfo> send_buffer_infos;

  public:
    MPI_Datatype mpi_data_t;
    const MPI_Datatype mpi_state_t = MPI_UINT64_T; // Has to match state_t
};
#endif