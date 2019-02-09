#include "rma_comm.hpp"
#include <assert.h>
#include <iostream>
#include <inttypes.h>

#define NBUFFER 4
#define NSHIFT 16 // sizeof(state_t) / NBUFFER

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define SET_MASK(i, n) (((state_t)n) << (16 * i))
#define GET_SIZE(p_state, i) ((int)(((*(p_state) >> (16 * i)) & 0xffff)))
#define SET_SIZE(p_state, i, n) *p_state |= SET_MASK(i, n);

template <typename T>
void RmaComm<T>::init(int buffer_size)
{
    this->buffer_size = buffer_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}

template <typename T>
void RmaComm<T>::advertise(int target_rank)
{
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
    BufferInfo buffer;
    buffer.size = buffer_size;

    // Malloc arrays
    buffer.wins = (MPI_Win *)malloc(sizeof(MPI_Win) * NBUFFER);
    buffer.lines = (T **)malloc(sizeof(T *) * NBUFFER);

    // State
    MPI_Win_allocate(0, sizeof(state_t), MPI_INFO_NULL, new_comm, &buffer.p_state, &buffer.win_state);
    assert(buffer.p_state == NULL);

    // Buffers
    for (int i = 0; i < NBUFFER; i++)
    {
        MPI_Win_allocate(0, sizeof(T),
                         MPI_INFO_NULL, new_comm,
                         &buffer.lines[i], &buffer.wins[i]);
        assert(buffer.lines[i] == NULL);
    }

    send_buffer_infos[target_rank] = buffer;
}

template <typename T>
void RmaComm<T>::subscribe(int source_rank)
{
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // 0 = recv
    int ranks[2] = {world_rank, source_rank};

    MPI_Group new_group;
    MPI_Group_incl(world_group, 2, ranks, &new_group);

    MPI_Comm new_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &new_comm);

    /* Creating RMA buffer */
    BufferInfo buffer;
    buffer.size = buffer_size;

    // Malloc arrays
    buffer.wins = (MPI_Win *)malloc(sizeof(MPI_Win) * NBUFFER);
    buffer.lines = (T **)malloc(sizeof(T *) * NBUFFER);

    // State
    MPI_Win_allocate(sizeof(state_t), sizeof(state_t), MPI_INFO_NULL, new_comm, &buffer.p_state, &buffer.win_state);
    assert(buffer.p_state != NULL);

    // Buffers
    for (int i = 0; i < NBUFFER; i++)
    {
        MPI_Win_allocate(sizeof(T) * buffer.size, sizeof(T),
                         MPI_INFO_NULL, new_comm,
                         &buffer.lines[i], &buffer.wins[i]);
        assert(buffer.lines[i] != NULL);
    }

    recv_buffer_infos[source_rank] = buffer;
}

template <typename T>
void RmaComm<T>::send(std::vector<T> &data, int dest)
{
    /* Find Buffer Info */
    auto iter = send_buffer_infos.find(dest);
    if (iter == send_buffer_infos.end())
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    BufferInfo buffer = iter->second;

    /* Find free state */
    state_t state;
    MPI_Fetch_and_op(NULL, &state, mpi_state_t, 0, 0, MPI_NO_OP, buffer.win_state);

    int free_buffer = -1;
    for (int i = 0; i < NBUFFER; i++)
    {
        if (GET_SIZE(&state, i) == 0)
        {
            free_buffer = i;
            break;
        }
    }
    if (free_buffer == -1)
        return;

    /* Amount of data to send and pointer */
    int n_send = static_cast<int>(MIN(data.size(), buffer.size));
    size_t n_after = data.size() - n_send;
    T const *p_data = data.data() + n_after;

    /* Put data */
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, buffer.wins[free_buffer]);
    MPI_Put(p_data, n_send, mpi_data_t, 0, 0, n_send, mpi_data_t, buffer.wins[free_buffer]);
    MPI_Win_unlock(0, buffer.wins[free_buffer]);

    data.resize(n_after);

    /* Set state */
    state_t mask = SET_MASK(free_buffer, n_send);
    MPI_Fetch_and_op(&mask, &state, mpi_state_t, 0, 0, MPI_BOR, buffer.win_state);

    assert(GET_SIZE(&state, free_buffer) == 0 && "This method of setting the size only works if it was 0 before. Now the state is corrupted.");
}

template <typename T>
bool RmaComm<T>::recv(std::vector<T> &data, int source)
{
    /* Find Buffer Info */
    auto iter = recv_buffer_infos.find(source);
    if (iter == recv_buffer_infos.end())
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    BufferInfo buffer = iter->second;

    /* Get state */
    state_t state;
    state_t mask = 0;
    MPI_Fetch_and_op(NULL, &state, mpi_state_t, 0, 0, MPI_NO_OP, buffer.win_state);
    bool result = (state != 0);

    /* Collect data */
    for (int i = 0; i < NBUFFER; i++)
    {
        if (GET_SIZE(&state, i) > 0)
        {
            printf("There is data in line %d\n", i);
            int add_size = GET_SIZE(&state, i);
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, buffer.wins[i]);
            data.insert(data.end(), buffer.lines[i], buffer.lines[i] + add_size);
            MPI_Win_unlock(0, buffer.wins[i]);

            printf("MASKi is %" PRIx64 "\n", ((state_t)0xffff) << (16 * i));
            mask |= ((state_t)0xffff) << (16 * i);
        }
    }
    mask = ~mask;

    printf("STATE is %" PRIx64 "\n", state);
    printf("MASK  is %" PRIx64 "\n", mask);

    /* Set state */
    MPI_Fetch_and_op(&mask, &state, mpi_state_t, 0, 0, MPI_BAND, buffer.win_state);

    if ((mask & state) > 0)
    {
        printf("Got intermediat data\n");
    }

    return result;
}

template <typename T>
void RmaComm<T>::print()
{
    using std::cout;
    using std::endl;

    for (auto const &buffer_info : recv_buffer_infos)
    {
        cout << "--- Subscribed to World Rank " << buffer_info.first << " ---\n";

        state_t state;
        state = *buffer_info.second.p_state;
        cout << "Sizes = ";
        for (int i = 0; i < NBUFFER; i++)
            cout << GET_SIZE(&state, i) << " ";
        cout << "\n";

        for (int i = 0; i < NBUFFER; i++)
        {
            cout << "\t[" << i << "]: ";
            for (int j = 0; j < GET_SIZE(&state, i); j++)
            {
                cout << buffer_info.second.lines[i][j] << " ";
            }
            cout << "\n";
        }
        cout << endl;
    }
}