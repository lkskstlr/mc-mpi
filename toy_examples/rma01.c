#define _BSD_SOURCE

#include <stdio.h>
#include <mpi.h>
#include <stdint.h>
#include <unistd.h>
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t state_t;
typedef int data_t;
#define MCMPI_STATE_T MPI_UINT64_T
#define NBUFFER 4
#define NSHIFT 16 // sizeof(state_t) / NBUFFER

#define SET_MASK(i, n) (((state_t)n) << (16 * i))
#define GET_SIZE(p_state, i) ((int)(((*(p_state) >> (16 * i)) & 0xffff)))
#define SET_SIZE(p_state, i, n) *p_state |= SET_MASK(i, n);

typedef struct mpi_buffer_struct
{
    MPI_Win win_state;
    state_t *p_state;

    MPI_Win *wins;
    data_t **p_buffers;
    int size;
} mpi_buffer;

mpi_buffer make_buffer_send(int size)
{
    mpi_buffer buffer;
    buffer.size = size;

    // Malloc arrays
    buffer.wins = (MPI_Win *)malloc(sizeof(MPI_Win) * NBUFFER);
    buffer.p_buffers = (data_t **)malloc(sizeof(data_t *) * NBUFFER);

    // State
    MPI_Win_allocate(0, sizeof(state_t), MPI_INFO_NULL, MPI_COMM_WORLD, &buffer.p_state, &buffer.win_state);
    assert(buffer.p_state == NULL);

    // Buffers
    for (int i = 0; i < NBUFFER; i++)
    {
        MPI_Win_allocate(0, sizeof(data_t),
                         MPI_INFO_NULL, MPI_COMM_WORLD,
                         &buffer.p_buffers[i], &buffer.wins[i]);
        assert(buffer.p_buffers[i] == NULL);
    }

    return buffer;
}

mpi_buffer make_buffer_recv(int size)
{
    mpi_buffer buffer;
    buffer.size = size;

    // Malloc arrays
    buffer.wins = (MPI_Win *)malloc(sizeof(MPI_Win) * NBUFFER);
    buffer.p_buffers = (data_t **)malloc(sizeof(data_t *) * NBUFFER);

    // State
    MPI_Win_allocate(sizeof(state_t), sizeof(state_t), MPI_INFO_NULL, MPI_COMM_WORLD, &buffer.p_state, &buffer.win_state);
    assert(buffer.p_state != NULL);

    // Buffers
    for (int i = 0; i < NBUFFER; i++)
    {
        MPI_Win_allocate(sizeof(data_t) * buffer.size, sizeof(data_t),
                         MPI_INFO_NULL, MPI_COMM_WORLD,
                         &buffer.p_buffers[i], &buffer.wins[i]);
        assert(buffer.p_buffers[i] != NULL);
    }

    return buffer;
}

int get_free_buffer(int target_rank, mpi_buffer buffer)
{
    state_t state;
    MPI_Fetch_and_op(NULL, &state, MPI_UINT64_T, target_rank, 0, MPI_NO_OP, buffer.win_state);

    for (int i = 0; i < NBUFFER; i++)
    {
        if (GET_SIZE(&state, i) == 0)
            return i;
    }
    return -1;
}

void set_buffer(int target_rank, mpi_buffer buffer, int i, int n)
{

    assert((i >= 0) && (i < NBUFFER) && "Only NBUFFER lines");
    state_t state;
    state_t mask = SET_MASK(i, n);
    MPI_Fetch_and_op(&mask, &state, MPI_UINT64_T, target_rank, 0, MPI_BOR, buffer.win_state);

    assert(GET_SIZE(&state, i) == 0 && "This method of setting the size only works if it was 0 before. Now the state is corrupted.");
}

void send_data(int target_rank, mpi_buffer buffer, int i, int n, data_t const *p_data)
{

    assert((i >= 0) && (i < NBUFFER) && "Only NBUFFER lines");

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, MPI_MODE_NOCHECK, buffer.wins[i]);
    MPI_Put(p_data, n, MPI_INT, target_rank, 0, n, MPI_INT, buffer.wins[i]);
    MPI_Win_unlock(target_rank, buffer.wins[i]);
}

void buffer_print(int world_size, int world_rank, mpi_buffer buffer)
{
    if (buffer.p_state == NULL)
        printf("%d/%d, Send Buffer: size = %d\n", world_rank, world_size, buffer.size);
    else
    {
        char size_str[100] = {0};
        char tmp_str[20] = {0};
        for (int i = NBUFFER - 1; i >= 0; i--)
        {
            sprintf(tmp_str, "%d ", GET_SIZE(buffer.p_state, i));
            strcat(size_str, tmp_str);
        }
        printf("%d/%d, Recv Buffer: size = %d, sizes = [%s]\n",
               world_rank, world_size, buffer.size, size_str);
    }
}

void buffer_print_lines(mpi_buffer buffer)
{
    if (buffer.p_buffers == NULL)
        return;

    char *str = (char *)malloc(sizeof(char) * NBUFFER * (buffer.size + 1) * 20);
    char tmp_str[20] = {0};
    for (int i = 0; i < NBUFFER; i++)
    {
        sprintf(tmp_str, "[%d]: ", i);
        strcat(str, tmp_str);

        for (int j = 0; j < GET_SIZE(buffer.p_state, i); j++)
        {
            sprintf(tmp_str, "%d ", buffer.p_buffers[i][j]);
            strcat(str, tmp_str);
        }
        strcat(str, "\n");
    }

    printf(str);
    return;
}

int main(int argc, char **argv)
{
    int world_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int buffer_size = 32;
    mpi_buffer buffer;

    if (world_rank == 0)
    {
        buffer = make_buffer_recv(buffer_size);
        // SET_SIZE(buffer.p_state, 0, 0);
        // SET_SIZE(buffer.p_state, 1, 889);
        // SET_SIZE(buffer.p_state, 2, 17);
        // SET_SIZE(buffer.p_state, 3, 221);
    }
    else
        buffer = make_buffer_send(buffer_size);

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(10000 * world_rank);
    buffer_print(world_size, world_rank, buffer);
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 1)
    {
        int free_buffer = get_free_buffer(0, buffer);
        printf("\n%d/%d free_buffer = %d\n", world_rank, world_size, free_buffer);

        if (free_buffer >= 0)
        {
            int n = 3;
            data_t p_data[3] = {11, 13, 17};
            send_data(0, buffer, free_buffer, n, p_data);
            set_buffer(0, buffer, free_buffer, n);
            printf("%d/%d set buffer size to %d\n", world_rank, world_size, n);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(10000 * world_rank);
    buffer_print(world_size, world_rank, buffer);
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0)
        buffer_print_lines(buffer);

    MPI_Finalize();
    return 0;
}