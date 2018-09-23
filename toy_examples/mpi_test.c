#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char const *argv[]) {
  MPI_Init(NULL, NULL);
  int world_size, world_rank;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int number_amount;
  if (world_rank == 0) {
    const int MAX_NUMBERS = 100;
    int numbers[MAX_NUMBERS];
    srand(time(NULL));
    number_amount = (rand() / (float)RAND_MAX) * MAX_NUMBERS;

    MPI_Request request;
    MPI_Status status;

    MPI_Issend(numbers, number_amount, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    printf("0 sent %d numbers to 1\n", number_amount);
    int flag = 0;
    long int counter = 0;
    int number_amount_dyn = -1;

    do {
      MPI_Test(&request, &flag, &status);
      counter++;
    } while (!flag);

    printf("0 did %ld tests\n", counter);
    if (request == MPI_REQUEST_NULL) {
      printf("0 says: request == MPI_REQUEST_NULL\n");
    }

  } else if (world_rank == 1) {
    sleep(1);

    MPI_Status status;
    // Probe for an incoming message from process zero
    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    // When probe returns, the status object has the size and other
    // attributes of the incoming message. Get the size of the message.
    MPI_Get_count(&status, MPI_INT, &number_amount);
    // Allocate a buffer just big enough to hold the incoming numbers
    int *number_buf = (int *)malloc(sizeof(int) * number_amount);
    // Now receive the message with the allocated buffer
    MPI_Recv(number_buf, number_amount, MPI_INT, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    printf("1 dynamically received %d numbers from 0.\n", number_amount);
    free(number_buf);
  }

  MPI_Finalize();
  return 0;
}