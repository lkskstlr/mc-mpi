#ifndef LATCHED_COMM
#define LATCHED_COMM

#include "async_comm.hpp"
#include <vector>

class LatchedComm : private AsyncComm<int> {
public:
  LatchedComm(int world_size, int world_rank, int tag, int init_msg = 0);
  bool send(int msg);
  bool bcast(int msg);
  std::vector<int> const &msgs();
  int msg();

private:
  int world_size;
  int my_msg;
  int last_msg;
  int tag;
  std::vector<int> vec_msgs;
};
#endif