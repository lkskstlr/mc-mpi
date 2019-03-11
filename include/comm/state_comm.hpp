#ifndef STATE_COMM
#define STATE_COMM

#include "async_comm.hpp"
#include <functional>
#include <vector>

class StateComm : public AsyncComm<int>
{
public:
  enum State : int
  {
    Running = 0,
    Finished,
    STATE_COUNT
  };

  StateComm(int world_size, int world_rank, int tag,
            std::function<State(std::vector<int>)> state_lambda,
            State init_state = State::Running, int init_msg = 0);
  bool send_msg(int msg);
  bool send_state();
  State recv_state();
  Stats::State reset_stats();

private:
  bool recv_msgs();

  const int world_size;
  const int tag;
  State state;
  int last_msg;
  std::vector<int> vec_msgs;
  std::function<State(std::vector<int>)>
      state_lambda; /** is called to decide state */
};
#endif