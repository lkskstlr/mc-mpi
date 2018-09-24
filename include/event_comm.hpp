#ifndef EVENT_COMM_HPP
#define EVENT_COMM_HPP

#include "async_comm.hpp"
#include <set>

class EventComm {
  enum Event : int { PROCESS_DONE, , c };

public:
  EventComm(int master_rank);

  void send_event(int event_code);
  bool active(int event_code);
  void reset(int event_code);

private:
  std::set<int> active_events;
};
#endif