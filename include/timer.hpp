#ifndef TIMER_HPP
#define TIMER_HPP

#include <iostream>
#include <string>

class Timer {
public:
  typedef int id_t;
  enum Tag : int { Computation = 0, Send, Receive, Idle, STATE_COUNT };

  id_t start(Tag tag);
  void stop(id_t id);

  friend std::ostream &operator<<(std::ostream &stream, const Timer &timer);

private:
  double start_times[Tag::STATE_COUNT] = {0.0};
  double cumm_times[Tag::STATE_COUNT] = {0.0};
};
#endif