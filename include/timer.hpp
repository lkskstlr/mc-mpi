#ifndef TIMER_HPP
#define TIMER_HPP

#include <iostream>

class Timer {
public:
  typedef int id_t;
  enum Tag : id_t { Computation = 0, Send, Recv, Idle, STATE_COUNT };
  typedef struct timestamp_tag {
    Tag tag;
    double time;
  } Timestamp;
  typedef struct state_tag {
    double cumm_times[Tag::STATE_COUNT] = {0.0};
    double starttime = 0.0;
    double endtime = 0.0;
  } State;

  Timestamp start(Tag tag);
  State restart(Timestamp &timestamp, Tag tag);
  void change(Timestamp &timestamp, Tag tag);
  void stop(Timestamp timestamp);
  void reset();
  double tick();

  friend std::ostream &operator<<(std::ostream &stream, const Timer &timer);

private:
  State state;
};
#endif