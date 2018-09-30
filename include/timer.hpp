#ifndef TIMER_HPP
#define TIMER_HPP

#include <iostream>
#include <string>

class Timer {
public:
  typedef int id_t;
  enum Tag : id_t {
    Computation = 0,
    Send,
    Recv,
    Idle,
    STATE_COUNT,
    NO_STATE = -1
  };
  typedef struct timestamp_tag {
    Tag tag;
    double time;
  } Timestamp;

  Timestamp start(Tag tag);
  void change(Timestamp &timestamp, Tag tag);
  void stop(Timestamp timestamp);
  double tick();

  friend std::ostream &operator<<(std::ostream &stream, const Timer &timer);

private:
  double cumm_times[Tag::STATE_COUNT] = {0.0};
};
#endif