#ifndef TIMER_HPP
#define TIMER_HPP

// I use the fact that structs with methods are still POD in memory which is
// guranteed by the standard. See:
// https://stackoverflow.com/questions/422830/structure-of-a-c-object-in-memory-vs-a-struct

#include <mpi.h>
#include <stdio.h>
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

    int sprintf(char *str);
    static int sprintf_header(char *str);
    static int sprintf_max_len();
    static MPI_Datatype mpi_t();
  } State;

  Timer();
  Timestamp start(Tag tag);
  State restart(Timestamp &timestamp, Tag tag);
  void change(Timestamp &timestamp, Tag tag);
  State stop(Timestamp timestamp);
  void reset();
  double tick() const;
  double time() const;

  double starttime() const;

  friend std::ostream &operator<<(std::ostream &stream, const Timer &timer);

 private:
  State state;
  const double offset;
};

#endif