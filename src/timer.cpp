#include "timer.hpp"
#include <mpi.h>
#include <string>

Timer::Timestamp Timer::start(Tag tag) {
  state.starttime = MPI_Wtime();
  return {tag, state.starttime};
}

void Timer::change(Timestamp &timestamp, Tag tag) {
  double time = MPI_Wtime();
  state.cumm_times[timestamp.tag] += (time - timestamp.time);

  // update timestamp
  timestamp.tag = tag;
  timestamp.time = time;
}

void Timer::stop(Timer::Timestamp timestamp) {
  state.endtime = MPI_Wtime();
  state.cumm_times[timestamp.tag] += (state.endtime - timestamp.time);
}

void Timer::reset() {
  state.starttime = 0.0;
  state.endtime = 0.0;
  for (int i = 0; i < Tag::STATE_COUNT; ++i) {
    state.cumm_times[i] = 0.0;
  }
}

Timer::State Timer::restart(Timestamp &timestamp, Tag tag) {
  // change context
  change(timestamp, tag);

  // populate result
  State res = state;
  res.endtime = timestamp.time;

  // change own state
  state.starttime = timestamp.time;

  // return
  return res;
}

double Timer::tick() { return MPI_Wtick(); }

std::ostream &operator<<(std::ostream &os, const Timer &timer) {
  using std::endl;
  const std::string timer_types[] = {"Computation", "Send", "Receive", "Idle"};

  os << "Timer: (";
  double sum = 0.0;
  for (Timer::id_t i = 0; i < Timer::Tag::STATE_COUNT; ++i) {
    sum += timer.state.cumm_times[i];
    os << timer_types[i] << "=" << timer.state.cumm_times[i] * 1000.0
       << " ms, ";
  }
  os << "Total=" << sum * 1000.0 << " ms) ";
  return os;
}