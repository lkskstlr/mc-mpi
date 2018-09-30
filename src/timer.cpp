#include "timer.hpp"
#include <mpi.h>

Timer::Timestamp Timer::start(Tag tag) { return {tag, MPI_Wtime()}; }

void Timer::change(Timestamp &timestamp, Tag tag) {
  double time = MPI_Wtime();
  cumm_times[timestamp.tag] += (time - timestamp.time);

  // update timestamp
  timestamp.tag = tag;
  timestamp.time = time;
}

void Timer::stop(Timer::Timestamp timestamp) {
  double endtime = MPI_Wtime();
  cumm_times[timestamp.tag] += (endtime - timestamp.time);
}

double Timer::tick() { return MPI_Wtick(); }

std::ostream &operator<<(std::ostream &os, const Timer &timer) {
  using std::endl;
  const std::string timer_types[] = {"Computation", "Send", "Receive", "Idle"};

  os << "Timer: (";
  double sum = 0.0;
  for (Timer::id_t i = 0; i < Timer::Tag::STATE_COUNT; ++i) {
    sum += timer.cumm_times[i];
    os << timer_types[i] << "=" << timer.cumm_times[i] * 1000.0 << " ms, ";
  }
  os << "Total=" << sum * 1000.0 << " ms) ";
  return os;
}