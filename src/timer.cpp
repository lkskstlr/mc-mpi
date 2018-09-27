#include "timer.hpp"
#include <mpi.h>

Timer::id_t Timer::start(Tag tag) {
  start_times[tag] = MPI_Wtime();
  return static_cast<id_t>(tag);
}

void Timer::stop(id_t id) {
  double endtime = MPI_Wtime();
  cumm_times[id] += (endtime - start_times[id]);
}

std::ostream &operator<<(std::ostream &os, const Timer &timer) {
  using std::endl;
  const std::string timer_types[] = {"Computation", "Send", "Receive", "Idle"};

  os << "Timer: (";
  id_t i = 0;
  for (; i < Timer::Tag::STATE_COUNT - 1; ++i) {
    os << timer_types[i] << "=" << timer.cumm_times[i] * 1000.0 << " ms, ";
  }
  os << timer_types[i] << "=" << timer.cumm_times[i] * 1000.0 << " ms) ";
  return os;
}