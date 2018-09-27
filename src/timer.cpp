#include "timer.hpp"
#include <mpi.h>

const Timer::Timestamp Timer::start(Tag tag) { return {tag, MPI_Wtime()}; }

void Timer::stop(Timer::Timestamp timestamp) {
  double endtime = MPI_Wtime();
  cumm_times[timestamp.tag] += (endtime - timestamp.starttime);
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