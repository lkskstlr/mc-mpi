#include "timer.hpp"
#include <mpi.h>
#include <time.h>
#include <string>

double init_offset() {
  if (MPI_WTIME_IS_GLOBAL == 1) {
    return 0.0;
  }

  struct timespec time_now;
  double t1, t2;

  t1 = MPI_Wtime();
  clock_gettime(CLOCK_REALTIME, &time_now);
  t2 = MPI_Wtime();

  double offset;
  double nsec_to_sec = 1e-9;
  offset =
      (time_now.tv_sec - 0.5 * (t1 + t2)) + (nsec_to_sec * time_now.tv_nsec);
  return offset;
}

MPI_Datatype Timer::State::mpi_t() {
  constexpr int nitems = 3;
  MPI_Datatype mpi_state_type;
  int blocklengths[nitems] = {Tag::STATE_COUNT, 1, 1};
  MPI_Datatype types[nitems] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
  MPI_Aint offsets[nitems];

  offsets[0] = offsetof(State, cumm_times);
  offsets[1] = offsetof(State, starttime);
  offsets[2] = offsetof(State, endtime);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_state_type);
  return mpi_state_type;
}

int Timer::State::sprintf(char *str) {
  return ::sprintf(str, "%.18e, %.18e, %.18e, %.18e, %.18e, %.18e, ", starttime,
                   endtime, cumm_times[Timer::Tag::Computation],
                   cumm_times[Timer::Tag::Send], cumm_times[Timer::Tag::Recv],
                   cumm_times[Timer::Tag::Idle]);
}

int Timer::State::sprintf_header(char *str) {
  return ::sprintf(str, "%s, %s, %s, %s, %s, %s, ", "starttime", "endtime",
                   "time_comp", "time_send", "time_recv", "time_idle");
}

int Timer::State::sprintf_max_len() { return 6 * 30; }

Timer::Timer() : offset(init_offset()){};

Timer::Timestamp Timer::start(Tag tag) {
  state.starttime = MPI_Wtime() + offset;
  return {tag, state.starttime};
}

void Timer::change(Timestamp &timestamp, Tag tag) {
  double time = MPI_Wtime() + offset;
  state.cumm_times[timestamp.tag] += (time - timestamp.time);

  // update timestamp
  timestamp.tag = tag;
  timestamp.time = time;
}

Timer::State Timer::stop(Timer::Timestamp timestamp) {
  state.endtime = MPI_Wtime() + offset;
  state.cumm_times[timestamp.tag] += (state.endtime - timestamp.time);

  return state;
}

void Timer::reset() {
  state.starttime = 0.0;
  state.endtime = 0.0;
  for (int i = 0; i < Tag::STATE_COUNT; ++i) {
    state.cumm_times[i] = 0.0;
  }
}

Timer::State Timer::restart(Timestamp &timestamp, Tag tag) {
  // change context; current time
  change(timestamp, tag);

  // populate result
  State res = state;
  res.endtime = timestamp.time;

  // change own state
  state.starttime = timestamp.time;
  for (int i = 0; i < Tag::STATE_COUNT; ++i) {
    state.cumm_times[i] = 0.0;
  }

  // return
  return res;
}

double Timer::tick() const { return MPI_Wtick(); }
double Timer::time() const { return MPI_Wtime() + offset; }
double Timer::starttime() const { return state.starttime; }

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