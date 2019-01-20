#include "worker.hpp"
#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <numeric>
#include <thread>
#include "yaml_dumper.hpp"
#include "yaml_loader.hpp"

Worker::Worker(int world_rank, const MCMPIOptions &options)
    : world_rank(world_rank),
      options(options),
      layer(decompose_domain(options.x_min, options.x_max, options.x_ini,
                             options.world_size, world_rank,
                             options.nb_cells_per_layer, options.nb_particles,
                             options.particle_min_weight)),
      particle_comm(world_rank, options.buffer_size),
      state_comm(options.world_size, world_rank, MCMPIOptions::Tag::State,
                 [nb_particles = options.nb_particles](std::vector<int> msgs) {
                   int sum = std::accumulate(msgs.begin(), msgs.end(), 0);
                   if (sum == nb_particles) {
                     return StateComm::State::Finished;
                   }
                   return StateComm::State::Running;
                 }),
      timer() {
  /* Time */
  if (world_rank == 0) {
    unix_timestamp_start = (unsigned long)time(NULL);
  }
  /* reserve */
  timer_states.reserve(100);
  stats_states.reserve(100);
  cycle_states.reserve(100);

  /* MPI_Wtime synchronization */
  printf("MPI_Wtick = %e\n", MPI_Wtick());
  printf("rank = %d, timer = %e, MPI_Wtime = %e\n", world_rank, timer::time(),
         MPI_Wtime());
}

void Worker::spin() {
  using std::chrono::high_resolution_clock;

  auto timestamp = timer.start(Timer::Tag::Idle);

  auto start = high_resolution_clock::now();
  auto finish = high_resolution_clock::now();
  std::vector<int> vec_nb_particles_disabled;
  if (world_rank == 0) {
    vec_nb_particles_disabled = std::vector<int>(options.world_size, 0);
  }

  bool flag = true;
  int nb_cycles_stats = 0;
  while (flag) {
    start = high_resolution_clock::now();

    /* Simulate Particles */
    timer.change(timestamp, Timer::Tag::Computation);
    {
      layer.simulate(options.cycle_nb_steps, particles_left, particles_right,
                     particles_disabled);

      if (world_rank == 0) {
        particles_disabled.insert(particles_disabled.end(),
                                  particles_left.begin(), particles_left.end());
        particles_left.clear();
      }
      if (world_rank == options.world_size - 1) {
        particles_disabled.insert(particles_disabled.end(),
                                  particles_right.begin(),
                                  particles_right.end());
        particles_right.clear();
      }
    }

    /* Sending Particles */
    timer.change(timestamp, Timer::Tag::Send);
    {
      particle_comm.send(particles_left, world_rank - 1,
                         MCMPIOptions::Tag::Particle);
      particle_comm.send(particles_right, world_rank + 1,
                         MCMPIOptions::Tag::Particle);
    }

    /* Receive Particles */
    timer.change(timestamp, Timer::Tag::Recv);
    {
      // recv
      particle_comm.recv(layer.particles, MPI_ANY_SOURCE,
                         MCMPIOptions::Tag::Particle);
    }

    /* Send Events */
    timer.change(timestamp, Timer::Tag::Send);
    {
      state_comm.send_msg(particles_disabled.size());
      state_comm.send_state();
    }

    /* Receive Events */
    timer.change(timestamp, Timer::Tag::Recv);
    {
      if (state_comm.recv_state() == StateComm::State::Finished) {
        flag = false;
        break;
      }
    }

    /* Timer State */
    if (timer.time() > timer.starttime() + options.statistics_cycle_time) {
      timer_states.push_back(timer.restart(timestamp, Timer::Tag::Idle));
      stats_states.push_back(particle_comm.reset_stats() +
                             state_comm.reset_stats());
      cycle_states.push_back(nb_cycles_stats);
      nb_cycles_stats = 0;
    }

    /* Idle */
    timer.change(timestamp, Timer::Tag::Idle);
    {
      finish = high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = finish - start;
      if (elapsed.count() < options.cycle_time * 1e3) {
        auto sleep_length = std::chrono::duration<double, std::milli>(
                                options.cycle_time * 1e3) -
                            elapsed;
        std::this_thread::sleep_for(sleep_length);
      }
    }

    nb_cycles_stats++;
  }

  timer_states.push_back(timer.stop(timestamp));
  stats_states.push_back(particle_comm.reset_stats() +
                         state_comm.reset_stats());
  cycle_states.push_back(nb_cycles_stats);
  return;
}

void Worker::dump() {
  int total_len = 0;
  int *displs = NULL;
  real_t *weights = NULL;

  gather_weights_absorbed(&total_len, &displs, &weights);

  // Max Used Buffersize
  unsigned long max_used_buffer = particle_comm.get_max_used_buffer();
  unsigned long max_used_buffer_global;
  MPI_Reduce(&max_used_buffer, &max_used_buffer_global, 1,
             MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    mkdir_out();

    dump_config(max_used_buffer_global);
    dump_weights_absorbed(total_len, displs, weights);
    layer.dump_WA();
  }

  write_file((char *)"out/stats.csv");

  MPI_Barrier(MPI_COMM_WORLD);
}

void Worker::write_file(char *filename) {
  // write times
  size_t max_len = static_cast<size_t>(Timer::State::sprintf_max_len()) *
                       (timer_states.size() + 1) +
                   static_cast<size_t>(Stats::State::sprintf_max_len()) *
                       (stats_states.size() + 1) +
                   10 * (stats_states.size() + 1) + 1000;

  char *buf = (char *)malloc(max_len);
  int offset = 0;

  if (world_rank == 0) {
    offset += sprintf(buf + offset, "rank, ");
    offset += Timer::State::sprintf_header(buf + offset);
    offset += Stats::State::sprintf_header(buf + offset);
    offset += sprintf(buf + offset, "nb_cycles, \n");
  }

  for (int i = 0; i < timer_states.size(); ++i) {
    offset += sprintf(buf + offset, "%d, ", world_rank);
    offset += timer_states[i].sprintf(buf + offset);
    offset += stats_states[i].sprintf(buf + offset);
    offset += sprintf(buf + offset, "%d, \n", cycle_states[i]);
  }

  if (offset >= max_len) {
    fprintf(stderr, "Abort in Worker::write_file");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int *recvcounts = NULL;
  recvcounts = (int *)malloc(options.world_size * sizeof(int));

  MPI_Allgather(&offset, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

  int totlen = 0;
  int *displs = NULL;

  displs = (int *)malloc(options.world_size * sizeof(int));

  displs[0] = 0;
  totlen += recvcounts[0] + 1;

  for (int i = 1; i < options.world_size; i++) {
    totlen += recvcounts[i];
    displs[i] = displs[i - 1] + recvcounts[i - 1];
  }

  // Write to collective file
  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                MPI_INFO_NULL, &file);

  MPI_File_write_at_all(file, displs[world_rank], buf, offset, MPI_CHAR,
                        MPI_STATUS_IGNORE);

  MPI_File_close(&file);
}

void Worker::gather_weights_absorbed(int *total_len, int **displs,
                                     real_t **weights) {
  int *recvcounts = NULL;

  if (world_rank == 0) {
    recvcounts = (int *)malloc(options.world_size * sizeof(int));
  }

  int my_len = static_cast<int>(layer.weights_absorbed.size());
  MPI_Gather(&my_len, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  *total_len = 0;

  if (world_rank == 0) {
    *displs = (int *)malloc(options.world_size * sizeof(int));

    (*displs)[0] = 0;
    *total_len += recvcounts[0];

    for (int i = 1; i < options.world_size; i++) {
      *total_len += recvcounts[i];
      (*displs)[i] = (*displs)[i - 1] + recvcounts[i - 1];
    }

    *weights = (real_t *)malloc((*total_len) * sizeof(real_t));
  }

  MPI_Gatherv(layer.weights_absorbed.data(), my_len, MCMPI_REAL_T, *weights,
              recvcounts, *displs, MCMPI_REAL_T, 0, MPI_COMM_WORLD);
}

void Worker::dump_config(int max_used_buffer) {
  YamlDumper yaml_dumper("out/config.yaml");
  yaml_dumper.comment("Read from config");
  yaml_dumper.dump_int("nb_cells_per_layer", options.nb_cells_per_layer);
  yaml_dumper.dump_double("x_min", options.x_min);
  yaml_dumper.dump_double("x_max", options.x_max);
  yaml_dumper.dump_double("x_ini", options.x_ini);
  yaml_dumper.dump_double("particle_min_weight", options.particle_min_weight);
  yaml_dumper.dump_int("nb_particles", options.nb_particles);
  yaml_dumper.dump_int("buffer_size", options.buffer_size);
  yaml_dumper.dump_double("cycle_time", options.cycle_time);
  yaml_dumper.dump_int("cycle_nb_steps", options.cycle_nb_steps);
  yaml_dumper.dump_double("statistics_cycle_time",
                          options.statistics_cycle_time);
  yaml_dumper.new_line();
  yaml_dumper.comment("Other values");
  yaml_dumper.dump_int("world_size", options.world_size);
  yaml_dumper.dump_unsigned_long("unix_timestamp_start", unix_timestamp_start);
  char _hostname[1000];
  gethostname(_hostname, 1000);
  yaml_dumper.dump_string("hostname", _hostname);
  yaml_dumper.dump_int("max_used_buffer", max_used_buffer);
}

std::vector<real_t> Worker::weights_absorbed() {
  return layer.weights_absorbed;
}

void Worker::mkdir_out() {
  DIR *dir = opendir("out");
  if (dir) {
    struct dirent *next_file;
    char filepath[256];

    while ((next_file = readdir(dir)) != NULL) {
      if (0 == strcmp(next_file->d_name, ".") ||
          0 == strcmp(next_file->d_name, "..")) {
        continue;
      }
      sprintf(filepath, "%s/%s", "out", next_file->d_name);
      remove(filepath);
    }

    closedir(dir);
    if (remove("out")) {
      fprintf(stderr, "Couldn't remove out dir, is it empty?\n");
      exit(1);
    }
  } else if (ENOENT == errno) {
    /* Directory does not exist. */
  } else {
    fprintf(stderr, "opendir failed.\n");
    exit(1);
  }

  mkdir("out", S_IRWXU | S_IRWXG | S_IRWXO);
}

// void Worker::dump_stats(int total_len, int const *displs,
//                         Timer::State const *states) {
//   FILE *file;
//   file = fopen("out/times.csv", "w");
//   if (!file) {
//     fprintf(stderr, "Couldn't open file out/times.csv for writing.\n");
//     exit(1);
//   }

//   fprintf(file, "proc, starttime, endtime, state_comp, state_send, "
//                 "state_recv, state_idle\n");
//   int proc = 0;
//   for (int i = 0; i < total_len; ++i) {
//     if (proc < options.world_size - 1 && displs[proc + 1] == i) {
//       proc++;
//     }
//     fprintf(file, "%d, %.18e, %.18e, %.18e, %.18e, %.18e, %.18e\n", proc,
//             states[i].starttime, states[i].endtime,
//             states[i].cumm_times[Timer::Tag::Computation],
//             states[i].cumm_times[Timer::Tag::Send],
//             states[i].cumm_times[Timer::Tag::Recv],
//             states[i].cumm_times[Timer::Tag::Idle]);
//   }

//   fclose(file);
// }

void Worker::dump_weights_absorbed(int total_len, int const *displs,
                                   real_t const *weights) {
  FILE *file;
  file = fopen("out/weights.csv", "w");
  if (!file) {
    fprintf(stderr, "Couldn't open file out/weights.csv for writing.\n");
    exit(1);
  }

  fprintf(file, "proc, x, weight\n");
  int proc = 0;
  real_t x_pos = layer.dx;
  for (int i = 0; i < total_len; ++i) {
    if (proc < options.world_size - 1 && displs[proc + 1] == i) {
      proc++;
    }
    fprintf(file, "%d, %.18e, %.18e\n", proc, layer.dx * (i + 0.5),
            weights[i] / layer.dx);
  }

  fclose(file);
}

Worker worker_from_config(std::string filepath, int world_size,
                          int world_rank) {
  YamlLoader yaml_loader(filepath);
  // // constants
  MCMPIOptions opt;
  opt.world_size = world_size;
  opt.nb_cells_per_layer = yaml_loader.load_int("nb_cells_per_layer");
  opt.x_min = yaml_loader.load_double("x_min");
  opt.x_max = yaml_loader.load_double("x_max");
  opt.x_ini = yaml_loader.load_double("x_ini");
  opt.particle_min_weight = yaml_loader.load_double("particle_min_weight");
  opt.nb_particles = yaml_loader.load_int("nb_particles");
  opt.buffer_size = yaml_loader.load_int("buffer_size");
  opt.cycle_time = yaml_loader.load_double("cycle_time");
  opt.cycle_nb_steps = yaml_loader.load_int("cycle_nb_steps");
  opt.statistics_cycle_time = yaml_loader.load_double("statistics_cycle_time");

  return Worker(world_rank, opt);
}
