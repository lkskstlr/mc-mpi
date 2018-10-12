#include "worker.hpp"
#include "yaml_dumper.hpp"
#include "yaml_loader.hpp"
#include <chrono>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <time.h>
#include <unistd.h>

using std::size_t;

Worker::Worker(int world_rank, const MCMPIOptions &options)
    : world_rank(world_rank), options(options),
      layer(decompose_domain(options.x_min, options.x_max, options.x_ini,
                             options.world_size, world_rank,
                             options.nb_cells_per_layer, options.nb_particles,
                             options.particle_min_weight)),
      particle_comm(world_rank, options.buffer_size) {

  /* Time */
  if (world_rank == 0) {
    unix_timestamp_start = (unsigned long)time(NULL);
  }
  /* reserve */
  timer_states.reserve(100);

  /* Commit Timer::State Type */
  timer_state_mpi_t = Timer::State::mpi_t();
  MPI_Type_commit(&timer_state_mpi_t);

  /* Event as int */
  event_comm.init(world_rank, MPI_INT, options.buffer_size);
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
  while (flag) {
    start = high_resolution_clock::now();

    /* Simulate Particles */
    timer.change(timestamp, Timer::Tag::Computation);
    {
      layer.simulate(options.cycle_nb_steps, particles_left, particles_right,
                     particles_disabled);
    }

    /* Sending Particles */
    timer.change(timestamp, Timer::Tag::Send);
    {
      if (world_rank == 0) {

        particle_comm.send(particles_right, world_rank + 1,
                           MCMPIOptions::Tag::Particle);
        particles_right.clear();
      } else if (world_rank == options.world_size - 1) {

        particle_comm.send(particles_left, world_rank - 1,
                           MCMPIOptions::Tag::Particle);
        particles_left.clear();
      } else {

        particle_comm.send(particles_left, world_rank - 1,
                           MCMPIOptions::Tag::Particle);
        particles_left.clear();
        particle_comm.send(particles_right, world_rank + 1,
                           MCMPIOptions::Tag::Particle);
        particles_right.clear();
      }
    }

    /* Receive Particles */
    timer.change(timestamp, Timer::Tag::Recv);
    {
      // recv
      particle_comm.recv(layer.particles, MPI_ANY_SOURCE,
                         MCMPIOptions::Tag::Particle);
    }

    /* Receive Events */
    timer.change(timestamp, Timer::Tag::Recv);
    {
      if (world_rank == 0) {
        vec_nb_particles_disabled[0] = particles_left.size();

        std::vector<int> tmp_vec;
        for (int source = 1; source < options.world_size; ++source) {
          tmp_vec.clear();
          if (event_comm.recv(tmp_vec, source, MCMPIOptions::Tag::Disable) &&
              !tmp_vec.empty()) {
            vec_nb_particles_disabled[source] = tmp_vec[0];
          }
        }

        int nb_total_disabled = 0;
        for (auto nb : vec_nb_particles_disabled) {
          nb_total_disabled += nb;
        }

        if (nb_total_disabled == options.nb_particles) {
          // end of simulation
          flag = false;
        }
      } else {
        std::vector<int> finished_vec;
        if (event_comm.recv(finished_vec, 0, MCMPIOptions::Tag::Finish)) {
          // end of simulation
          flag = false;
        }
      }
    }

    /* Send Events */
    timer.change(timestamp, Timer::Tag::Send);
    {
      if (world_rank == 0) {
        if (!flag) {
          for (int i = 1; i < options.world_size; ++i) {
            event_comm.send(1, i, MCMPIOptions::Tag::Finish);
          }
        }
      } else {
        int nb_particles_disabled = particles_disabled.size();
        if (world_rank == options.world_size - 1) {
          nb_particles_disabled += particles_right.size();
        }
        if (nb_particles_disabled > 0) {
          event_comm.send(nb_particles_disabled, 0, MCMPIOptions::Tag::Disable);
        }
      }
    }

    /* Timer State */
    if (timer.time() > timer.starttime() + options.statistics_cycle_time) {
      // dump
      timer_states.push_back(timer.restart(timestamp, Timer::Tag::Idle));
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
  }

  timer_states.push_back(timer.stop(timestamp));
  return;
}

void Worker::dump() {
  int total_len = 0;
  int *displs = NULL;
  Timer::State *states = NULL;

  gather_times(&total_len, &displs, &states);

  if (world_rank == 0) {
    mkdir_out();

    dump_config();
    dump_times(total_len, displs, states);
  }

  free(displs);
  free(states);
}

void Worker::gather_times(int *total_len, int **displs, Timer::State **states) {
  int *recvcounts = NULL;

  if (world_rank == 0) {
    recvcounts = (int *)malloc(options.world_size * sizeof(int));
  }

  int my_len = static_cast<int>(timer_states.size());
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

    *states = (Timer::State *)malloc((*total_len) * sizeof(Timer::State));
  }

  MPI_Gatherv(timer_states.data(), my_len, timer_state_mpi_t, *states,
              recvcounts, *displs, timer_state_mpi_t, 0, MPI_COMM_WORLD);
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

void Worker::dump_config() {
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
}

std::vector<real_t> Worker::weights_absorbed() {
  return layer.weights_absorbed;
}

void Worker::mkdir_out() {
  char filename[100] = "out/times.csv";
  if (access(filename, F_OK) != -1) {
    if (remove(filename)) {
      fprintf(stderr, "Couldn't remove '%s'\n", filename);
      exit(1);
    }
  }

  strncpy(filename, "out/config.yaml", 100);
  if (access(filename, F_OK) != -1) {
    if (remove(filename)) {
      fprintf(stderr, "Couldn't remove '%s'\n", filename);
      exit(1);
    }
  }

  DIR *dir = opendir("out");
  if (dir) {
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

void Worker::dump_times(int total_len, int const *displs,
                        Timer::State const *states) {
  FILE *file;
  file = fopen("out/times.csv", "w");
  if (!file) {
    fprintf(stderr, "Couldn't open file out/times.csv for writing.\n");
    exit(1);
  }

  fprintf(file, "proc, starttime, endtime, state_comp, state_send, "
                "state_recv, state_idle\n");
  int proc = 0;
  for (int i = 0; i < total_len; ++i) {
    if (proc < options.world_size - 1 && displs[proc + 1] == i) {
      proc++;
    }
    fprintf(file, "%d, %.18e, %.18e, %.18e, %.18e, %.18e, %.18e\n", proc,
            states[i].starttime, states[i].endtime,
            states[i].cumm_times[Timer::Tag::Computation],
            states[i].cumm_times[Timer::Tag::Send],
            states[i].cumm_times[Timer::Tag::Recv],
            states[i].cumm_times[Timer::Tag::Idle]);
  }

  fclose(file);
}

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
    fprintf(file, "%d, %.18e, %.18e\n", proc, 0.1, 0.1);
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
