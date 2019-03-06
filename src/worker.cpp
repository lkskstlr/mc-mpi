#include "worker.hpp"
#include "yaml_dumper.hpp"
#include "yaml_loader.hpp"
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <numeric>
#include <stdio.h>
#include <mpi.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <time.h>
#include <unistd.h>

float get_power()
{
  return 1;
}

Layer mpi_decompose_domain(MCMPIOptions const &options)
{
  //MPI_Status status;

  int my_rank, world_size;

  float r_min, r_max;
  int cell_min, cell_max;

  int nb_cells = 1000;
  int cell_weights[1000];
  int cell_weights_sum;
  int cell_weights_so_far;

  float computation_power_own;
  float *computation_power_all;
  float computation_power_all_sum;
  float computation_power_so_far;

  int i;
  int flag_min;
  int flag_max;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  printf("[%d]Debugging\t\tWorld size: %d\t My rank: %d\n", my_rank, world_size, my_rank);

  //Hardcode cell weights and get the total sum
  for (i = 0, cell_weights_sum = 0; i < nb_cells; i++)
  {
    if (i != 707)
      cell_weights[i] = 1;
    else
      cell_weights[i] = 100;

    // cell_weights[i] = i;

    cell_weights_sum += cell_weights[i];
  }

  printf("\n[%d]Debugging\t\tCells: %d\tCell weights sum: %d\n", my_rank, nb_cells, cell_weights_sum);

  //Get own computing power and allocate memory for all of the other ranks' computer power
  computation_power_own = get_power();
  computation_power_all = (float *)malloc(world_size * sizeof(float));

  //All ranks communicate among themselves the computing power of each
  MPI_Allgather(&computation_power_own, 1, MPI_FLOAT, computation_power_all, 1, MPI_FLOAT, MPI_COMM_WORLD);

  // printf("[%d]Debugging\t\tOwn computation power: %f\n", my_rank, computation_power_own);
  // for (i = 0; i < world_size; i++)
  // {
  //   printf("[%d]Computation power rank %d: %f\n", my_rank, i, computation_power_all[i]);
  // }

  //Get the sum of all compute powers
  for (i = 0, computation_power_all_sum = 0; i < world_size; i++)
    computation_power_all_sum += computation_power_all[i];

  //Get the addition of the compute powers so far, EXCLUDING own's
  for (i = 0, computation_power_so_far = 0; i < my_rank; i++)
    computation_power_so_far += computation_power_all[i];

  r_min = (cell_weights_sum / computation_power_all_sum) * computation_power_so_far;

  //Get the addition of the compute powers so far, INCLUDING own's
  computation_power_so_far += computation_power_all[my_rank];

  r_max = (cell_weights_sum / computation_power_all_sum) * computation_power_so_far;

  for (i = 0, cell_weights_so_far = 0, flag_min = 0, flag_max = 0; !(flag_min && flag_max); i++)
  {
    if (!flag_min && cell_weights_so_far >= r_min)
    {
      cell_min = i;
      flag_min = 1;
    }
    if (!flag_max && cell_weights_so_far >= r_max)
    {
      cell_max = i;
      flag_max = 1;
    }
    cell_weights_so_far += cell_weights[i];
  }

  float x_min_layer = (float)cell_min / (float)nb_cells;
  float x_max_layer = (float)cell_max / (float)nb_cells;

  Layer layer(x_min_layer, x_max_layer, cell_min, cell_max - cell_min, options.particle_min_weight);
  if ((x_min_layer <= options.x_ini) && (options.x_ini < x_max_layer))
  {
    printf("World rank: %d / %d\n", my_rank, world_size);
    seed_t seed = 5127801;
    layer.create_particles(options.x_ini, 1.0 / options.nb_particles, options.nb_particles, seed);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  usleep(10000 * my_rank);
  printf("[%d]r_min: %f\tr_max: %f\tx_min: %f\tx_max: %f\tcell_min: %d\tcell_max: %d\n", my_rank, r_min, r_max, layer.x_min, layer.x_max, cell_min, cell_max);
  printf("[%d]Layer: index_start = %d, m = %d, x_min = %f, x_max = %f\n", my_rank, layer.index_start, layer.m, layer.x_min, layer.x_max);

  return layer;
}

Worker::Worker(int world_rank, const MCMPIOptions &options)
    : world_rank(world_rank), options(options), layer(mpi_decompose_domain(options)), timer()
{
}

void Worker::dump()
{
  int total_len = 0;
  int *displs = NULL;
  real_t *weights = NULL;

  gather_weights_absorbed(&total_len, &displs, &weights);

  if (world_rank == 0)
  {
    mkdir_out();

    dump_config();
    dump_weights_absorbed(total_len, displs, weights);
    layer.dump_WA();
  }

  write_file((char *)"out/stats.csv");

  MPI_Barrier(MPI_COMM_WORLD);
}

void Worker::write_file(char *filename)
{
  // write times
  size_t max_len = static_cast<size_t>(Timer::State::sprintf_max_len()) *
                       (timer_states.size() + 1) +
                   static_cast<size_t>(Stats::State::sprintf_max_len()) *
                       (stats_states.size() + 1) +
                   10 * (stats_states.size() + 1) + 1000;

  char *buf = (char *)malloc(max_len);
  int offset = 0;

  if (world_rank == 0)
  {
    offset += sprintf(buf + offset, "rank, ");
    if (timer_states.size() > 0)
      offset += Timer::State::sprintf_header(buf + offset);
    if (stats_states.size() > 0)
      offset += Stats::State::sprintf_header(buf + offset);
    offset += sprintf(buf + offset, "nb_cycles, \n");
  }

  for (int i = 0; i < (int)timer_states.size(); ++i)
  {
    offset += sprintf(buf + offset, "%d, ", world_rank);
    if (timer_states.size() > 0)
      offset += timer_states[i].sprintf(buf + offset);
    if (stats_states.size() > 0)
      offset += stats_states[i].sprintf(buf + offset);
    offset += sprintf(buf + offset, "%d, \n", cycle_states[i]);
  }

  if ((size_t)offset >= max_len)
  {
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

  for (int i = 1; i < options.world_size; i++)
  {
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
                                     real_t **weights)
{
  int *recvcounts = NULL;

  if (world_rank == 0)
  {
    recvcounts = (int *)malloc(options.world_size * sizeof(int));
  }

  int my_len = static_cast<int>(layer.weights_absorbed.size());
  MPI_Gather(&my_len, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  *total_len = 0;

  if (world_rank == 0)
  {
    *displs = (int *)malloc(options.world_size * sizeof(int));

    (*displs)[0] = 0;
    *total_len += recvcounts[0];

    for (int i = 1; i < options.world_size; i++)
    {
      *total_len += recvcounts[i];
      (*displs)[i] = (*displs)[i - 1] + recvcounts[i - 1];
    }

    *weights = (real_t *)malloc((*total_len) * sizeof(real_t));
  }

  MPI_Gatherv(layer.weights_absorbed.data(), my_len, MCMPI_REAL_T, *weights,
              recvcounts, *displs, MCMPI_REAL_T, 0, MPI_COMM_WORLD);
}

void Worker::dump_config()
{
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
  yaml_dumper.dump_int("nb_particles_per_cycle",
                       options.nb_particles_per_cycle);
  yaml_dumper.dump_int("nthread", options.nthread);
  yaml_dumper.dump_double("statistics_cycle_time",
                          options.statistics_cycle_time);
  yaml_dumper.new_line();
  yaml_dumper.comment("Other values");
  yaml_dumper.dump_int("world_size", options.world_size);
  char _hostname[1000];
  gethostname(_hostname, 1000);
  yaml_dumper.dump_string("hostname", _hostname);
}

std::vector<real_t> Worker::weights_absorbed()
{
  return layer.weights_absorbed;
}

void Worker::mkdir_out()
{
  DIR *dir = opendir("out");
  if (dir)
  {
    struct dirent *next_file;
    char filepath[512];

    while ((next_file = readdir(dir)) != NULL)
    {
      if (0 == strcmp(next_file->d_name, ".") ||
          0 == strcmp(next_file->d_name, ".."))
      {
        continue;
      }
      sprintf(filepath, "%s/%s", "out", next_file->d_name);
      remove(filepath);
    }

    closedir(dir);
    if (remove("out"))
    {
      fprintf(stderr, "Couldn't remove out dir, is it empty?\n");
      exit(1);
    }
  }
  else if (ENOENT == errno)
  {
    /* Directory does not exist. */
  }
  else
  {
    fprintf(stderr, "opendir failed.\n");
    exit(1);
  }

  mkdir("out", S_IRWXU | S_IRWXG | S_IRWXO);
}

void Worker::dump_weights_absorbed(int total_len, int const *displs,
                                   real_t const *weights)
{
  FILE *file;
  file = fopen("out/weights.csv", "w");
  if (!file)
  {
    fprintf(stderr, "Couldn't open file out/weights.csv for writing.\n");
    exit(1);
  }

  fprintf(file, "proc, x, weight\n");
  int proc = 0;
  for (int i = 0; i < total_len; ++i)
  {
    if (proc < options.world_size - 1 && displs[proc + 1] == i)
    {
      proc++;
    }
    fprintf(file, "%d, %.18e, %.18e\n", proc, layer.dx * (i + 0.5),
            weights[i] / layer.dx);
  }

  fclose(file);
}

MCMPIOptions options_from_config(std::string filepath, int world_size)
{
  YamlLoader yaml_loader(filepath);
  // // constants
  MCMPIOptions opt;
  opt.world_size = world_size;
  //opt.nb_cells_per_layer = yaml_loader.load_int("nb_cells_per_layer");
  if (1000 % world_size != 0)
  {
    fprintf(stderr, "1000 is not divideable by world_size\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  opt.nb_cells_per_layer = 1000 / world_size;
  opt.x_min = yaml_loader.load_double("x_min");
  opt.x_max = yaml_loader.load_double("x_max");
  opt.x_ini = yaml_loader.load_double("x_ini");
  opt.particle_min_weight = yaml_loader.load_double("particle_min_weight");
  opt.nb_particles = yaml_loader.load_int("nb_particles");
  opt.buffer_size = yaml_loader.load_int("buffer_size");
  opt.cycle_time = yaml_loader.load_double("cycle_time");
  opt.nb_particles_per_cycle = yaml_loader.load_int("nb_particles_per_cycle");
  opt.nthread = yaml_loader.load_int("nthread");
  opt.statistics_cycle_time = yaml_loader.load_double("statistics_cycle_time");

  return opt;
}
