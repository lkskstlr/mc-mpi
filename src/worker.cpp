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
#include <math.h>

#define WORKER_MPI_DECOMPOSE_DOMAIN_PRINT 0
#define WORKER_MPI_DECOMPOSE_DOMAIN_DIFFERENT_THREADS 0

float get_power(int nthread)
{
  using std::chrono::high_resolution_clock;

  constexpr real_t x_min = 0.0;
  constexpr real_t x_max = 1.0;
  const real_t x_ini = sqrtf(2.0) / 2.0;
  constexpr int world_size = 1;
  constexpr int world_rank = 0;
  constexpr int nb_cells_per_layer = 1000;
  constexpr int nb_particles = 100000;
  constexpr real_t particle_min_weight = 0.0;

  Layer layer(decompose_domain(x_min, x_max, x_ini, world_size, world_rank,
                               nb_cells_per_layer, nb_particles,
                               particle_min_weight));

  auto start = high_resolution_clock::now();
  layer.simulate(-1, nthread);
  auto finish = high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = finish - start;
  return 10000 / elapsed.count();
}

Layer mpi_decompose_domain(MCMPIOptions &options)
{

  int my_rank, world_size;

  float r_min, r_max;
  int cell_min = 0, cell_max = 0;

  int nb_cells = 1000;
  int cell_weights[1000];
  /*
  int cell_weights[1000] = {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                            43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
                            43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44,
                            44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
                            44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45,
                            45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                            45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                            46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                            46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
                            47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
                            47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
                            48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
                            49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49,
                            49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50,
                            50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                            50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51,
                            51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51,
                            51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52,
                            52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,
                            52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53,
                            53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53,
                            53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 54,
                            54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
                            54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
                            55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
                            55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
                            55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
                            56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
                            56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
                            57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
                            57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                            58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                            58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59,
                            59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59,
                            59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60,
                            60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                            60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61,
                            61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
                            61, 61, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62,
                            62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62,
                            62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63,
                            63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
                            63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64,
                            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                            64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65,
                            65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
                            65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66,
                            66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
                            66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 67,
                            67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
                            67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
                            68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69,
                            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
                            69, 69, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
                            70, 70, 70, 70, 71, 2000, 70, 70, 70, 70, 70, 70, 70,
                            70, 70, 70, 70, 70, 70, 69, 69, 69, 69, 69, 69, 69,
                            69, 69, 69, 69, 69, 69, 69, 69, 69, 68, 68, 68, 68,
                            68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
                            68, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
                            67, 67, 67, 67, 67, 67, 67, 67, 66, 66, 66, 66, 66,
                            66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
                            66, 66, 66, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
                            65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
                            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 63, 63,
                            63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
                            63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 62, 62,
                            62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62,
                            62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62,
                            61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
                            61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
                            61, 61, 61, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                            60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                            60, 60, 60, 60, 60, 60, 60, 59, 59, 59, 59, 59, 59,
                            59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59,
                            59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59,
                            58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58};
                            */
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

  if (WORKER_MPI_DECOMPOSE_DOMAIN_DIFFERENT_THREADS)
    options.nthread = (my_rank % 8) + 1;
  else
    options.nthread = -1;

  int m_left = 41;
  int m_right = 41;
  int d = 2100;
  int h = 71;
  //Hardcode cell weights and get the total sum
  /*
  cell_weights is a function according to the above parameters, m_left, m_right, d and h.
  Left of cell 707 (initial cell for particles), the function is a straight line with slope
  m_left and at cell 707 it should be h. To the right of cell 707 it is a straight line
  with slope minus m_right which also should be h at cell 707. Cell 707's weight is d
  (this represents a delta for the computation it takes to create a new particle)
  */
  for (i = 0, cell_weights_sum = 0; i < nb_cells; i++)
  {
    if (i < 707)
      cell_weights[i] = h - (int)(((float)(707 - i) / (float)1000) * m_left);
    else if (i > 707)
      cell_weights[i] = h - (int)(((float)(i - 707) / (float)1000) * m_right);
    else
      cell_weights[i] = d;

    cell_weights_sum += cell_weights[i];
  }

  //Get own computing power and allocate memory for all of the other ranks' computer power
  computation_power_own = get_power(options.nthread);
  computation_power_all = (float *)malloc(world_size * sizeof(float));

  //All ranks communicate among themselves the computing power of each
  MPI_Allgather(&computation_power_own, 1, MPI_FLOAT, computation_power_all, 1, MPI_FLOAT, MPI_COMM_WORLD);

  //Get the sum of all compute powers
  for (i = 0, computation_power_all_sum = 0; i < world_size; i++)
    computation_power_all_sum += computation_power_all[i];

  //Get the addition of the compute powers so far, EXCLUDING own's
  for (i = 0, computation_power_so_far = 0; i < my_rank; i++)
    computation_power_so_far += computation_power_all[i];

  //Calculate the minumum number of standarized work load
  r_min = (cell_weights_sum / computation_power_all_sum) * computation_power_so_far;

  //Get the addition of the compute powers so far, INCLUDING own's
  computation_power_so_far += computation_power_all[my_rank];

  //Calculate the maximum number of standarized work load
  r_max = (cell_weights_sum / computation_power_all_sum) * computation_power_so_far;

  //The starting cell for the domain is when enough cells before reach the minimum amount of standarized work load
  //The ending cell is when the maximum amount of standarized work load is reached
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
    if (i == nb_cells - 1)
    {
      cell_max = nb_cells;
      flag_max = 1;
    }
    cell_weights_so_far += cell_weights[i];
  }

  float x_min_layer = (float)cell_min / (float)nb_cells;
  float x_max_layer = (float)cell_max / (float)nb_cells;

  //Create the domain and check to see if it is the one that generates the particles
  Layer layer(x_min_layer, x_max_layer, cell_min, cell_max - cell_min, options.particle_min_weight);
  if ((x_min_layer <= options.x_ini) && (options.x_ini < x_max_layer))
  {
    if (WORKER_MPI_DECOMPOSE_DOMAIN_PRINT)
      printf("World rank: %d / %d\n", my_rank, world_size);
    seed_t seed = 5127801;
    layer.create_particles(options.x_ini, 1.0 / options.nb_particles, options.nb_particles, seed);
  }
  int work_load = 0;
  for (i = cell_min; i < cell_max; i++)
  {
    work_load += cell_weights[i];
  }

  if (WORKER_MPI_DECOMPOSE_DOMAIN_PRINT)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(20000 * my_rank);
    if (my_rank == 0)
    {
      printf("\nDebugging\t\tCells: %d\tCell weights sum: %d\n", nb_cells, cell_weights_sum);
      // for (i = 0; i < 1000; i += 10)
      //   printf("Cell_weights[%d] = %d\n", i, cell_weights[i]);
    }
    printf("[%d]Computation power: %f\n", my_rank, computation_power_own);
    printf("[%d]r_min: %f\tr_max: %f\tx_min: %f\tx_max: %f\tcell_min: %d\tcell_max: %d\n", my_rank, r_min, r_max, layer.x_min, layer.x_max, cell_min, cell_max);
    printf("[%d]Layer: index_start = %d, m = %d, x_min = %f, x_max = %f\n", my_rank, layer.index_start, layer.m, layer.x_min, layer.x_max);
    printf("[%d]Work load = %d\n", my_rank, work_load);
  }
  return layer;
}

Worker::Worker(int world_rank, MCMPIOptions &options)
    : world_rank(world_rank), options(options), layer(mpi_decompose_domain(options)), timer()
{
  if (WORKER_MPI_DECOMPOSE_DOMAIN_DIFFERENT_THREADS)
    options.nthread = (world_rank % 8) + 1;
  else
    options.nthread = -1;
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
