#include "worker.hpp"
#include "yaml_dumper.hpp"
#include "yaml_loader.hpp"
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <time.h>
#include <unistd.h>

// From: http://www.cs.yorku.ca/~oz/hash.html
unsigned long hash(unsigned char *str)
{
  unsigned long hash = 5381;
  int c;

  while (c = *str++)
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

  return hash;
}

MCMPI_Local(int *local_size, int *local_rank)
{
  int world_rank;
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  char hostname[2048] = {0};
  gethostname(hostname, 2048);
  unsigned long my_hash = hash(hostname);
  unsigned long *hashes = (unsigned long *)malloc(sizeof(unsigned long) * world_size);

  MPI_Allgather(&my_hash, 1, MPI_UNSIGNED_LONG, hashes, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  usleep(100000 * world_rank);
  printf("%d: ", world_size);
  for (int j = 0; j < world_size; j++)
    printf("%lu ", hashes[j]);
  printf("\n");

  free(hashes);
}

Worker::Worker(int world_rank, const MCMPIOptions &options)
    : world_rank(world_rank), options(options),
      layer(decompose_domain(options.x_min, options.x_max, options.x_ini,
                             options.world_size, world_rank,
                             options.nb_cells, options.nb_particles,
                             options.particle_min_weight)),
      timer()
{
  char *slurm_job_id = getenv("SLURM_JOB_ID");
  if (slurm_job_id)
  {
    char buffer[256] = "../out/";
    foldername = std::string(strcat(buffer, slurm_job_id));
  }
  else
  {
    foldername = std::string("out");
  }
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

    char buffer[256];
    sprintf(buffer, "%s/config.yaml", foldername.c_str());
    dump_config(buffer);
    dump_weights_absorbed(total_len, displs, weights);
    layer.dump_WA();
  }

  char buffer[256];
  sprintf(buffer, "%s/stats.csv", foldername.c_str());
  write_file(buffer);

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
  // Does not seem to work, maybe the filesystem does not support it
  // MPI_File file;
  // int ret_open = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE,
  //                              MPI_INFO_NULL, &file);

  // MPI_File_set_size(file, 0);
  // MPI_Offset mpi_offset = displs[world_rank];

  // MPI_Status status;
  // int ret_write = MPI_File_write_at_all(file, mpi_offset, buf, offset, MPI_CHAR,
  //                                       &status);

  // int count;
  // MPI_Get_count(&status, MPI_CHAR, &count);

  // if (count != (int)offset)
  // {
  //   fprintf(stderr, "Abort in Worker::write_file. Number of char written not correct.");
  //   MPI_Abort(MPI_COMM_WORLD, 1);
  // }

  // MPI_Barrier(MPI_COMM_WORLD);
  // usleep(10000 * world_rank);

  // printf("r = %d, displ = %d, offset = %d, ret_open = %d, ret_write = %d, bytes_written = %d, strlen = %zu\n",
  //        world_rank, displs[world_rank], offset, ret_open, ret_write, count, strlen(buf));
  // // printf("%s\n", buf);
  // MPI_File_close(&file);
  // MPI_Barrier(MPI_COMM_WORLD);

  char *totalstring = NULL;
  if (world_rank == 0)
  {
    totalstring = (char *)malloc(sizeof(char) * totlen);
    totalstring[totlen - 1] = 0;
  }

  MPI_Gatherv(buf, offset, MPI_CHAR,
              totalstring, recvcounts, displs, MPI_CHAR,
              0, MPI_COMM_WORLD);

  if (world_rank == 0)
  {
    // printf("%c\n", totalstring[totlen - 1]);
    // printf("%s\n\n", totalstring);
    // printf("%d ; %zu\n", totlen, strlen(totalstring));

    FILE *file = fopen(filename, "w");

    int results = fputs(totalstring, file);
    if (results == EOF)
    {
      fprintf(stderr, "Abort in Worker::write_file. Couldn't write file.");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fclose(file);
  }

  MPI_Barrier(MPI_COMM_WORLD);
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

void Worker::dump_config(char *filename)
{
  YamlDumper yaml_dumper(filename);
  yaml_dumper.comment("Read from config");
  yaml_dumper.dump_int("nb_cells", options.nb_cells);
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
  DIR *dir = opendir(foldername.c_str());
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
      sprintf(filepath, "%s/%s", foldername.c_str(), next_file->d_name);
      remove(filepath);
    }

    closedir(dir);
    if (remove(foldername.c_str()))
    {
      fprintf(stderr, "Couldn't remove %s dir, is it empty?\n", foldername.c_str());
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

  mkdir(foldername.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
}

void Worker::dump_weights_absorbed(int total_len, int const *displs,
                                   real_t const *weights)
{
  FILE *file;
  char buffer[256];
  sprintf(buffer, "%s/weights.csv", foldername.c_str());
  file = fopen(buffer, "w");
  if (!file)
  {
    fprintf(stderr, "Couldn't open file %s/weights.csv for writing.\n", foldername.c_str());
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
  opt.nb_cells = yaml_loader.load_int("nb_cells");
  opt.x_min = yaml_loader.load_double("x_min");
  opt.x_max = yaml_loader.load_double("x_max");
  opt.x_ini = yaml_loader.load_double("x_ini");
  opt.particle_min_weight = yaml_loader.load_double("particle_min_weight");
  opt.nb_particles = yaml_loader.load_int("nb_particles");
  opt.cycle_time = yaml_loader.load_double("cycle_time");
  opt.nb_particles_per_cycle = yaml_loader.load_int("nb_particles_per_cycle");
  opt.nthread = yaml_loader.load_int("nthread");
  opt.statistics_cycle_time = yaml_loader.load_double("statistics_cycle_time");

  return opt;
}
