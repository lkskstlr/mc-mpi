#include "yaml_loader.hpp"
#include <stdlib.h>

YamlLoader::YamlLoader(std::string filepath) {
  FILE *fp;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;

  fp = fopen(filepath.c_str(), "r");
  if (fp == NULL) {
    fprintf(stderr, "Couldn't open Yaml File %s.\n", filepath.c_str());
    exit(1);
  }

  while ((read = getline(&line, &len, fp)) != -1) {
    std::string s(line);
    s = s.substr(0, s.length() - 1);
    int n = s.find(":");
    if (n == std::string::npos) {
      fprintf(stderr, "Yaml File %s was not correctly formatted.\n",
              filepath.c_str());
      exit(1);
    }
    variables.emplace(s.substr(0, n), s.substr(n + 1, s.length()));
  }

  fclose(fp);
  if (line) {
    free(line);
  }
}

int YamlLoader::load_int(std::string variable) {
  return std::stoi(variables.at(variable));
}
double YamlLoader::load_double(std::string variable) {
  return std::stod(variables.at(variable));
}