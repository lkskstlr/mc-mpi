#ifndef YAML_DUMPER_HPP
#define YAML_DUMPER_HPP

#include <stdlib.h>
#include <string>

class YamlDumper {
public:
  YamlDumper(std::string filepath);
  ~YamlDumper();

  void dump_int(std::string name, int value);
  void dump_double(std::string name, double value);

private:
  FILE *file;
};
#endif