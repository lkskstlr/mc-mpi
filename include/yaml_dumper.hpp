#ifndef YAML_DUMPER_HPP
#define YAML_DUMPER_HPP

#include <stdlib.h>
#include <string>

class YamlDumper {
public:
  YamlDumper(std::string filepath);
  ~YamlDumper();

  void dump_int(std::string name, int value);
  void dump_unsigned_long(std::string name, unsigned long value);
  void dump_double(std::string name, double value);
  void dump_string(std::string name, std::string value);

  void new_line();
  void comment(std::string comment);

private:
  FILE *file;
};
#endif