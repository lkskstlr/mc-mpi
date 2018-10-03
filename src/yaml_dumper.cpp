#include "yaml_dumper.hpp"

YamlDumper::YamlDumper(std::string filepath) {
  file = fopen(filepath.c_str(), "w");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s for writing.\n", filepath.c_str());
    exit(1);
  }
}

YamlDumper::~YamlDumper() { fclose(file); }

void YamlDumper::dump_int(std::string name, int value) {
  fprintf(file, "%s: %d\n", name.c_str(), value);
}
void YamlDumper::dump_double(std::string name, double value) {
  fprintf(file, "%s: %.18e\n", name.c_str(), value);
}