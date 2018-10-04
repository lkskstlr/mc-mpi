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
void YamlDumper::dump_unsigned_long(std::string name, unsigned long value) {
  fprintf(file, "%s: %zu\n", name.c_str(), value);
}
void YamlDumper::dump_double(std::string name, double value) {
  fprintf(file, "%s: %.18e\n", name.c_str(), value);
}

void YamlDumper::dump_string(std::string name, std::string value) {
  fprintf(file, "%s: %s\n", name.c_str(), value.c_str());
}

void YamlDumper::new_line() { fprintf(file, "\n"); }

void YamlDumper::comment(std::string comment) {
  fprintf(file, "# %s\n", comment.c_str());
}