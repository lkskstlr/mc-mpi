#ifndef YAML_LOADER_HPP
#define YAML_LOADER_HPP

#include <map>
#include <string>

class YamlLoader {
public:
  YamlLoader(std::string filepath);

  int load_int(std::string variable);
  double load_double(std::string variable);

private:
  std::map<std::string, std::string> variables;
};
#endif