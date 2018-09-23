#include <cstdint>
#include <tuple>
#include <vector>
#include <string>
#include <iostream>

typedef double Float;
typedef std::size_t Id_t;
typedef std::tuple<Id_t, Float> Edge;

struct Node {
    Id_t n;
    bool taken;
};

class Graph {
  public:
    Graph(Id_t n, Float density);
    Graph(Id_t n, std::string const& type);

    Id_t add_node(Float value);
    void add_edge(Float value, Id_t n1, Id_t n2);
    friend std::ostream &operator<<(std::ostream &os,
                                    const Graph &graph);
    void write_d3();
    void write_d3(std::string const& filename);

  public:
    void generate_partners();
    std::vector<Float> values;
    std::vector<std::vector<Edge>> edges;
};