#include <stdio.h>
#include <string>

int main(int argc, char const *argv[]) {
  double a_dec = std::stod("10.0");
  double a_sci = std::stod(" 7.071067690849304199e-10");
  double a_int = std::stod("-123413");

  printf("a_dec = %f\n", a_dec);
  printf("a_sci = %e\n", a_sci);
  printf("a_int = %f\n", a_int);
  return 0;
}