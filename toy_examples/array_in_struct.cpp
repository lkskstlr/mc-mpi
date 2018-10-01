#include <stdio.h>

typedef struct S_tag {
  float data[5];
  int a;
} S;

int main(int argc, char const *argv[]) {
  printf("Size = %zu should be 24\n", sizeof(S));
  return 0;
}