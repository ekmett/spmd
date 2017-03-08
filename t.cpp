#include "spmd.h"

using namespace spmd;

int main(int argc, char ** argv) { 
  dispatch([&](auto kernel) {
     typename decltype(kernel)::template varying<int> v;
     return 1;
  });
  return 0;
}
