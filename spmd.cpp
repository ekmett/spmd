#include "spmd.h"
#include <stdio.h>

namespace spmd {
  thread_local mask execution_mask;
}

using namespace spmd;

int main(int argc, char ** argv) {

  // do vectorized stuff here

}