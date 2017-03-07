#pragma once
// cpu detection
namespace spmd {
  namespace cpu {
    enum isa : int { 
      error = 0,
      sse2 = 1,
      sse4 = 2,
      avx = 3,
      avx11 = 4,
      avx2 = 5,
      avx512_knl = 6,
      avx512_skx = 7,
      max_intel = avx512_skx,
      amd_neon = 8,
      max_isa = amd_neon
    };
    isa system_isa();
  }
}