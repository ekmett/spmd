#include <immintrin.h>

namespace spmd {
  namespace avx2 { 
    namespace detail {

      extern __m256 log256_ps(__m256 x);
      extern __m256 exp256_ps(__m256 x);
      extern __m256 sin256_ps(__m256 x);
      extern __m256 cos256_ps(__m256 x);
      extern void sincos256_ps(__m256 x, __m256 *s, __m256 *c);

    } // namespace math
  } // namespace avx2
} // namespace spmd
