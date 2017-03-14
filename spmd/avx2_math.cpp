#include <immintrin.h>

#include "math.h"

namespace spmd {
  namespace avx2 { 
    namespace detail {

/* yes I know, the top of this file is quite ugly */
#ifdef _MSC_VER
# define SPMD_ALIGN32_BEG __declspec(align(32))
# define SPMD_ALIGN32_END
#else
# define SPMD_ALIGN32_BEG
# define SPMD_ALIGN32_END __attribute__((aligned(32)))
#endif

#define SPMD_PI32AVX_CONST(Name, Val) \
  static const SPMD_ALIGN32_BEG int pi32avx_##Name[4] SPMD_ALIGN32_END = { Val, Val, Val, Val }

SPMD_PI32AVX_CONST(1, 1);
SPMD_PI32AVX_CONST(inv1, ~1);
SPMD_PI32AVX_CONST(2, 2);
SPMD_PI32AVX_CONST(4, 4);

/* declare some AVX constants -- why can't I figure a better way to do that? */
#define SPMD_PS256_CONST(Name, Val) \
  static const SPMD_ALIGN32_BEG float ps256_##Name[8] SPMD_ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }
#define SPMD_PI32_CONST256(Name, Val) \
  static const SPMD_ALIGN32_BEG int pi32_256_##Name[8] SPMD_ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }
#define SPMD_PS256_CONST_TYPE(Name, Type, Val) \
  static const SPMD_ALIGN32_BEG Type ps256_##Name[8] SPMD_ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }

SPMD_PS256_CONST(1  , 1.0f);
SPMD_PS256_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
SPMD_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
SPMD_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
SPMD_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

SPMD_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);
SPMD_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

SPMD_PI32_CONST256(0, 0);
SPMD_PI32_CONST256(1, 1);
SPMD_PI32_CONST256(inv1, ~1);
SPMD_PI32_CONST256(2, 2);
SPMD_PI32_CONST256(4, 4);
SPMD_PI32_CONST256(0x7f, 0x7f);

SPMD_PS256_CONST(cephes_SQRTHF, 0.707106781186547524);
SPMD_PS256_CONST(cephes_log_p0, 7.0376836292E-2);
SPMD_PS256_CONST(cephes_log_p1, - 1.1514610310E-1);
SPMD_PS256_CONST(cephes_log_p2, 1.1676998740E-1);
SPMD_PS256_CONST(cephes_log_p3, - 1.2420140846E-1);
SPMD_PS256_CONST(cephes_log_p4, + 1.4249322787E-1);
SPMD_PS256_CONST(cephes_log_p5, - 1.6668057665E-1);
SPMD_PS256_CONST(cephes_log_p6, + 2.0000714765E-1);
SPMD_PS256_CONST(cephes_log_p7, - 2.4999993993E-1);
SPMD_PS256_CONST(cephes_log_p8, + 3.3333331174E-1);
SPMD_PS256_CONST(cephes_log_q1, -2.12194440e-4);
SPMD_PS256_CONST(cephes_log_q2, 0.693359375);

/* natural logarithm computed for 8 simultaneous float 
   return NaN for x <= 0
*/ 
__attribute__((target("avx2,fma"))) __m256 log256_ps(__m256 x) {
  __m256i imm0;
  __m256 one = *(__m256*)ps256_1;

  //__m256 invalid_mask = _mm256_cmple_ps(x, _mm256_setzero_ps());
  __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

  x = _mm256_max_ps(x, *(__m256*)ps256_min_norm_pos);  /* cut off denormalized stuff */

  // can be done with AVX2
  imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

  /* keep only the fractional part */
  x = _mm256_and_ps(x, *(__m256*)ps256_inv_mant_mask);
  x = _mm256_or_ps(x, *(__m256*)ps256_0p5);

  // this is again another AVX2 instruction
  imm0 = _mm256_sub_epi32(imm0, *(__m256i*)pi32_256_0x7f);
  __m256 e = _mm256_cvtepi32_ps(imm0);

  e = _mm256_add_ps(e, one);

  /* part2: 
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  //__m256 mask = _mm256_cmplt_ps(x, *(__m256*)ps256_cephes_SQRTHF);
  __m256 mask = _mm256_cmp_ps(x, *(__m256*)ps256_cephes_SQRTHF, _CMP_LT_OS);
  __m256 tmp = _mm256_and_ps(x, mask);
  x = _mm256_sub_ps(x, one);
  e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
  x = _mm256_add_ps(x, tmp);

  __m256 z = _mm256_mul_ps(x,x);

  __m256 y = *(__m256*)ps256_cephes_log_p0;
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_log_p1);
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_log_p2);
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_log_p3);
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_log_p4);
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_log_p5);
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_log_p6);
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_log_p7);
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_log_p8);
  y = _mm256_mul_ps(y, x);

  y = _mm256_mul_ps(y, z);
  
  tmp = _mm256_mul_ps(e, *(__m256*)ps256_cephes_log_q1);
  y = _mm256_add_ps(y, tmp);


  tmp = _mm256_mul_ps(z, *(__m256*)ps256_0p5);
  y = _mm256_sub_ps(y, tmp);

  tmp = _mm256_mul_ps(e, *(__m256*)ps256_cephes_log_q2);
  x = _mm256_add_ps(x, y);
  x = _mm256_add_ps(x, tmp);
  x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
  return x;
}

SPMD_PS256_CONST(exp_hi,	88.3762626647949f);
SPMD_PS256_CONST(exp_lo,	-88.3762626647949f);

SPMD_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
SPMD_PS256_CONST(cephes_exp_C1, 0.693359375);
SPMD_PS256_CONST(cephes_exp_C2, -2.12194440e-4);

SPMD_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
SPMD_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
SPMD_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
SPMD_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
SPMD_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
SPMD_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

__attribute__((target("avx2,fma"))) __m256 exp256_ps(__m256 x) {
  __m256 tmp = _mm256_setzero_ps(), fx;
  __m256i imm0;
  __m256 one = *(__m256*)ps256_1;

  x = _mm256_min_ps(x, *(__m256*)ps256_exp_hi);
  x = _mm256_max_ps(x, *(__m256*)ps256_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_fmadd_ps(x, *(__m256*)ps256_cephes_LOG2EF, *(__m256*)ps256_0p5);

  /* how to perform a floorf with SSE: just below */
  //imm0 = _mm256_cvttps_epi32(fx);
  //tmp  = _mm256_cvtepi32_ps(imm0);
  
  tmp = _mm256_floor_ps(fx);

  /* if greater, substract 1 */
  //__m256 mask = _mm256_cmpgt_ps(tmp, fx);    
  __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);    
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);

  tmp = _mm256_mul_ps(fx, *(__m256*)ps256_cephes_exp_C1);
  __m256 z = _mm256_mul_ps(fx, *(__m256*)ps256_cephes_exp_C2);
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);

  z = _mm256_mul_ps(x,x);
  
  __m256 y = *(__m256*)ps256_cephes_exp_p0;
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_exp_p1);
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_exp_p2);
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_exp_p3);
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_exp_p4);
  y = _mm256_fmadd_ps(y, x, *(__m256*)ps256_cephes_exp_p5);
  y = _mm256_fmadd_ps(y, z, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  // another two AVX2 instructions
  imm0 = _mm256_add_epi32(imm0, *(__m256i*)pi32_256_0x7f);
  imm0 = _mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}

SPMD_PS256_CONST(minus_cephes_DP1, -0.78515625);
SPMD_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
SPMD_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
SPMD_PS256_CONST(sincof_p0, -1.9515295891E-4);
SPMD_PS256_CONST(sincof_p1,  8.3321608736E-3);
SPMD_PS256_CONST(sincof_p2, -1.6666654611E-1);
SPMD_PS256_CONST(coscof_p0,  2.443315711809948E-005);
SPMD_PS256_CONST(coscof_p1, -1.388731625493765E-003);
SPMD_PS256_CONST(coscof_p2,  4.166664568298827E-002);
SPMD_PS256_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI

/* evaluation of 8 sines at onces using AVX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
__attribute__((target(("avx2,fma")))) __m256 sin256_ps(__m256 x) { // any x
  __m256 xmm1, xmm2 = _mm256_setzero_ps(), xmm3, sign_bit, y;
  __m256i imm0, imm2;

  sign_bit = x;
  /* take the absolute value */
  x = _mm256_and_ps(x, *(__m256*)ps256_inv_sign_mask);
  /* extract the sign bit (upper one) */
  sign_bit = _mm256_and_ps(sign_bit, *(__m256*)ps256_sign_mask);
  
  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, *(__m256*)ps256_cephes_FOPI);

  /*
    Here we start a series of integer operations, which are in the
    realm of AVX2.
    If we don't have AVX, let's perform them using SSE2 directives
  */

  /* store the integer part of y in mm0 */
  imm2 = _mm256_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  // another two AVX2 instruction
  imm2 = _mm256_add_epi32(imm2, *(__m256i*)pi32_256_1);
  imm2 = _mm256_and_si256(imm2, *(__m256i*)pi32_256_inv1);
  y = _mm256_cvtepi32_ps(imm2);

  /* get the swap sign flag */
  imm0 = _mm256_and_si256(imm2, *(__m256i*)pi32_256_4);
  imm0 = _mm256_slli_epi32(imm0, 29);
  /* get the polynom selection mask 
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
  imm2 = _mm256_and_si256(imm2, *(__m256i*)pi32_256_2);
  imm2 = _mm256_cmpeq_epi32(imm2,*(__m256i*)pi32_256_0);
 
  __m256 swap_sign_bit = _mm256_castsi256_ps(imm0);
  __m256 poly_mask = _mm256_castsi256_ps(imm2);
  sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

  /* The magic pass: "Extended precision modular arithmetic" 
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  x = _mm256_fmadd_ps(y, *(__m256*)ps256_minus_cephes_DP1, x);
  x = _mm256_fmadd_ps(y, *(__m256*)ps256_minus_cephes_DP2, x);
  x = _mm256_fmadd_ps(y, *(__m256*)ps256_minus_cephes_DP3, x);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(__m256*)ps256_coscof_p0;
  __m256 z = _mm256_mul_ps(x,x);

  y = _mm256_fmadd_ps(y, z, *(__m256*)ps256_coscof_p1);
  y = _mm256_fmadd_ps(y, z, *(__m256*)ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  __m256 tmp = _mm256_mul_ps(z, *(__m256*)ps256_0p5);
  y = _mm256_sub_ps(y, tmp);
  y = _mm256_add_ps(y, *(__m256*)ps256_1);
  
  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  __m256 y2 = *(__m256*)ps256_sincof_p0;
  y2 = _mm256_fmadd_ps(y2, z, *(__m256*)ps256_sincof_p1);
  y2 = _mm256_fmadd_ps(y2, z, *(__m256*)ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_fmadd_ps(y2, x, x);

  /* select the correct result from the two polynoms */  
  xmm3 = poly_mask;
  y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
  y = _mm256_andnot_ps(xmm3, y);
  y = _mm256_add_ps(y,y2);
  /* update the sign */
  y = _mm256_xor_ps(y, sign_bit);

  return y;
}

/* almost the same as sin_ps */
__attribute__((target(("avx2,fma")))) __m256 cos256_ps(__m256 x) { // any x
  __m256 xmm1, xmm2 = _mm256_setzero_ps(), xmm3, y;
  __m256i imm0, imm2;

  /* take the absolute value */
  x = _mm256_and_ps(x, *(__m256*)ps256_inv_sign_mask);
  
  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, *(__m256*)ps256_cephes_FOPI);
  
  /* store the integer part of y in mm0 */
  imm2 = _mm256_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = _mm256_add_epi32(imm2, *(__m256i*)pi32_256_1);
  imm2 = _mm256_and_si256(imm2, *(__m256i*)pi32_256_inv1);
  y = _mm256_cvtepi32_ps(imm2);
  imm2 = _mm256_sub_epi32(imm2, *(__m256i*)pi32_256_2);
  
  /* get the swap sign flag */
  imm0 = _mm256_andnot_si256(imm2, *(__m256i*)pi32_256_4);
  imm0 = _mm256_slli_epi32(imm0, 29);
  /* get the polynom selection mask */
  imm2 = _mm256_and_si256(imm2, *(__m256i*)pi32_256_2);
  imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i*)pi32_256_0);

  __m256 sign_bit = _mm256_castsi256_ps(imm0);
  __m256 poly_mask = _mm256_castsi256_ps(imm2);

  /* The magic pass: "Extended precision modular arithmetic" 
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  x = _mm256_fmadd_ps(y, *(__m256*)ps256_minus_cephes_DP1, x);
  x = _mm256_fmadd_ps(y, *(__m256*)ps256_minus_cephes_DP2, x);
  x = _mm256_fmadd_ps(y, *(__m256*)ps256_minus_cephes_DP3, x);
  
  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(__m256*)ps256_coscof_p0;
  __m256 z = _mm256_mul_ps(x,x);

  y = _mm256_fmadd_ps(y, z, *(__m256*)ps256_coscof_p1);
  y = _mm256_fmadd_ps(y, z, *(__m256*)ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  __m256 tmp = _mm256_mul_ps(z, *(__m256*)ps256_0p5);
  y = _mm256_sub_ps(y, tmp);
  y = _mm256_add_ps(y, *(__m256*)ps256_1);
  
  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  __m256 y2 = *(__m256*)ps256_sincof_p0;
  y2 = _mm256_fmadd_ps(y2, z, *(__m256*)ps256_sincof_p1);
  y2 = _mm256_fmadd_ps(y2, z, *(__m256*)ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_fmadd_ps(y2, x, x);

  /* select the correct result from the two polynoms */  
  xmm3 = poly_mask;
  y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
  y = _mm256_andnot_ps(xmm3, y);
  y = _mm256_add_ps(y,y2);
  /* update the sign */
  y = _mm256_xor_ps(y, sign_bit);

  return y;
}

/* since sin256_ps and cos256_ps are almost identical, sincos256_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
__attribute__((target(("avx2,fma")))) void sincos256_ps(__m256 x, __m256 *s, __m256 *c) {

  __m256 xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
  __m256i imm0, imm2, imm4;

  sign_bit_sin = x;
  /* take the absolute value */
  x = _mm256_and_ps(x, *(__m256*)ps256_inv_sign_mask);
  /* extract the sign bit (upper one) */
  sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(__m256*)ps256_sign_mask);
  
  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, *(__m256*)ps256_cephes_FOPI);

  /* store the integer part of y in imm2 */
  imm2 = _mm256_cvttps_epi32(y);

  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = _mm256_add_epi32(imm2, *(__m256i*)pi32_256_1);
  imm2 = _mm256_and_si256(imm2, *(__m256i*)pi32_256_inv1);

  y = _mm256_cvtepi32_ps(imm2);
  imm4 = imm2;

  /* get the swap sign flag for the sine */
  imm0 = _mm256_and_si256(imm2, *(__m256i*)pi32_256_4);
  imm0 = _mm256_slli_epi32(imm0, 29);
  //__m256 swap_sign_bit_sin = _mm256_castsi256_ps(imm0);

  /* get the polynom selection mask for the sine*/
  imm2 = _mm256_and_si256(imm2, *(__m256i*)pi32_256_2);
  imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i*)pi32_256_0);
  //__m256 poly_mask = _mm256_castsi256_ps(imm2);

  __m256 swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
  __m256 poly_mask = _mm256_castsi256_ps(imm2);

  /* The magic pass: "Extended precision modular arithmetic" 
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  x = _mm256_fmadd_ps(y, *(__m256*)ps256_minus_cephes_DP1, x);
  x = _mm256_fmadd_ps(y, *(__m256*)ps256_minus_cephes_DP2, x);
  x = _mm256_fmadd_ps(y, *(__m256*)ps256_minus_cephes_DP3, x);

  imm4 = _mm256_sub_epi32(imm4, *(__m256i*)pi32_256_2);
  imm4 = _mm256_andnot_si256(imm4, *(__m256i*)pi32_256_4);
  imm4 = _mm256_slli_epi32(imm4, 29);

  __m256 sign_bit_cos = _mm256_castsi256_ps(imm4);

  sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);
  
  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  __m256 z = _mm256_mul_ps(x,x);
  y = *(__m256*)ps256_coscof_p0;

  y = _mm256_fmadd_ps(y, z, *(__m256*)ps256_coscof_p1);
  y = _mm256_fmadd_ps(y, z, *(__m256*)ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  __m256 tmp = _mm256_mul_ps(z, *(__m256*)ps256_0p5);
  y = _mm256_sub_ps(y, tmp);
  y = _mm256_add_ps(y, *(__m256*)ps256_1);
  
  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  __m256 y2 = *(__m256*)ps256_sincof_p0;

  y2 = _mm256_fmadd_ps(y2, z, *(__m256*)ps256_sincof_p1);
  y2 = _mm256_fmadd_ps(y2, z, *(__m256*)ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_fmadd_ps(y2, x, x);

  /* select the correct result from the two polynoms */  
  xmm3 = poly_mask;
  __m256 ysin2 = _mm256_and_ps(xmm3, y2);
  __m256 ysin1 = _mm256_andnot_ps(xmm3, y);
  y2 = _mm256_sub_ps(y2,ysin2);
  y = _mm256_sub_ps(y, ysin1);

  xmm1 = _mm256_add_ps(ysin1,ysin2);
  xmm2 = _mm256_add_ps(y,y2);
 
  /* update the sign */
  *s = _mm256_xor_ps(xmm1, sign_bit_sin);
  *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

    } // namespace math
  } // namespace avx2
} // namespace spmd

/* 
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

// MODIFICATIONS FOR "SPMD" PROJECT
// 1. Remove v8sf/v8si/v4si typedefs (don't want to expose these in the interface)
// 2. Define ALIGN32_BEG/ALIGN32_END properly for MSVC
// 3. Explicitly cast 0x80000000 to int for sign_mask to silence warning.
// 4. Replace _mm256_and_si128 and _mm256_andnot_si128 with _mm256_and_si256 and _mm256_andnot_si256, respectively.
// 5. switch #warning to #pragma message for MSVC
// 6. rename functions from "_mm256*" to "mm256*", since MSVC doesn't want you to redefine intrinsics
// 7. Apply ALIGN32_BEG/ALIGN32_END to imm_xmm_union

// Modifications for yet another "SPMD" project by Edward Kmett
// 1. removed pre-avx2 support
// 2. converted to fit surrounding namespacing discipline mostly by prefixing defines and avoiding dangerous _ prefixes.
// 3. incorporated FMA3 since my oldest platform is Haswell
