#ifndef AMD_RPP_RPP_CPU_SIMD_HPP
#define AMD_RPP_RPP_CPU_SIMD_HPP
#if 1
#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif

#define __AVX2__ 1

#define M256I(m256i_register) (*((_m256i_union*)&m256i_register))
typedef union {
    char               m256i_i8[32];
    short              m256i_i16[16];
    int                m256i_i32[8];
    long long          m256i_i64[4];
    __m128i            m256i_i128[2];
}_m256i_union;

#if defined(_MSC_VER)
#define SIMD_ALIGN_VAR(type, name, alignment) \
  __declspec(align(alignment)) type name
#else
#define SIMD_ALIGN_VAR(type, name, alignment) \
  type __attribute__((__aligned__(alignment))) name
#endif // _MSC_VER

#define SIMD_CONST_PI(name, val0, val1, val2, val3) \
  SIMD_ALIGN_VAR(static const int, _xmm_const_##name[4], 16) = { \
    static_cast<int>(val3), \
    static_cast<int>(val2), \
    static_cast<int>(val1), \
    static_cast<int>(val0)  \
  }

#define SIMD_CONST_PS(name, val0, val1, val2, val3) \
  SIMD_ALIGN_VAR(static const float, _xmm_const_##name[4], 16) = { \
    static_cast<float>(val3), \
    static_cast<float>(val2), \
    static_cast<float>(val1), \
    static_cast<float>(val0)  \
  }

#define SIMD_GET_PS(name) (*(const __m128  *)_xmm_const_##name)

// Shuffle floats in `src` by using SSE2 `pshufd` instead of `shufps`, if possible.
#define SIMD_SHUFFLE_PS(src, imm) \
  _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(src), imm))

SIMD_CONST_PI(full       , 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
SIMD_CONST_PI(sn         , 0x80000000, 0x80000000, 0x80000000, 0x80000000);
SIMD_CONST_PS(m6_m6_m6_m6,-6.0f ,-6.0f ,-6.0f ,-6.0f);
SIMD_CONST_PS(m4o6_m4o6_m4o6_m4o6,-4.0f / 6.0f,-4.0f / 6.0f,-4.0f / 6.0f,-4.0f / 6.0f);
SIMD_CONST_PS(eps        , 1e-9f, 1e-9f, 1e-9f, 1e-9f);
SIMD_CONST_PS(p1         , 1.0f , 1.0f , 1.0f , 1.0f);

SIMD_CONST_PS(p4o6_p2o6_p3o6_p0  , 4.0f / 6.0f, 2.0f / 6.0f, 3.0f / 6.0f, 0.0f);
SIMD_CONST_PI(abs        , 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);
SIMD_CONST_PS(m6_m6_p6_p0,-6.0f ,-6.0f , 6.0f , 0.0f);
SIMD_CONST_PS(p1_p1_m2_p0, 1.0f , 1.0f ,-2.0f , 0.0f);
SIMD_CONST_PS(m1_m1_m1_p1,-1.0f ,-1.0f ,-1.0f , 1.0f);
SIMD_CONST_PS(p0         , 0.0f , 0.0f , 0.0f , 0.0f);

inline __m128i _mm_mullo_epi8(__m128i a, __m128i b)
{
    __m128i zero = _mm_setzero_si128();
    __m128i Alo = _mm_cvtepu8_epi16(a);
    __m128i Ahi = _mm_unpackhi_epi8(a, zero);
    __m128i Blo = _mm_cvtepu8_epi16(b);
    __m128i Bhi = _mm_unpackhi_epi8(b, zero);
    __m128i Clo = _mm_mullo_epi16(Alo, Blo);
    __m128i Chi = _mm_mullo_epi16(Ahi, Bhi);
    __m128i maskLo = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i maskHi = _mm_set_epi8(14, 12, 10, 8, 6, 4, 2, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
    __m128i C = _mm_or_si128(_mm_shuffle_epi8(Clo, maskLo), _mm_shuffle_epi8(Chi, maskHi));

     return C;
}

inline void _mm_print_epi8(__m128i vPrintArray)
{
  char printArray[16];
  _mm_storeu_si128((__m128i *)printArray, vPrintArray);
  for (int ct = 0; ct < 16; ct++)
  {
      printf("%d ", printArray[ct]);
  }
}

inline void _mm_print_epi32(__m128i vPrintArray)
{
  int printArray[4];
  _mm_storeu_si128((__m128i *)printArray, vPrintArray);
  for (int ct = 0; ct < 4; ct++)
  {
      printf("%d ", printArray[ct]);
  }
}

inline void _mm_print_ps(__m128 vPrintArray)
{
  float printArray[4];
  _mm_storeu_ps(printArray, vPrintArray);
  for (int ct = 0; ct < 4; ct++)
  {
      printf("%0.6f ", printArray[ct]);
  }
}

#define CHECK_SIMD  0
#define FP_BITS     16
#define FP_MUL      (1<<FP_BITS)


static inline Rpp32u HorMin(__m128i pmin)
{
    pmin = _mm_min_epu8(pmin, _mm_shuffle_epi32(pmin, _MM_SHUFFLE(3, 2, 3, 2)));
    pmin = _mm_min_epu8(pmin, _mm_shuffle_epi32(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin = _mm_min_epu8(pmin, _mm_shufflelo_epi16(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin = _mm_min_epu8(pmin, _mm_srli_epi16(pmin, 8));
    return (_mm_cvtsi128_si32(pmin) & 0x000000FF);    
}

static inline Rpp32u HorMax(__m128i pmax)
{
    pmax = _mm_min_epu8(pmax, _mm_shuffle_epi32(pmax, _MM_SHUFFLE(3, 2, 3, 2)));
    pmax = _mm_min_epu8(pmax, _mm_shuffle_epi32(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax = _mm_min_epu8(pmax, _mm_shufflelo_epi16(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax = _mm_min_epu8(pmax, _mm_srli_epi16(pmax, 8));
    return (_mm_cvtsi128_si32(pmax) & 0x000000FF);
}

#if __AVX__
static inline Rpp32u HorMin256(__m256i pmin)
{
    __m128i pmin_128;
    pmin = _mm256_min_epu8(pmin, _mm256_permute4x64_epi64(pmin, _MM_SHUFFLE(3, 2, 3, 2)));
    pmin = _mm256_min_epu8(pmin, _mm256_permute4x64_epi64(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin_128 = M256I(pmin).m256i_i128[0];
    pmin_128 = _mm_min_epu8(pmin_128, _mm_shufflelo_epi16(pmin_128, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin_128 = _mm_min_epu8(pmin_128, _mm_srli_epi16(pmin_128, 8));
    return (_mm_cvtsi128_si32(pmin_128) & 0x000000FF);
}

static inline Rpp32u HorMax256(__m256i pmax)
{
    __m128i pmax_128;
    pmax = _mm256_max_epu8(pmax, _mm256_permute4x64_epi64(pmax, _MM_SHUFFLE(3, 2, 3, 2)));
    pmax = _mm256_max_epu8(pmax, _mm256_permute4x64_epi64(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax_128 = M256I(pmax).m256i_i128[0];
    pmax_128 = _mm_max_epi8(pmax_128, _mm_shufflelo_epi16(pmax_128, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax_128 = _mm_max_epi8(pmax_128, _mm_srli_epi16(pmax_128, 8));
    return (_mm_cvtsi128_si32(pmax_128) & 0x000000FF);    
}
#endif

#define __SSE4_1__ 1

static  inline __m128 fast_exp_sse (__m128 x)
{
    __m128 t, f, e, p, r;
    __m128i i, j;
    __m128 l2e = _mm_set1_ps (1.442695041f);  /* log2(e) */
    __m128 c0  = _mm_set1_ps (0.3371894346f);
    __m128 c1  = _mm_set1_ps (0.657636276f);
    __m128 c2  = _mm_set1_ps (1.00172476f);

    /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */   
    t = _mm_mul_ps (x, l2e);             /* t = log2(e) * x */
#ifdef __SSE4_1__
    e = _mm_floor_ps (t);                /* floor(t) */
    i = _mm_cvtps_epi32 (e);             /* (int)floor(t) */
#else /* __SSE4_1__*/
    i = _mm_cvttps_epi32 (t);            /* i = (int)t */
    j = _mm_srli_epi32 (_mm_castps_si128 (x), 31); /* signbit(t) */
    i = _mm_sub_epi32 (i, j);            /* (int)t - signbit(t) */
    e = _mm_cvtepi32_ps (i);             /* floor(t) ~= (int)t - signbit(t) */
#endif /* __SSE4_1__*/
    f = _mm_sub_ps (t, e);               /* f = t - floor(t) */
    p = c0;                              /* c0 */
    p = _mm_mul_ps (p, f);               /* c0 * f */
    p = _mm_add_ps (p, c1);              /* c0 * f + c1 */
    p = _mm_mul_ps (p, f);               /* (c0 * f + c1) * f */
    p = _mm_add_ps (p, c2);              /* p = (c0 * f + c1) * f + c2 ~= 2^f */
    j = _mm_slli_epi32 (i, 23);          /* i << 23 */
    r = _mm_castsi128_ps (_mm_add_epi32 (j, _mm_castps_si128 (p))); /* r = p * 2^i*/
    return r;
}

#define set1_ps_hex(x) _mm_castsi128_ps(_mm_set1_epi32(x))

static const __m128 _ps_0 = _mm_set1_ps(0.f);
static const __m128 _ps_1 = _mm_set1_ps(1.f);
static const __m128 _ps_0p5 = _mm_set1_ps(0.5f);
static const __m128 _ps_min_norm_pos = set1_ps_hex(0x00800000);
static const __m128 _ps_mant_mask = set1_ps_hex(0x7f800000);
static const __m128 _ps_inv_mant_mask = set1_ps_hex(~0x7f800000);
static const __m128 _ps_sign_mask = set1_ps_hex(0x80000000);
static const __m128 _ps_inv_sign_mask = set1_ps_hex(~0x80000000);

static const __m128i _pi32_1 = _mm_set1_epi32(1);
static const __m128i _pi32_inv1 = _mm_set1_epi32(~1);
static const __m128i _pi32_2 = _mm_set1_epi32(2);
static const __m128i _pi32_4 = _mm_set1_epi32(4);
static const __m128i _pi32_0x7f = _mm_set1_epi32(0x7f);

static const __m128 _ps_minus_cephes_DP1 = _mm_set1_ps(-0.78515625f);
static const __m128 _ps_minus_cephes_DP2 = _mm_set1_ps(-2.4187564849853515625e-4f);
static const __m128 _ps_minus_cephes_DP3 = _mm_set1_ps(-3.77489497744594108e-8f);
static const __m128 _ps_sincof_p0 = _mm_set1_ps(-1.9515295891E-4f);
static const __m128 _ps_sincof_p1 = _mm_set1_ps( 8.3321608736E-3f);
static const __m128 _ps_sincof_p2 = _mm_set1_ps(-1.6666654611E-1f);
static const __m128 _ps_coscof_p0 = _mm_set1_ps( 2.443315711809948E-005f);
static const __m128 _ps_coscof_p1 = _mm_set1_ps(-1.388731625493765E-003f);
static const __m128 _ps_coscof_p2 = _mm_set1_ps( 4.166664568298827E-002f);
static const __m128 _ps_cephes_FOPI = _mm_set1_ps(1.27323954473516f); // 4 / M_PI

static inline void sincos_ps(__m128 x, __m128 *s, __m128 *c) {

#if 0
#ifdef MATH_SSE41 // _mm_round_ps is SSE4.1
  // XXX Added in MathGeoLib: Take a modulo of the input in 2pi to try to enhance the precision with large input values.
  x = modf_ps(x, _mm_set1_ps(2.f*3.141592654f));
#endif
#endif

  /* extract the sign bit (upper one) */
  __m128 sign_bit_sin = _mm_and_ps(x, _ps_sign_mask);
  /* take the absolute value */
  x = _mm_xor_ps(x, sign_bit_sin);
  
  /* scale by 4/Pi */
  __m128 y = _mm_mul_ps(x, _ps_cephes_FOPI);
    
  /* store the integer part of y in emm2 */
  __m128i emm2 = _mm_cvttps_epi32(y);

  /* j=(j+1) & (~1) (see the cephes sources) */
  emm2 = _mm_add_epi32(emm2, _pi32_1);
  emm2 = _mm_and_si128(emm2, _pi32_inv1);
  y = _mm_cvtepi32_ps(emm2);

  __m128i emm4 = emm2;

  /* get the swap sign flag for the sine */
  __m128i emm0 = _mm_and_si128(emm2, _pi32_4);
  emm0 = _mm_slli_epi32(emm0, 29);
  __m128 swap_sign_bit_sin = _mm_castsi128_ps(emm0);

  /* get the polynom selection mask for the sine*/
  emm2 = _mm_and_si128(emm2, _pi32_2);
  emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
  __m128 poly_mask = _mm_castsi128_ps(emm2);
  /* The magic pass: "Extended precision modular arithmetic" 
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  __m128 xmm1 = _mm_mul_ps(y, _ps_minus_cephes_DP1);
  __m128 xmm2 = _mm_mul_ps(y, _ps_minus_cephes_DP2);
  __m128 xmm3 = _mm_mul_ps(y, _ps_minus_cephes_DP3);
  x = _mm_add_ps(_mm_add_ps(x, xmm1), _mm_add_ps(xmm2, xmm3));

  emm4 = _mm_sub_epi32(emm4, _pi32_2);
  emm4 = _mm_andnot_si128(emm4, _pi32_4);
  emm4 = _mm_slli_epi32(emm4, 29);
  __m128 sign_bit_cos = _mm_castsi128_ps(emm4);

  sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);
  
  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  __m128 z = _mm_mul_ps(x,x);
  y = _ps_coscof_p0;

  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, _ps_coscof_p1);
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, _ps_coscof_p2);
  y = _mm_mul_ps(y, _mm_mul_ps(z, z));
  __m128 tmp = _mm_mul_ps(z, _ps_0p5);
  y = _mm_sub_ps(y, tmp);
  y = _mm_add_ps(y, _ps_1);
  
  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  __m128 y2 = _ps_sincof_p0;
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_add_ps(y2, _ps_sincof_p1);
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_add_ps(y2, _ps_sincof_p2);
  y2 = _mm_mul_ps(y2, _mm_mul_ps(z, x));
  y2 = _mm_add_ps(y2, x);

  /* select the correct result from the two polynoms */  
  xmm3 = poly_mask;
  __m128 ysin2 = _mm_and_ps(xmm3, y2);
  __m128 ysin1 = _mm_andnot_ps(xmm3, y);
  y2 = _mm_sub_ps(y2,ysin2);
  y = _mm_sub_ps(y, ysin1);

  xmm1 = _mm_add_ps(ysin1,ysin2);
  xmm2 = _mm_add_ps(y,y2);
 
  /* update the sign */
  *s = _mm_xor_ps(xmm1, sign_bit_sin);
  *c = _mm_xor_ps(xmm2, sign_bit_cos);
}

static const __m128 _ps_atanrange_hi = _mm_set1_ps(2.414213562373095);
static const __m128 _ps_atanrange_lo = _mm_set1_ps(0.4142135623730950);
static const __m128 _ps_cephes_PIF = _mm_set1_ps(3.141592653589793238);
static const __m128 _ps_cephes_PIO2F = _mm_set1_ps(1.5707963267948966192);
static const __m128 _ps_cephes_PIO4F = _mm_set1_ps(0.7853981633974483096);

static const __m128 _ps_atancof_p0 = _mm_set1_ps(8.05374449538e-2);
static const __m128 _ps_atancof_p1 = _mm_set1_ps(1.38776856032E-1);
static const __m128 _ps_atancof_p2 = _mm_set1_ps(1.99777106478E-1);
static const __m128 _ps_atancof_p3 = _mm_set1_ps(3.33329491539E-1);

static inline __m128 atan_ps( __m128 x )
{
	__m128 sign_bit, y;

	sign_bit = x;
	/* take the absolute value */
	x = _mm_and_ps( x, _ps_inv_sign_mask );
	/* extract the sign bit (upper one) */
	sign_bit = _mm_and_ps( sign_bit, _ps_sign_mask );

/* range reduction, init x and y depending on range */

	/* x > 2.414213562373095 */
	__m128 cmp0 = _mm_cmpgt_ps( x, _ps_atanrange_hi );
	/* x > 0.4142135623730950 */
	__m128 cmp1 = _mm_cmpgt_ps( x, _ps_atanrange_lo );

	/* x > 0.4142135623730950 && !( x > 2.414213562373095 ) */
	__m128 cmp2 = _mm_andnot_ps( cmp0, cmp1 );

	/* -( 1.0/x ) */
	__m128 y0 = _mm_and_ps( cmp0, _ps_cephes_PIO2F );
	__m128 x0 = _mm_div_ps( _ps_1, x );
	x0 = _mm_xor_ps( x0, _ps_sign_mask );

	__m128 y1 = _mm_and_ps( cmp2, _ps_cephes_PIO4F );
	/* (x-1.0)/(x+1.0) */
	__m128 x1_o = _mm_sub_ps( x, _ps_1 );
	__m128 x1_u = _mm_add_ps( x, _ps_1 );
	__m128 x1 = _mm_div_ps( x1_o, x1_u );

	__m128 x2 = _mm_and_ps( cmp2, x1 );
	x0 = _mm_and_ps( cmp0, x0 );
	x2 = _mm_or_ps( x2, x0 );
	cmp1 = _mm_or_ps( cmp0, cmp2 );
	x2 = _mm_and_ps( cmp1, x2 );
	x = _mm_andnot_ps( cmp1, x );
	x = _mm_or_ps( x2, x );

	y = _mm_or_ps( y0, y1 );

	__m128 zz = _mm_mul_ps( x, x );
	__m128 acc = _ps_atancof_p0;
	acc = _mm_mul_ps( acc, zz );
	acc = _mm_sub_ps( acc, _ps_atancof_p1 );
	acc = _mm_mul_ps( acc, zz );
	acc = _mm_add_ps( acc, _ps_atancof_p2 );
	acc = _mm_mul_ps( acc, zz );
	acc = _mm_sub_ps( acc, _ps_atancof_p3 );
	acc = _mm_mul_ps( acc, zz );
	acc = _mm_mul_ps( acc, x );
	acc = _mm_add_ps( acc, x );
	y = _mm_add_ps( y, acc );

	/* update the sign */
	y = _mm_xor_ps( y, sign_bit );

	return y;
}

static inline __m128 atan2_ps( __m128 y, __m128 x )
{
	__m128 x_eq_0 = _mm_cmpeq_ps( x, _ps_0 );
	__m128 x_gt_0 = _mm_cmpgt_ps( x, _ps_0 );
	__m128 x_le_0 = _mm_cmple_ps( x, _ps_0 );
	__m128 y_eq_0 = _mm_cmpeq_ps( y, _ps_0 );
	__m128 x_lt_0 = _mm_cmplt_ps( x, _ps_0 );
	__m128 y_lt_0 = _mm_cmplt_ps( y, _ps_0 );

	__m128 zero_mask = _mm_and_ps( x_eq_0, y_eq_0 );
	__m128 zero_mask_other_case = _mm_and_ps( y_eq_0, x_gt_0 );
	zero_mask = _mm_or_ps( zero_mask, zero_mask_other_case );

	__m128 pio2_mask = _mm_andnot_ps( y_eq_0, x_eq_0 );
	__m128 pio2_mask_sign = _mm_and_ps( y_lt_0, _ps_sign_mask );
	__m128 pio2_result = _ps_cephes_PIO2F;
	pio2_result = _mm_xor_ps( pio2_result, pio2_mask_sign );
	pio2_result = _mm_and_ps( pio2_mask, pio2_result );

	__m128 pi_mask = _mm_and_ps( y_eq_0, x_le_0 );
	__m128 pi = _ps_cephes_PIF;
	__m128 pi_result = _mm_and_ps( pi_mask, pi );

	__m128 swap_sign_mask_offset = _mm_and_ps( x_lt_0, y_lt_0 );
	swap_sign_mask_offset = _mm_and_ps( swap_sign_mask_offset, _ps_sign_mask );

	__m128 offset0 = _mm_setzero_ps();
	__m128 offset1 = _ps_cephes_PIF;
	offset1 = _mm_xor_ps( offset1, swap_sign_mask_offset );

	__m128 offset = _mm_andnot_ps( x_lt_0, offset0 );
	offset = _mm_and_ps( x_lt_0, offset1 );

	__m128 arg = _mm_div_ps( y, x );
	__m128 atan_result = atan_ps( arg );
	atan_result = _mm_add_ps( atan_result, offset );

	/* select between zero_result, pio2_result and atan_result */

	__m128 result = _mm_andnot_ps( zero_mask, pio2_result );
	atan_result = _mm_andnot_ps( pio2_mask, atan_result );
	atan_result = _mm_andnot_ps( pio2_mask, atan_result);
	result = _mm_or_ps( result, atan_result );
	result = _mm_or_ps( result, pi_result );

	return result;
}

#endif
#endif //AMD_RPP_RPP_CPU_SIMD_HPP