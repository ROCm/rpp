/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_CPU_COMMON_H
#define RPP_CPU_COMMON_H

#include <math.h>
#include <algorithm>
#include <typeinfo>
#include <cstring>
#include <rppdefs.h>
#include <omp.h>
#include <half/half.hpp>
using halfhpp = half_float::half;
typedef halfhpp Rpp16f;
#include "rpp_cpu_simd.hpp"

#define PI                              3.14159265
#define PI_OVER_180                     0.0174532925
#define ONE_OVER_255                    0.00392156862745f
#define ONE_OVER_256                    0.00390625f
#define RPP_128_OVER_255                0.50196078431f
#define RAD(deg)                        (deg * PI / 180)
#define RPPABS(a)                       ((a < 0) ? (-a) : (a))
#define RPPMIN2(a,b)                    ((a < b) ? a : b)
#define RPPMIN3(a,b,c)                  ((a < b) && (a < c) ?  a : ((b < c) ? b : c))
#define RPPMAX2(a,b)                    ((a > b) ? a : b)
#define RPPMAX3(a,b,c)                  ((a > b) && (a > c) ?  a : ((b > c) ? b : c))
#define RPPINRANGE(a, x, y)             ((a >= x) && (a <= y) ? 1 : 0)
#define RPPPRANGECHECK(value, a, b)     (value < (Rpp32f) a) ? ((Rpp32f) a) : ((value < (Rpp32f) b) ? value : ((Rpp32f) b))
#define RPPFLOOR(a)                     ((int) a)
#define RPPCEIL(a)                      ((int) (a + 1.0))
#define RPPISEVEN(a)                    ((a % 2 == 0) ? 1 : 0)
#define RPPPIXELCHECK(pixel)            (pixel < (Rpp32f) 0) ? ((Rpp32f) 0) : ((pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255))
#define RPPPIXELCHECKF32(pixel)         (pixel < (Rpp32f) 0) ? ((Rpp32f) 0) : ((pixel < (Rpp32f) 1) ? pixel : ((Rpp32f) 1))
#define RPPPIXELCHECKI8(pixel)          (pixel < (Rpp32f) -128) ? ((Rpp32f) -128) : ((pixel < (Rpp32f) 127) ? pixel : ((Rpp32f) 127))
#define RPPISGREATER(pixel, value)      ((pixel > value) ? 1 : 0)
#define RPPISLESSER(pixel, value)       ((pixel < value) ? 1 : 0)
#define XORWOW_COUNTER_INC              0x587C5     // Hex 0x587C5 = Dec 362437U - xorwow counter increment
#define XORWOW_EXPONENT_MASK            0x3F800000  // Hex 0x3F800000 = Bin 0b111111100000000000000000000000 - 23 bits of mantissa set to 0, 01111111 for the exponent, 0 for the sign bit
#define RGB_TO_GREY_WEIGHT_RED          0.299f
#define RGB_TO_GREY_WEIGHT_GREEN        0.587f
#define RGB_TO_GREY_WEIGHT_BLUE         0.114f
#define INTERP_BILINEAR_KERNEL_SIZE     2           // Kernel size needed for Bilinear Interpolation
#define INTERP_BILINEAR_KERNEL_RADIUS   1.0f        // Kernel radius needed for Bilinear Interpolation
#define INTERP_BILINEAR_NUM_COEFFS      4           // Number of coefficents needed for Bilinear Interpolation
#define NEWTON_METHOD_INITIAL_GUESS     0x5f3759df          // Initial guess for Newton Raphson Inverse Square Root
#define RPP_2POW32                      0x100000000         // (2^32)
#define RPP_2POW32_INV                  2.3283064e-10f      // (1 / 2^32)
#define RPP_2POW32_INV_DIV_2            1.164153218e-10f    // RPP_2POW32_INV / 2
#define RPP_2POW32_INV_MUL_2PI          1.46291812e-09f     // (1 / 2^32) * 2PI
#define RPP_2POW32_INV_MUL_2PI_DIV_2    7.3145906e-10f      // RPP_2POW32_INV_MUL_2PI / 2
#define RPP_255_OVER_1PT57              162.3380757272f     // (255 / 1.570796) - multiplier used in phase computation
#define ONE_OVER_1PT57                  0.6366199048f       // (1 / 1.570796) i.e. 2/pi - multiplier used in phase computation

const __m128 xmm_p2Pow32 = _mm_set1_ps(RPP_2POW32);
const __m128 xmm_p2Pow32Inv = _mm_set1_ps(RPP_2POW32_INV);
const __m128 xmm_p2Pow32InvDiv2 = _mm_set1_ps(RPP_2POW32_INV_DIV_2);
const __m128 xmm_p2Pow32InvMul2Pi = _mm_set1_ps(RPP_2POW32_INV_MUL_2PI);
const __m128 xmm_p2Pow32InvMul2PiDiv2 = _mm_set1_ps(RPP_2POW32_INV_MUL_2PI_DIV_2);
const __m128i xmm_newtonMethodInitialGuess = _mm_set1_epi32(NEWTON_METHOD_INITIAL_GUESS);

const __m256 avx_p2Pow32 = _mm256_set1_ps(RPP_2POW32);
const __m256 avx_p2Pow32Inv = _mm256_set1_ps(RPP_2POW32_INV);
const __m256 avx_p2Pow32InvDiv2 = _mm256_set1_ps(RPP_2POW32_INV_DIV_2);
const __m256 avx_p2Pow32InvMul2Pi = _mm256_set1_ps(RPP_2POW32_INV_MUL_2PI);
const __m256 avx_p2Pow32InvMul2PiDiv2 = _mm256_set1_ps(RPP_2POW32_INV_MUL_2PI_DIV_2);
const __m256i avx_newtonMethodInitialGuess = _mm256_set1_epi32(NEWTON_METHOD_INITIAL_GUESS);

#if __AVX2__
#define SIMD_FLOAT_VECTOR_LENGTH        8
#else
#define SIMD_FLOAT_VECTOR_LENGTH        4
#endif

/*Constants used for Gaussian interpolation*/
// Here sigma is considered as 0.5f
#define GAUSSCONSTANT1                 -2.0f          // 1 / (sigma * sigma * -1 * 2);
#define GAUSSCONSTANT2                  0.7978845608028654f // 1 / ((2 * PI)*(1/2) * sigma)
static uint16_t wyhash16_x;

alignas(64) const Rpp32f sch_mat[16] = {0.701f, -0.299f, -0.300f, 0.0f, -0.587f, 0.413f, -0.588f, 0.0f, -0.114f, -0.114f, 0.886f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
alignas(64) const Rpp32f ssh_mat[16] = {0.168f, -0.328f, 1.250f, 0.0f, 0.330f, 0.035f, -1.050f, 0.0f, -0.497f, 0.292f, -0.203f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
alignas(64) const Rpp32u multiseedStreamOffset[8] = {0x15E975, 0x2359A3, 0x42CC61, 0x1925A7, 0x123AA3, 0x21F149, 0x2DDE23, 0x2A93BB};    // Prime numbers for multiseed stream initialization

inline uint32_t hash16(uint32_t input, uint32_t key) {
  uint32_t hash = input * key;
  return ((hash >> 16) ^ hash) & 0xFFFF;
}

inline uint16_t wyhash16() {
  wyhash16_x += 0xfc15;
  return hash16(wyhash16_x, 0x2ab);
}

inline uint16_t rand_range16(const uint16_t s) {
    uint16_t x = wyhash16();
    uint32_t m = (uint32_t)x * (uint32_t)s;
    uint16_t l = (uint16_t)m;
    if (l < s) {
        uint16_t t = -s % s;
        while (l < t) {
            x = wyhash16();
            m = (uint32_t)x * (uint32_t)s;
            l = (uint16_t)m;
        }
    }
    return m >> 16;
}

static unsigned int g_seed;

inline void fast_srand( int seed )
{
    g_seed = seed;
}

inline int fastrand()
{
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16)&0x7FFF;
}

#if !GPU_SUPPORT
enum class RPPTensorDataType
{
    U8 = 0,
    FP32,
    FP16,
    I8,
};

struct RPPTensorFunctionMetaData
{
    RPPTensorDataType _in_type = RPPTensorDataType::U8;
    RPPTensorDataType _out_type = RPPTensorDataType::U8;
    RppiChnFormat _in_format = RppiChnFormat::RPPI_CHN_PACKED;
    RppiChnFormat _out_format = RppiChnFormat::RPPI_CHN_PLANAR;
    Rpp32u _in_channels = 3;

    RPPTensorFunctionMetaData(RppiChnFormat in_chn_format, RPPTensorDataType in_tensor_type,
                              RPPTensorDataType out_tensor_type, Rpp32u in_channels,
                              bool out_format_change) : _in_format(in_chn_format), _in_type(in_tensor_type),
                                                        _out_type(out_tensor_type), _in_channels(in_channels)
    {
        if (out_format_change)
        {
            if (_in_format == RPPI_CHN_PLANAR)
                _out_format = RppiChnFormat::RPPI_CHN_PACKED;
            else
                _out_format = RppiChnFormat::RPPI_CHN_PLANAR;
        }
        else
            _out_format = _in_format;
    }
};
#endif // GPU_SUPPORT

// Computes strides for ND Tensor
inline void compute_strides(Rpp32u *strides, Rpp32u *shape, Rpp32u tensorDim)
{
    if (tensorDim > 0)
    {
        Rpp32u v = 1;
        for (Rpp32u i = tensorDim - 1; i > 0; i--)
        {
            strides[i] = v;
            v *= shape[i];
        }
        strides[0] = v;
    }
}

// Uses fast inverse square root algorithm from Lomont, C., 2003. FAST INVERSE SQUARE ROOT. [online] lomont.org. Available at: <http://www.lomont.org/papers/2003/InvSqrt.pdf>
inline float rpp_host_math_inverse_sqrt_1(float x)
{
    float xHalf = 0.5f * x;
    int i = *(int*)&x;                              // float bits in int
    i = NEWTON_METHOD_INITIAL_GUESS - (i >> 1);     // initial guess for Newton's method
    x = *(float*)&i;                                // new bits to float
    x = x * (1.5f - xHalf * x * x);                 // One round of Newton's method

    return x;
}

// SSE implementation of fast inverse square root algorithm from Lomont, C., 2003. FAST INVERSE SQUARE ROOT. [online] lomont.org. Available at: <http://www.lomont.org/papers/2003/InvSqrt.pdf>
inline __m128 rpp_host_math_inverse_sqrt_4_sse(__m128 p)
{
    __m128 pHalfNeg;
    __m128i pxI;
    pHalfNeg = _mm_mul_ps(_ps_n0p5, p);                                         // float xHalfNeg = -0.5f * x;
    pxI = *(__m128i *)&p;                                                       // int i = *(int*)&x;
    pxI = _mm_sub_epi32(xmm_newtonMethodInitialGuess, _mm_srli_epi32(pxI, 1));  // i = NEWTON_METHOD_INITIAL_GUESS - (i >> 1);
    p = *(__m128 *)&pxI;                                                        // x = *(float*)&i;
    p = _mm_mul_ps(p, _mm_fmadd_ps(p, _mm_mul_ps(p, pHalfNeg), _ps_1p5));       // x = x * (1.5f - xHalf * x * x);

    return p;
}

// AVX2 implementation of fast inverse square root algorithm from Lomont, C., 2003. FAST INVERSE SQUARE ROOT. [online] lomont.org. Available at: <http://www.lomont.org/papers/2003/InvSqrt.pdf>
inline __m256 rpp_host_math_inverse_sqrt_8_avx(__m256 p)
{
    __m256 pHalfNeg;
    __m256i pxI;
    pHalfNeg = _mm256_mul_ps(_ps_n0p5_avx, p);                                          // float xHalfNeg = -0.5f * x;
    pxI = *(__m256i *)&p;                                                               // int i = *(int*)&x;
    pxI = _mm256_sub_epi32(avx_newtonMethodInitialGuess, _mm256_srli_epi32(pxI, 1));    // i = NEWTON_METHOD_INITIAL_GUESS - (i >> 1);
    p = *(__m256 *)&pxI;                                                                // x = *(float*)&i;
    p = _mm256_mul_ps(p, _mm256_fmadd_ps(p, _mm256_mul_ps(p, pHalfNeg), _ps_1p5_avx));  // x = x * (1.5f - xHalf * x * x);

    return p;
}

inline Rpp32f rpp_host_math_exp_lim256approx(Rpp32f x)
{
  x = 1.0 + x * ONE_OVER_256;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;

  return x;
}

inline void rpp_host_math_fmadd8(__m256 *p, __m256 *pFmaddParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pFmaddParams[0], pFmaddParams[1]);    // fmadd adjustment
}

template<Rpp32s STREAM_SIZE>
inline void rpp_host_rng_xorwow_f32_initialize_multiseed_stream(RpptXorwowState *xorwowInitialState, Rpp32u seed)
{
    Rpp32u xorwowSeedStream[STREAM_SIZE];

    // Loop to initialize seed stream of size STREAM_SIZE based on user seed and offset
    for (int i = 0; i < STREAM_SIZE; i++)
        xorwowSeedStream[i] = seed + multiseedStreamOffset[i];

    // Loop to initialize STREAM_SIZE xorwow initial states for multi-stream random number generation
    for (int i = 0; i < STREAM_SIZE; i++)
    {
        xorwowInitialState[i].x[0] = 0x75BCD15 + xorwowSeedStream[i];      // state param x[0] offset 123456789U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[1] = 0x159A55E5 + xorwowSeedStream[i];     // state param x[1] offset 362436069U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[2] = 0x1F123BB5 + xorwowSeedStream[i];     // state param x[2] offset 521288629U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[3] = 0x5491333 + xorwowSeedStream[i];      // state param x[3] offset 88675123U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[4] = 0x583F19 + xorwowSeedStream[i];       // state param x[4] offset 5783321U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].counter = 0x64F0C9 + xorwowSeedStream[i];    // state param counter offset 6615241U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
    }
}

template<Rpp32s STREAM_SIZE>
inline void rpp_host_rng_xorwow_f32_initialize_multiseed_stream_boxmuller(RpptXorwowStateBoxMuller *xorwowInitialState, Rpp32u seed)
{
    Rpp32u xorwowSeedStream[STREAM_SIZE];

    // Loop to initialize seed stream of size STREAM_SIZE based on user seed and offset
    for (int i = 0; i < STREAM_SIZE; i++)
        xorwowSeedStream[i] = seed + multiseedStreamOffset[i];

    // Loop to initialize STREAM_SIZE xorwow initial states for multi-stream random number generation
    for (int i = 0; i < STREAM_SIZE; i++)
    {
        xorwowInitialState[i].x[0] = 0x75BCD15 + xorwowSeedStream[i];      // state param x[0] offset 123456789U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[1] = 0x159A55E5 + xorwowSeedStream[i];     // state param x[1] offset 362436069U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[2] = 0x1F123BB5 + xorwowSeedStream[i];     // state param x[2] offset 521288629U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[3] = 0x5491333 + xorwowSeedStream[i];      // state param x[3] offset 88675123U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[4] = 0x583F19 + xorwowSeedStream[i];       // state param x[4] offset 5783321U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].counter = 0x64F0C9 + xorwowSeedStream[i];    // state param counter offset 6615241U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].boxMullerFlag = 0;
        xorwowInitialState[i].boxMullerExtra = 0.0f;
    }
}

template<typename T>
inline void rpp_host_rng_xorwow_state_offsetted_avx(T *xorwowInitialStatePtr, T &xorwowState, Rpp32u offset, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter)
{
    xorwowState = xorwowInitialStatePtr[0];
    xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + offset;

    __m256i pxOffset = _mm256_set1_epi32(offset);
    pxXorwowStateX[0] = _mm256_add_epi32(_mm256_setr_epi32(xorwowInitialStatePtr[0].x[0], xorwowInitialStatePtr[1].x[0], xorwowInitialStatePtr[2].x[0], xorwowInitialStatePtr[3].x[0], xorwowInitialStatePtr[4].x[0], xorwowInitialStatePtr[5].x[0], xorwowInitialStatePtr[6].x[0], xorwowInitialStatePtr[7].x[0]), pxOffset);
    pxXorwowStateX[1] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[1], xorwowInitialStatePtr[1].x[1], xorwowInitialStatePtr[2].x[1], xorwowInitialStatePtr[3].x[1], xorwowInitialStatePtr[4].x[1], xorwowInitialStatePtr[5].x[1], xorwowInitialStatePtr[6].x[1], xorwowInitialStatePtr[7].x[1]);
    pxXorwowStateX[2] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[2], xorwowInitialStatePtr[1].x[2], xorwowInitialStatePtr[2].x[2], xorwowInitialStatePtr[3].x[2], xorwowInitialStatePtr[4].x[2], xorwowInitialStatePtr[5].x[2], xorwowInitialStatePtr[6].x[2], xorwowInitialStatePtr[7].x[2]);
    pxXorwowStateX[3] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[3], xorwowInitialStatePtr[1].x[3], xorwowInitialStatePtr[2].x[3], xorwowInitialStatePtr[3].x[3], xorwowInitialStatePtr[4].x[3], xorwowInitialStatePtr[5].x[3], xorwowInitialStatePtr[6].x[3], xorwowInitialStatePtr[7].x[3]);
    pxXorwowStateX[4] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[4], xorwowInitialStatePtr[1].x[4], xorwowInitialStatePtr[2].x[4], xorwowInitialStatePtr[3].x[4], xorwowInitialStatePtr[4].x[4], xorwowInitialStatePtr[5].x[4], xorwowInitialStatePtr[6].x[4], xorwowInitialStatePtr[7].x[4]);
    *pxXorwowStateCounter = _mm256_setr_epi32(xorwowInitialStatePtr[0].counter, xorwowInitialStatePtr[1].counter, xorwowInitialStatePtr[2].counter, xorwowInitialStatePtr[3].counter, xorwowInitialStatePtr[4].counter, xorwowInitialStatePtr[5].counter, xorwowInitialStatePtr[6].counter, xorwowInitialStatePtr[7].counter);
}

template<typename T>
inline void rpp_host_rng_xorwow_state_offsetted_sse(T *xorwowInitialStatePtr, T &xorwowState, Rpp32u offset, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter)
{
    xorwowState = xorwowInitialStatePtr[0];
    xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + offset;

    __m128i pxOffset = _mm_set1_epi32(offset);
    pxXorwowStateX[0] = _mm_add_epi32(_mm_setr_epi32(xorwowInitialStatePtr[0].x[0], xorwowInitialStatePtr[1].x[0], xorwowInitialStatePtr[2].x[0], xorwowInitialStatePtr[3].x[0]), pxOffset);
    pxXorwowStateX[1] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[1], xorwowInitialStatePtr[1].x[1], xorwowInitialStatePtr[2].x[1], xorwowInitialStatePtr[3].x[1]);
    pxXorwowStateX[2] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[2], xorwowInitialStatePtr[1].x[2], xorwowInitialStatePtr[2].x[2], xorwowInitialStatePtr[3].x[2]);
    pxXorwowStateX[3] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[3], xorwowInitialStatePtr[1].x[3], xorwowInitialStatePtr[2].x[3], xorwowInitialStatePtr[3].x[3]);
    pxXorwowStateX[4] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[4], xorwowInitialStatePtr[1].x[4], xorwowInitialStatePtr[2].x[4], xorwowInitialStatePtr[3].x[4]);
    *pxXorwowStateCounter = _mm_setr_epi32(xorwowInitialStatePtr[0].counter, xorwowInitialStatePtr[1].counter, xorwowInitialStatePtr[2].counter, xorwowInitialStatePtr[3].counter);
}

inline void rpp_host_rng_xorwow_8_state_update_avx(__m256i *pxXorwowStateXParam, __m256i *pxXorwowStateCounterParam)
{
    // Initialize avx-xorwow specific constants
    __m256i pxXorwowCounterInc = _mm256_set1_epi32(XORWOW_COUNTER_INC);

    // Save current first and last x-params of xorwow state and compute pxT
    __m256i pxT = pxXorwowStateXParam[0];                                                           // uint t  = xorwowState->x[0];
    __m256i pxS = pxXorwowStateXParam[4];                                                           // uint s  = xorwowState->x[4];
    pxT = _mm256_xor_si256(pxT, _mm256_srli_epi32(pxT, 2));                                         // t ^= t >> 2;
    pxT = _mm256_xor_si256(pxT, _mm256_slli_epi32(pxT, 1));                                         // t ^= t << 1;
    pxT = _mm256_xor_si256(pxT, _mm256_xor_si256(pxS, _mm256_slli_epi32(pxS, 4)));                  // t ^= s ^ (s << 4);

    // Update all 6 xorwow state params
    pxXorwowStateXParam[0] = pxXorwowStateXParam[1];                                                // xorwowState->x[0] = xorwowState->x[1];
    pxXorwowStateXParam[1] = pxXorwowStateXParam[2];                                                // xorwowState->x[1] = xorwowState->x[2];
    pxXorwowStateXParam[2] = pxXorwowStateXParam[3];                                                // xorwowState->x[2] = xorwowState->x[3];
    pxXorwowStateXParam[3] = pxXorwowStateXParam[4];                                                // xorwowState->x[3] = xorwowState->x[4];
    pxXorwowStateXParam[4] = pxT;                                                                   // xorwowState->x[4] = t;
    *pxXorwowStateCounterParam = _mm256_add_epi32(*pxXorwowStateCounterParam, pxXorwowCounterInc);  // xorwowState->counter += XORWOW_COUNTER_INC;
}

inline __m256i rpp_host_rng_xorwow_8_u32_avx(__m256i *pxXorwowStateXParam, __m256i *pxXorwowStateCounterParam)
{
    // Update xorwow state
    rpp_host_rng_xorwow_8_state_update_avx(pxXorwowStateXParam, pxXorwowStateCounterParam);

    // Return u32 random number
    return  _mm256_add_epi32(pxXorwowStateXParam[4], *pxXorwowStateCounterParam);   // return x[4] + counter
}

inline __m256 rpp_host_rng_xorwow_8_f32_avx(__m256i *pxXorwowStateXParam, __m256i *pxXorwowStateCounterParam)
{
    // Update xorwow state
    rpp_host_rng_xorwow_8_state_update_avx(pxXorwowStateXParam, pxXorwowStateCounterParam);

    // Initialize avx-xorwow specific constants
    __m256i px7FFFFF = _mm256_set1_epi32(0x7FFFFF);
    __m256i pxExponentFloat = _mm256_set1_epi32(XORWOW_EXPONENT_MASK);

    // Create float representation and return 0 <= pxS < 1
    __m256i pxS = _mm256_or_si256(pxExponentFloat, _mm256_and_si256(_mm256_add_epi32(pxXorwowStateXParam[4], *pxXorwowStateCounterParam), px7FFFFF));   // uint out = (XORWOW_EXPONENT_MASK | ((xorwowState->x[4] + xorwowState->counter) & 0x7FFFFF));
    return _mm256_sub_ps(*(__m256 *)&pxS, avx_p1);                                                                                                      // return  *(float *)&out - 1;
}

inline void rpp_host_rng_xorwow_4_state_update_sse(__m128i *pxXorwowStateXParam, __m128i *pxXorwowStateCounterParam)
{
    // Initialize sse-xorwow specific constants
    __m128i pxXorwowCounterInc = _mm_set1_epi32(XORWOW_COUNTER_INC);

    // Save current first and last x-params of xorwow state and compute pxT
    __m128i pxT = pxXorwowStateXParam[0];                                                       // uint t  = xorwowState->x[0];
    __m128i pxS = pxXorwowStateXParam[4];                                                       // uint s  = xorwowState->x[4];
    pxT = _mm_xor_si128(pxT, _mm_srli_epi32(pxT, 2));                                           // t ^= t >> 2;
    pxT = _mm_xor_si128(pxT, _mm_slli_epi32(pxT, 1));                                           // t ^= t << 1;
    pxT = _mm_xor_si128(pxT, _mm_xor_si128(pxS, _mm_slli_epi32(pxS, 4)));                       // t ^= s ^ (s << 4);

    // Update all 6 xorwow state params
    pxXorwowStateXParam[0] = pxXorwowStateXParam[1];                                            // xorwowState->x[0] = xorwowState->x[1];
    pxXorwowStateXParam[1] = pxXorwowStateXParam[2];                                            // xorwowState->x[1] = xorwowState->x[2];
    pxXorwowStateXParam[2] = pxXorwowStateXParam[3];                                            // xorwowState->x[2] = xorwowState->x[3];
    pxXorwowStateXParam[3] = pxXorwowStateXParam[4];                                            // xorwowState->x[3] = xorwowState->x[4];
    pxXorwowStateXParam[4] = pxT;                                                               // xorwowState->x[4] = t;
    *pxXorwowStateCounterParam = _mm_add_epi32(*pxXorwowStateCounterParam, pxXorwowCounterInc); // xorwowState->counter += XORWOW_COUNTER_INC;
}

inline __m128i rpp_host_rng_xorwow_4_u32_sse(__m128i *pxXorwowStateXParam, __m128i *pxXorwowStateCounterParam)
{
    // Update xorwow state
    rpp_host_rng_xorwow_4_state_update_sse(pxXorwowStateXParam, pxXorwowStateCounterParam);

    // Return u32 random number
    return  _mm_add_epi32(pxXorwowStateXParam[4], *pxXorwowStateCounterParam);   // return x[4] + counter
}

inline __m128 rpp_host_rng_xorwow_4_f32_sse(__m128i *pxXorwowStateXParam, __m128i *pxXorwowStateCounterParam)
{
    // Update xorwow state
    rpp_host_rng_xorwow_4_state_update_sse(pxXorwowStateXParam, pxXorwowStateCounterParam);

    // Initialize sse-xorwow specific constants
    __m128i px7FFFFF = _mm_set1_epi32(0x7FFFFF);
    __m128i pxExponentFloat = _mm_set1_epi32(XORWOW_EXPONENT_MASK);

    // Create float representation and return 0 <= pxS < 1
    __m128i pxS = _mm_or_si128(pxExponentFloat, _mm_and_si128(_mm_add_epi32(pxXorwowStateXParam[4], *pxXorwowStateCounterParam), px7FFFFF));    // uint out = (XORWOW_EXPONENT_MASK | ((xorwowState->x[4] + xorwowState->counter) & 0x7FFFFF));
    return _mm_sub_ps(*(__m128 *)&pxS, xmm_p1);                                                                                                 // return  *(float *)&out - 1;
}

template<typename T>
inline void rpp_host_rng_xorwow_state_update(T *xorwowState)
{
    // Save current first and last x-params of xorwow state and compute t
    Rpp32s t  = xorwowState->x[0];
    Rpp32s s  = xorwowState->x[4];
    t ^= t >> 2;
    t ^= t << 1;
    t ^= s ^ (s << 4);

    // Update all 6 xorwow state params
    xorwowState->x[0] = xorwowState->x[1];                              // set new state param x[0]
    xorwowState->x[1] = xorwowState->x[2];                              // set new state param x[1]
    xorwowState->x[2] = xorwowState->x[3];                              // set new state param x[2]
    xorwowState->x[3] = xorwowState->x[4];                              // set new state param x[3]
    xorwowState->x[4] = t;                                              // set new state param x[4]
    xorwowState->counter = xorwowState->counter + XORWOW_COUNTER_INC;   // set new state param counter
}

template<typename T>
inline Rpp32u rpp_host_rng_xorwow_u32(T *xorwowState)
{
    // Update xorwow state
    rpp_host_rng_xorwow_state_update(xorwowState);

    // Return u32 random number
    return  xorwowState->x[4] + xorwowState->counter;   // return x[4] + counter
}

template<typename T>
inline Rpp32f rpp_host_rng_xorwow_f32(T *xorwowState)
{
    // Update xorwow state
    rpp_host_rng_xorwow_state_update(xorwowState);

    // Create float representation and return 0 <= outFloat < 1
    Rpp32u out = (XORWOW_EXPONENT_MASK | ((xorwowState->x[4] + xorwowState->counter) & 0x7FFFFF));  // bitmask 23 mantissa bits, OR with exponent
    Rpp32f outFloat = *(Rpp32f *)&out;                                                              // reinterpret out as float
    return  outFloat - 1;                                                                           // return 0 <= outFloat < 1
}

inline void rpp_host_rng_16_gaussian_f32_avx(__m256 *pRngVals, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter)
{
    __m256 pU, pV, pS;                                                                                  // Rpp32f u, v, s;
    pU = _mm256_cvtepi32_ps(rpp_host_rng_xorwow_8_u32_avx(pxXorwowStateX, pxXorwowStateCounter));       // u = (Rpp32f)rpp_host_rng_xorwow_u32(xorwowState);
    pV = _mm256_cvtepi32_ps(rpp_host_rng_xorwow_8_u32_avx(pxXorwowStateX, pxXorwowStateCounter));       // v = (Rpp32f)rpp_host_rng_xorwow_u32(xorwowState);
    pS = _mm256_cmp_ps(pU, avx_p0, _CMP_LT_OQ);                                                         // Adjust int32 out of bound values in float for u
    pU = _mm256_or_ps(_mm256_andnot_ps(pS, pU), _mm256_and_ps(pS, _mm256_add_ps(avx_p2Pow32, pU)));     // Adjust int32 out of bound values in float for u
    pS = _mm256_cmp_ps(pV, avx_p0, _CMP_LT_OQ);                                                         // Adjust int32 out of bound values in float for v
    pV = _mm256_or_ps(_mm256_andnot_ps(pS, pV), _mm256_and_ps(pS, _mm256_add_ps(avx_p2Pow32, pV)));     // Adjust int32 out of bound values in float for v
    pU = _mm256_fmadd_ps(pU, avx_p2Pow32Inv, avx_p2Pow32InvDiv2);                                       // u = u * RPP_2POW32_INV + RPP_2POW32_INV_DIV_2;
    pV = _mm256_fmadd_ps(pV, avx_p2Pow32InvMul2Pi, avx_p2Pow32InvMul2PiDiv2);                           // v = v * RPP_2POW32_INV_MUL_2PI + RPP_2POW32_INV_MUL_2PI_DIV_2;
    pS = _mm256_sqrt_ps(_mm256_mul_ps(avx_pm2, log_ps(pU)));                                            // s = sqrt(-2.0f * std::log(u));
    sincos_ps(pV, &pU, &pV);                                                                            // std::sin(v) and std::cos(v) computation
    pRngVals[0] = _mm256_mul_ps(pU, pS);                                                                // u = std::sin(v) * s;
    pRngVals[1] = _mm256_mul_ps(pV, pS);                                                                // v = std::cos(v) * s;
}

inline void rpp_host_rng_8_gaussian_f32_sse(__m128 *pRngVals, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter)
{
    __m128 pU, pV, pS;                                                                          // Rpp32f u, v, s;
    pU = _mm_cvtepi32_ps(rpp_host_rng_xorwow_4_u32_sse(pxXorwowStateX, pxXorwowStateCounter));  // u = (Rpp32f)rpp_host_rng_xorwow_u32(xorwowState);
    pV = _mm_cvtepi32_ps(rpp_host_rng_xorwow_4_u32_sse(pxXorwowStateX, pxXorwowStateCounter));  // v = (Rpp32f)rpp_host_rng_xorwow_u32(xorwowState);
    pS = _mm_cmplt_ps(pU, xmm_p0);                                                              // Adjust int32 out of bound values in float for u
    pU = _mm_or_ps(_mm_andnot_ps(pS, pU), _mm_and_ps(pS, _mm_add_ps(xmm_p2Pow32, pU)));         // Adjust int32 out of bound values in float for u
    pS = _mm_cmplt_ps(pV, xmm_p0);                                                              // Adjust int32 out of bound values in float for v
    pV = _mm_or_ps(_mm_andnot_ps(pS, pV), _mm_and_ps(pS, _mm_add_ps(xmm_p2Pow32, pV)));         // Adjust int32 out of bound values in float for v
    pU = _mm_fmadd_ps(pU, xmm_p2Pow32Inv, xmm_p2Pow32InvDiv2);                                  // u = u * RPP_2POW32_INV + RPP_2POW32_INV_DIV_2;
    pV = _mm_fmadd_ps(pV, xmm_p2Pow32InvMul2Pi, xmm_p2Pow32InvMul2PiDiv2);                      // v = v * RPP_2POW32_INV_MUL_2PI + RPP_2POW32_INV_MUL_2PI_DIV_2;
    pS = _mm_sqrt_ps(_mm_mul_ps(xmm_pm2, log_ps(pU)));                                          // s = sqrt(-2.0f * std::log(u));
    sincos_ps(pV, &pU, &pV);                                                                    // std::sin(v) and std::cos(v) computation
    pRngVals[0] = _mm_mul_ps(pU, pS);                                                           // u = std::sin(v) * s;
    pRngVals[1] = _mm_mul_ps(pV, pS);                                                           // v = std::cos(v) * s;
}

inline float rpp_host_rng_1_gaussian_f32(RpptXorwowStateBoxMuller *xorwowState)
{
    if(!xorwowState->boxMullerFlag)
    {
        Rpp32f u, v, s;
        u = (Rpp32f)rpp_host_rng_xorwow_u32(xorwowState) * RPP_2POW32_INV + RPP_2POW32_INV_DIV_2;
        v = (Rpp32f)rpp_host_rng_xorwow_u32(xorwowState) * RPP_2POW32_INV_MUL_2PI + RPP_2POW32_INV_MUL_2PI_DIV_2;
        s = sqrt(-2.0f * std::log(u));
        u = std::sin(v) * s;
        v = std::cos(v) * s;
        xorwowState->boxMullerExtra = v;
        xorwowState->boxMullerFlag = 1;
        return u;
    }
    xorwowState->boxMullerFlag = 0;

    return xorwowState->boxMullerExtra;
}

inline int power_function(int a, int b)
{
    int product = 1;
    for(int i = 0; i < b; i++)
        product *= product * a;
    return product;
}

inline void saturate_pixel(Rpp32f &pixel, Rpp8u* dst)
{
    *dst = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel)));
}

inline void saturate_pixel(Rpp32f &pixel, Rpp8s* dst)
{
    *dst = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(pixel) - 128));
}

inline void saturate_pixel(Rpp32f &pixel, Rpp32f* dst)
{
    *dst = RPPPIXELCHECKF32(pixel);
}

inline void saturate_pixel(Rpp32f &pixel, Rpp16f* dst)
{
    *dst = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel));
}

template <typename T>
RppStatus subtract_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus add_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus multiply_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus min_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus max_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);

template <typename T>
RppStatus bitwise_AND_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus inclusive_OR_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus exclusive_OR_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);


// Specific Helper Functions

inline Rpp32f gaussian_2d_relative(Rpp32s locI, Rpp32s locJ, Rpp32f std_dev)
{
    Rpp32f relativeGaussian;
    Rpp32f exp1, exp2;
    exp1 = -(locJ * locJ) / (2 * std_dev * std_dev);
    exp2 = -(locI * locI) / (2 * std_dev * std_dev);
    relativeGaussian = exp(exp1 + exp2);

    return relativeGaussian;
}

// Generate Functions

inline void generate_gaussian_kernel_host(Rpp32f stdDev, Rpp32f* kernel, Rpp32u kernelSize)
{
    Rpp32f s, sum = 0.0, multiplier;
    int bound = ((kernelSize - 1) / 2);
    Rpp32u c = 0;
    s = 1 / (2 * stdDev * stdDev);
    multiplier = (1 / M_PI) * (s);
    for (int i = -bound; i <= bound; i++)
    {
        for (int j = -bound; j <= bound; j++)
        {
            kernel[c] = multiplier * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        kernel[i] /= sum;
    }
}

inline RppStatus generate_gaussian_kernel_asymmetric_host(Rpp32f stdDev, Rpp32f* kernel, Rpp32u kernelSizeX, Rpp32u kernelSizeY)
{
    Rpp32f s, sum = 0.0, multiplier;
    if (kernelSizeX % 2 == 0)
    {
        return RPP_ERROR;
    }
    if (kernelSizeY % 2 == 0)
    {
        return RPP_ERROR;
    }
    int boundX = ((kernelSizeX - 1) / 2);
    int boundY = ((kernelSizeY - 1) / 2);
    Rpp32u c = 0;
    s = 1 / (2 * stdDev * stdDev);
    multiplier = (1 / M_PI) * (s);
    for (int i = -boundY; i <= boundY; i++)
    {
        for (int j = -boundX; j <= boundX; j++)
        {
            kernel[c] = multiplier * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSizeX * kernelSizeY); i++)
    {
        kernel[i] /= sum;
    }
    return RPP_SUCCESS;
}

template <typename T>
inline void generate_bilateral_kernel_host(Rpp32f multiplierI, Rpp32f multiplierS, Rpp32f multiplier, Rpp32f* kernel, Rpp32u kernelSize, int bound,
                                         T* srcPtrWindow, RppiSize srcSizeMod, Rpp32u remainingElementsInRow, Rpp32u incrementToWindowCenter,
                                         RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f sum = 0.0;
    Rpp32f* kernelTemp;
    kernelTemp = kernel;

    T *srcPtrWindowTemp, *srcPtrWindowCenter;
    srcPtrWindowTemp = srcPtrWindow;
    srcPtrWindowCenter = srcPtrWindow + incrementToWindowCenter;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int i = -bound; i <= bound; i++)
        {
            for (int j = -bound; j <= bound; j++)
            {
                T pixel = *srcPtrWindowCenter - *srcPtrWindowTemp;
                pixel = RPPABS(pixel);
                pixel = pixel * pixel;
                *kernelTemp = multiplier * exp((multiplierS * (i*i + j*j)) + (multiplierI * pixel));
                sum = sum + *kernelTemp;
                kernelTemp++;
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = -bound; i <= bound; i++)
        {
            for (int j = -bound; j <= bound; j++)
            {
                T pixel = *srcPtrWindowCenter - *srcPtrWindowTemp;
                pixel = RPPABS(pixel);
                pixel = pixel * pixel;
                *kernelTemp = multiplier * exp((multiplierS * (i*i + j*j)) + (multiplierI * pixel));
                sum = sum + *kernelTemp;
                kernelTemp++;
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }

    kernelTemp = kernel;
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        *kernelTemp = *kernelTemp / sum;
        kernelTemp++;
    }
}

template <typename T>
inline RppStatus generate_evenly_padded_image_host(T* srcPtr, RppiSize srcSize, T* srcPtrMod, RppiSize srcSizeMod,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    if (RPPISEVEN(srcSize.height) != RPPISEVEN(srcSizeMod.height)
        || RPPISEVEN(srcSize.width) != RPPISEVEN(srcSizeMod.width)
        || srcSizeMod.height < srcSize.height
        || srcSizeMod.width < srcSize.width)
    {
        return RPP_ERROR;
    }
    T *srcPtrTemp, *srcPtrModTemp;
    srcPtrTemp = srcPtr;
    srcPtrModTemp = srcPtrMod;
    int bound = (srcSizeMod.height - srcSize.height) / 2;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            memset (srcPtrModTemp,(T) 0,bound * srcSizeMod.width * sizeof(T));
            srcPtrModTemp += (bound * srcSizeMod.width);
            for (int i = 0; i < srcSize.height; i++)
            {
                memset (srcPtrModTemp,(T) 0,bound * sizeof(T));
                srcPtrModTemp += bound;

                memcpy(srcPtrModTemp, srcPtrTemp, srcSize.width * sizeof(T));
                srcPtrModTemp += srcSize.width;
                srcPtrTemp += srcSize.width;

                memset (srcPtrModTemp,(T) 0,bound * sizeof(T));
                srcPtrModTemp += bound;
            }
            memset (srcPtrModTemp,(T) 0,bound * srcSizeMod.width * sizeof(T));
            srcPtrModTemp += (bound * srcSizeMod.width);
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u numOfPixelsVtBorder = bound * channel;
        Rpp32u numOfPixelsHrBorder = numOfPixelsVtBorder * srcSizeMod.width;

        memset (srcPtrModTemp,(T) 0,numOfPixelsHrBorder * sizeof(T));
        srcPtrModTemp += (numOfPixelsHrBorder);

        for (int i = 0; i < srcSize.height; i++)
        {
            memset (srcPtrModTemp,(T) 0,numOfPixelsVtBorder * sizeof(T));
            srcPtrModTemp += (numOfPixelsVtBorder);

            memcpy(srcPtrModTemp, srcPtrTemp, elementsInRow * sizeof(T));
            srcPtrModTemp += elementsInRow;
            srcPtrTemp += elementsInRow;

            memset (srcPtrModTemp,(T) 0,numOfPixelsVtBorder * sizeof(T));
            srcPtrModTemp += (numOfPixelsVtBorder);
        }

        memset (srcPtrModTemp,(T) 0,numOfPixelsHrBorder * sizeof(T));
        srcPtrModTemp += (numOfPixelsHrBorder);
    }

    return RPP_SUCCESS;
}

template <typename T>
inline void generate_corner_padded_image_host(T* srcPtr, RppiSize srcSize, T* srcPtrMod, RppiSize srcSizeMod, Rpp32u padType,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *srcPtrModTemp;
    srcPtrTemp = srcPtr;
    srcPtrModTemp = srcPtrMod;
    Rpp32u boundY = srcSizeMod.height - srcSize.height;
    Rpp32u boundX = srcSizeMod.width - srcSize.width;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            if (padType == 1 || padType == 2)
            {
                memset (srcPtrModTemp,(T) 0,boundY * srcSizeMod.width * sizeof(T));
                srcPtrModTemp += (boundY * srcSizeMod.width);
            }

            if (padType == 1 || padType == 3)
            {
                for (int i = 0; i < srcSize.height; i++)
                {
                    memset (srcPtrModTemp,(T) 0,boundX * sizeof(T));
                    srcPtrModTemp += boundX;

                    memcpy(srcPtrModTemp, srcPtrTemp, srcSize.width * sizeof(T));
                    srcPtrModTemp += srcSize.width;
                    srcPtrTemp += srcSize.width;
                }
            }

            if (padType == 2 || padType == 4)
            {
                for (int i = 0; i < srcSize.height; i++)
                {
                    memcpy(srcPtrModTemp, srcPtrTemp, srcSize.width * sizeof(T));
                    srcPtrModTemp += srcSize.width;
                    srcPtrTemp += srcSize.width;

                    memset (srcPtrModTemp,(T) 0,boundX * sizeof(T));
                    srcPtrModTemp += boundX;
                }
            }

            if (padType == 3 || padType == 4)
            {
                memset (srcPtrModTemp,(T) 0,boundY * srcSizeMod.width * sizeof(T));
                srcPtrModTemp += (boundY * srcSizeMod.width);
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u numOfPixelsVtBorder = boundX * channel;
        Rpp32u numOfPixelsHrBorder = boundY * channel * srcSizeMod.width;

        if (padType == 1 || padType == 2)
        {
            memset (srcPtrModTemp,(T) 0,numOfPixelsHrBorder * sizeof(T));
            srcPtrModTemp += (numOfPixelsHrBorder);
        }

        if (padType == 1 || padType == 3)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                memset (srcPtrModTemp,(T) 0,numOfPixelsVtBorder * sizeof(T));
                srcPtrModTemp += (numOfPixelsVtBorder);

                memcpy(srcPtrModTemp, srcPtrTemp, elementsInRow * sizeof(T));
                srcPtrModTemp += elementsInRow;
                srcPtrTemp += elementsInRow;
            }
        }

        if (padType == 2 || padType == 4)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                memcpy(srcPtrModTemp, srcPtrTemp, elementsInRow * sizeof(T));
                srcPtrModTemp += elementsInRow;
                srcPtrTemp += elementsInRow;

                memset (srcPtrModTemp,(T) 0,numOfPixelsVtBorder * sizeof(T));
                srcPtrModTemp += (numOfPixelsVtBorder);
            }
        }

        if (padType == 3 || padType == 4)
        {
            memset (srcPtrModTemp,(T) 0,numOfPixelsHrBorder * sizeof(T));
            srcPtrModTemp += (numOfPixelsHrBorder);
        }
    }
}

inline void generate_box_kernel_host(Rpp32f* kernel, Rpp32u kernelSize)
{
    Rpp32f* kernelTemp;
    kernelTemp = kernel;
    Rpp32f kernelValue = 1.0 / (Rpp32f) (kernelSize * kernelSize);
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        *kernelTemp = kernelValue;
        kernelTemp++;
    }
}

template <typename T>
inline void generate_crop_host(T* srcPtr, RppiSize srcSize, T* srcPtrSubImage, RppiSize srcSizeSubImage, T* dstPtr,
                             RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrSubImageTemp, *dstPtrTemp;
    srcPtrSubImageTemp = srcPtrSubImage;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = (srcSize.width - srcSizeSubImage.width);
        for (int c = 0; c < channel; c++)
        {
            srcPtrSubImageTemp = srcPtrSubImage + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSizeSubImage.height; i++)
            {
                Rpp32u bufferLength = srcSizeSubImage.width;
                Rpp32u alignedLength = bufferLength & ~15;

                __m128i px0;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrSubImageTemp);
                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrSubImageTemp +=16;
                    dstPtrTemp +=16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = *srcPtrSubImageTemp++;
                }
                srcPtrSubImageTemp += remainingElementsInRow;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = channel * (srcSize.width - srcSizeSubImage.width);
        Rpp32u elementsInRowCrop = channel * srcSizeSubImage.width;
        for (int i = 0; i < srcSizeSubImage.height; i++)
        {
            Rpp32u bufferLength = elementsInRowCrop;
            Rpp32u alignedLength = bufferLength & ~15;

            __m128i px0;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrSubImageTemp);
                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                srcPtrSubImageTemp +=16;
                dstPtrTemp +=16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                *dstPtrTemp++ = *srcPtrSubImageTemp++;
            }
            srcPtrSubImageTemp += remainingElementsInRow;
        }
    }
}

inline RppStatus generate_sobel_kernel_host(Rpp32f* kernel, Rpp32u type)
{
    Rpp32f* kernelTemp;
    kernelTemp = kernel;

    if (type == 1)
    {
        Rpp32f kernelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        Rpp32f* kernelXTemp;
        kernelXTemp = kernelX;

        for (int i = 0; i < 9; i++)
        {
            *kernelTemp = *kernelXTemp;
            kernelTemp++;
            kernelXTemp++;
        }
    }
    else if (type == 2)
    {
        Rpp32f kernelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
        Rpp32f* kernelYTemp;
        kernelYTemp = kernelY;

        for (int i = 0; i < 9; i++)
        {
            *kernelTemp = *kernelYTemp;
            kernelTemp++;
            kernelYTemp++;
        }
    }
    else
    {
        return RPP_ERROR;
    }

    return RPP_SUCCESS;
}

template <typename T>
inline void generate_bressenham_line_host(T *dstPtr, RppiSize dstSize, Rpp32u *endpoints, Rpp32u *rasterCoordinates)
{
    Rpp32u *rasterCoordinatesTemp;
    rasterCoordinatesTemp = rasterCoordinates;

    Rpp32s x0 = *endpoints;
    Rpp32s y0 = *(endpoints + 1);
    Rpp32s x1 = *(endpoints + 2);
    Rpp32s y1 = *(endpoints + 3);

    Rpp32s dx, dy;
    Rpp32s stepX, stepY;

    dx = x1 - x0;
    dy = y1 - y0;

    if (dy < 0)
    {
        dy = -dy;
        stepY = -1;
    }
    else
    {
        stepY = 1;
    }

    if (dx < 0)
    {
        dx = -dx;
        stepX = -1;
    }
    else
    {
        stepX = 1;
    }

    dy <<= 1;
    dx <<= 1;

    if ((0 <= x0) && (x0 < dstSize.width) && (0 <= y0) && (y0 < dstSize.height))
    {
        *(dstPtr + (y0 * dstSize.width) + x0) = (T) 255;
        *rasterCoordinatesTemp = y0;
        rasterCoordinatesTemp++;
        *rasterCoordinatesTemp = x0;
        rasterCoordinatesTemp++;
    }

    if (dx > dy)
    {
        Rpp32s fraction = dy - (dx >> 1);
        while (x0 != x1)
        {
            x0 += stepX;
            if (fraction >= 0)
            {
                y0 += stepY;
                fraction -= dx;
            }
            fraction += dy;
            if ((0 <= x0) && (x0 < dstSize.width) && (0 <= y0) && (y0 < dstSize.height))
            {
                *(dstPtr + (y0 * dstSize.width) + x0) = (T) 255;
                *rasterCoordinatesTemp = y0;
                rasterCoordinatesTemp++;
                *rasterCoordinatesTemp = x0;
                rasterCoordinatesTemp++;
            }
        }
    }
    else
    {
        int fraction = dx - (dy >> 1);
        while (y0 != y1)
        {
            if (fraction >= 0)
            {
                x0 += stepX;
                fraction -= dy;
            }
            y0 += stepY;
            fraction += dx;
            if ((0 <= x0) && (x0 < dstSize.width) && (0 <= y0) && (y0 < dstSize.height))
            {
                *(dstPtr + (y0 * dstSize.width) + x0) = (T) 255;
                *rasterCoordinatesTemp = y0;
                rasterCoordinatesTemp++;
                *rasterCoordinatesTemp = x0;
                rasterCoordinatesTemp++;
            }
        }
    }
}

// copy ROI of voxel data from input to output
template<typename T>
void copy_3d_host_tensor(T *srcPtr,
                         RpptGenericDescPtr srcGenericDescPtr,
                         T *dstPtr,
                         RpptGenericDescPtr dstGenericDescPtr,
                         RpptROI3D *roi,
                         RppLayoutParams layoutParams)
{
    if((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
    {
        T *srcPtrDepth = srcPtr + (roi->xyzwhdROI.xyz.z * srcGenericDescPtr->strides[1]) + (roi->xyzwhdROI.xyz.y * srcGenericDescPtr->strides[2]) + (roi->xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
        T *dstPtrDepth = dstPtr;
        Rpp32u width = roi->xyzwhdROI.roiWidth * srcGenericDescPtr->dims[4];
        for(int i = 0; i < roi->xyzwhdROI.roiDepth; i++)
        {
            T *srcPtrRow = srcPtrDepth;
            T *dstPtrRow = dstPtrDepth;
            for(int j = 0; j < roi->xyzwhdROI.roiHeight; j++)
            {
                memcpy(dstPtrRow, srcPtrRow, width * sizeof(T));
                srcPtrRow += srcGenericDescPtr->strides[2];
                dstPtrRow += dstGenericDescPtr->strides[2];
            }
            srcPtrDepth += srcGenericDescPtr->strides[1];
            dstPtrDepth += dstGenericDescPtr->strides[1];
        }
    }
    else if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
    {
        T *srcPtrChannel = srcPtr + (roi->xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2]) + (roi->xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3]) + (roi->xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
        T *dstPtrChannel = dstPtr;
        int channels = srcGenericDescPtr->dims[1];
        for(int c = 0; c < channels; c++)
        {
            T *srcPtrDepth = srcPtrChannel;
            T *dstPtrDepth = dstPtrChannel;
            for(int i = 0; i < roi->xyzwhdROI.roiDepth; i++)
            {
                T *srcPtrRow = srcPtrDepth;
                T *dstPtrRow = dstPtrDepth;
                for(int j = 0; j < roi->xyzwhdROI.roiHeight; j++)
                {
                    memcpy(dstPtrRow, srcPtrRow, roi->xyzwhdROI.roiWidth * sizeof(T));
                    srcPtrRow += srcGenericDescPtr->strides[3];
                    dstPtrRow += dstGenericDescPtr->strides[3];
                }
                srcPtrDepth += srcGenericDescPtr->strides[2];
                dstPtrDepth += dstGenericDescPtr->strides[2];
            }
            srcPtrChannel += srcGenericDescPtr->strides[1];
            dstPtrChannel += dstGenericDescPtr->strides[1];
        }
    }
}














// Kernels for functions

template<typename T, typename U>
inline void convolution_kernel_host(T* srcPtrWindow, U* dstPtrPixel, RppiSize srcSize,
                                       Rpp32f* kernel, RppiSize kernelSize, Rpp32u remainingElementsInRow, U maxVal, U minVal,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f pixel = 0.0;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;

    Rpp32f* kernelPtrTemp;
    kernelPtrTemp = kernel;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize.height; m++)
        {
            for (int n = 0; n < kernelSize.width; n++)
            {
                pixel += ((*kernelPtrTemp) * (Rpp32f)(*srcPtrWindowTemp));
                kernelPtrTemp++;
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize.height; m++)
        {
            for (int n = 0; n < kernelSize.width; n++)
            {
                pixel += ((*kernelPtrTemp) * (Rpp32f)(*srcPtrWindowTemp));
                kernelPtrTemp++;
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    (pixel < (Rpp32f) minVal) ? pixel = (Rpp32f) minVal : ((pixel < (Rpp32f) maxVal) ? pixel : pixel = (Rpp32f) maxVal);
    *dstPtrPixel = (U) round(pixel);
}

template<typename T>
inline void histogram_kernel_host(T* srcPtr, RppiSize srcSize, Rpp32u* histogram,
                                Rpp8u bins,
                                Rpp32u channel)
{
    if (bins == 0)
    {
        *histogram = channel * srcSize.height * srcSize.width;
    }
    else
    {
        Rpp8u rangeInBin = 256 / (bins + 1);
        T *srcPtrTemp;
        srcPtrTemp = srcPtr;
        for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
        {
            *(histogram + (*srcPtrTemp / rangeInBin)) += 1;
            srcPtrTemp++;
        }
    }
}

template <typename T, typename U>
inline void accumulate_kernel_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize,
                                        RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr1Temp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;

    Rpp32s pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *srcPtr1Temp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
    }
}

template <typename U>
inline void normalize_kernel_host(U* dstPtrROI, RppiSize dstSize, Rpp32u channel)
{
    U* dstPtrROITemp;
    dstPtrROITemp = dstPtrROI;

    U multiplier = (U) (1.0 / 255.0);

    Rpp32u imageDim = dstSize.height * dstSize.width * channel;

    for (int i = 0; i < imageDim; i++)
    {
        *dstPtrROITemp = *dstPtrROITemp * multiplier;
        dstPtrROITemp++;
    }
}

template <typename T, typename U>
inline RppStatus resize_kernel_host(T* srcPtr, RppiSize srcSize, U* dstPtr, RppiSize dstSize,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (dstSize.height < 0 || dstSize.width < 0)
        {
            return RPP_ERROR;
        }

        Rpp32f hRatio = (((Rpp32f) (dstSize.height - 1)) / ((Rpp32f) (srcSize.height - 1)));
        Rpp32f wRatio = (((Rpp32f) (dstSize.width - 1)) / ((Rpp32f) (srcSize.width - 1)));
        Rpp32f srcLocationRow, srcLocationColumn, pixel;
        Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
        T *srcPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
        U *dstPtrTemp;
        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;

        if ((typeid(Rpp16f) == typeid(T)) || (typeid(Rpp16f) == typeid(U)))
        {
            for (int c = 0; c < channel; c++)
            {
                for (int i = 0; i < dstSize.height; i++)
                {
                    srcLocationRow = ((Rpp32f) i) / hRatio;
                    srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    if (srcLocationRowFloor > (srcSize.height - 2))
                    {
                        srcLocationRowFloor = srcSize.height - 2;
                    }

                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                    srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                    for (int j = 0; j < dstSize.width; j++)
                    {
                        srcLocationColumn = ((Rpp32f) j) / wRatio;
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                        if (srcLocationColumnFloor > (srcSize.width - 2))
                        {
                            srcLocationColumnFloor = srcSize.width - 2;
                        }
                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (U) pixel;
                        dstPtrTemp ++;
                    }
                }
                srcPtrTemp += srcSize.height * srcSize.width;
            }
        }
        else
        {
            for (int c = 0; c < channel; c++)
            {
                for (int i = 0; i < dstSize.height; i++)
                {
                    srcLocationRow = ((Rpp32f) i) / hRatio;
                    srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    if (srcLocationRowFloor > (srcSize.height - 2))
                    {
                        srcLocationRowFloor = srcSize.height - 2;
                    }

                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                    srcPtrBottomRow  = srcPtrTopRow + srcSize.width;
#pragma omp simd
                    for (int j = 0; j < dstSize.width; j++)
                    {
                        srcLocationColumn = ((Rpp32f) j) / wRatio;
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                        if (srcLocationColumnFloor > (srcSize.width - 2))
                        {
                            srcLocationColumnFloor = srcSize.width - 2;
                        }
                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (U) pixel;
                        dstPtrTemp ++;
                    }
                }
                srcPtrTemp += srcSize.height * srcSize.width;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        if (dstSize.height < 0 || dstSize.width < 0)
        {
            return RPP_ERROR;
        }

        Rpp32f hRatio = (((Rpp32f) (dstSize.height - 1)) / ((Rpp32f) (srcSize.height - 1)));
        Rpp32f wRatio = (((Rpp32f) (dstSize.width - 1)) / ((Rpp32f) (srcSize.width - 1)));
        Rpp32f srcLocationRow, srcLocationColumn, pixel;
        Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
        T *srcPtrTemp;
        U *dstPtrTemp;
        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;

        Rpp32u heightLimit = srcSize.height - 2;
        Rpp32u widthLimit = srcSize.width - 2;

        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRow = ((Rpp32f) i) / hRatio;
            srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
            Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

            if (srcLocationRowFloor > heightLimit)
            {
                srcLocationRowFloor = heightLimit;
            }

            T *srcPtrTopRow, *srcPtrBottomRow;
            srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
            srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

            Rpp32u bufferLength = dstSize.width;
            Rpp32u alignedLength = (bufferLength / 4) * 4;

            Rpp32u srcLocCF[4] = {0};
            Rpp32f param1[4] = {0};
            Rpp32f param2[4] = {0};
            Rpp32f param3[4] = {0};
            Rpp32f param4[4] = {0};

            __m128 pWRatio = _mm_set1_ps(1.0 / wRatio);
            __m128 p0, p2, p4, p5, p6, p7, pColFloor;
            __m128 p1 = _mm_set1_ps(weightedHeight);
            __m128 p3 = _mm_set1_ps(1 - weightedHeight);
            __m128 pOne = _mm_set1_ps(1.0);
            __m128i pxColFloor;

            Rpp64u vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
            {
                p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                p0 = _mm_mul_ps(p0, pWRatio);
                pColFloor = _mm_floor_ps(p0);
                pxColFloor = _mm_cvtps_epi32(pColFloor);
                p0 = _mm_sub_ps(p0, pColFloor);
                p2  = _mm_sub_ps(pOne, p0);

                p4 = _mm_mul_ps(p3, p2);
                p5 = _mm_mul_ps(p3, p0);
                p6 = _mm_mul_ps(p1, p2);
                p7 = _mm_mul_ps(p1, p0);

                _mm_storeu_si128((__m128i*) srcLocCF, pxColFloor);
                _mm_storeu_ps(param1, p4);
                _mm_storeu_ps(param2, p5);
                _mm_storeu_ps(param3, p6);
                _mm_storeu_ps(param4, p7);

                for (int pos = 0; pos < 4; pos++)
                {
                    if (srcLocCF[pos] > widthLimit)
                    {
                        srcLocCF[pos] = widthLimit;
                    }
                    srcLocCF[pos] *= channel;

                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp++ = (U) ((*(srcPtrTopRow + c + srcLocCF[pos])) * param1[pos])
                                            + ((*(srcPtrTopRow + c + srcLocCF[pos] + channel)) * param2[pos])
                                            + ((*(srcPtrBottomRow + c + srcLocCF[pos])) * param3[pos])
                                            + ((*(srcPtrBottomRow + c + srcLocCF[pos] + channel)) * param4[pos]);
                    }
                }
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                srcLocationColumn = ((Rpp32f) vectorLoopCount) / wRatio;
                srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                if (srcLocationColumnFloor > (srcSize.width - 2))
                {
                    srcLocationColumnFloor = srcSize.width - 2;
                }

                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                for (int c = 0; c < channel; c++)
                {
                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth))
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                    *dstPtrTemp = (U) pixel;
                    dstPtrTemp ++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
inline void resize_crop_kernel_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    RppiSize srcSizeSubImage;
    T *srcPtrSubImage;

    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    T *srcPtrResize = (T*) calloc(channel * srcSizeSubImage.height * srcSizeSubImage.width, sizeof(T));

    generate_crop_host(srcPtr, srcSize, srcPtrSubImage, srcSizeSubImage, srcPtrResize, chnFormat, channel);

    resize_kernel_host(srcPtrResize, srcSizeSubImage, dstPtr, dstSize, chnFormat, channel);

    free(srcPtrResize);
}

template<typename T>
inline void erode_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T pixel;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    pixel = *srcPtrWindowTemp;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp < pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp < pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    *dstPtrPixel = pixel;
}

template<typename T>
inline void dilate_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T pixel;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    pixel = *srcPtrWindowTemp;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp > pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp > pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    *dstPtrPixel = pixel;
}

template<typename T>
inline void median_filter_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T *kernel = (T*)calloc(kernelSize * kernelSize, sizeof(T));
    T *kernelTemp;
    kernelTemp = kernel;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *kernelTemp = *srcPtrWindowTemp;
                srcPtrWindowTemp++;
                kernelTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *kernelTemp = *srcPtrWindowTemp;
                srcPtrWindowTemp += channel;
                kernelTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }

    std::sort(kernel, kernel + (kernelSize * kernelSize));

    *dstPtrPixel = *(kernel + (((kernelSize * kernelSize) - 1) / 2));

    free(kernel);
}

template<typename T>
inline void local_binary_pattern_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u remainingElementsInRow, T* centerPixelPtr,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    int pixel = (int) 0;
    T *srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 0);
        }
        srcPtrWindowTemp++;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 1);
        }
        srcPtrWindowTemp++;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 2);
        }
        srcPtrWindowTemp++;
        srcPtrWindowTemp += remainingElementsInRow;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 7);
        }
        srcPtrWindowTemp += 2;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 3);
        }
        srcPtrWindowTemp++;
        srcPtrWindowTemp += remainingElementsInRow;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 6);
        }
        srcPtrWindowTemp++;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 5);
        }
        srcPtrWindowTemp++;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 4);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 0);
        }
        srcPtrWindowTemp += channel;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 1);
        }
        srcPtrWindowTemp += channel;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 2);
        }
        srcPtrWindowTemp += channel;
        srcPtrWindowTemp += remainingElementsInRow;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 7);
        }
        srcPtrWindowTemp += (2 * channel);

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 3);
        }
        srcPtrWindowTemp += channel;
        srcPtrWindowTemp += remainingElementsInRow;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 6);
        }
        srcPtrWindowTemp += channel;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 5);
        }
        srcPtrWindowTemp += channel;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 4);
        }
    }

    *dstPtrPixel = (T) RPPPIXELCHECK(pixel);
}

template<typename T>
inline void non_max_suppression_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow, T windowCenter,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T pixel;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    pixel = *srcPtrWindowTemp;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp > pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp > pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    if (windowCenter >= pixel)
    {
        *dstPtrPixel = windowCenter;
    }
    else
    {
        *dstPtrPixel = (T) 0;
    }
}

template<typename T>
inline void canny_non_max_suppression_kernel_host(T* dstPtrPixel, T windowCenter, T *position1Ptr, T *position2Ptr)
{
    if ((windowCenter >= *position1Ptr) && (windowCenter >= *position2Ptr))
    {
        *dstPtrPixel = windowCenter;
    }
    else
    {
        *dstPtrPixel = (T) 0;
    }
}

template<typename T>
inline void canny_hysterisis_edge_tracing_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow, T windowCenter, Rpp32u bound,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;

    for (int m = 0; m < kernelSize; m++)
    {
        for (int n = 0; n < kernelSize; n++)
        {
            if (*srcPtrWindowTemp == (T) 255)
            {
                *dstPtrPixel = (T) 255;
            }
            srcPtrWindowTemp++;
        }
        srcPtrWindowTemp += remainingElementsInRow;
    }
    *dstPtrPixel = (T) 0;
}

template<typename T, typename U>
inline void harris_corner_detector_kernel_host(T* srcPtrWindowX, T* srcPtrWindowY, U* dstPtrPixel, RppiSize srcSize,
                                             Rpp32u kernelSize, Rpp32u remainingElementsInRow, Rpp32f kValue, Rpp32f threshold,
                                             RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f pixel;

    T *srcPtrWindowTempX, *srcPtrWindowTempY;
    srcPtrWindowTempX = srcPtrWindowX;
    srcPtrWindowTempY = srcPtrWindowY;

    Rpp32f sumXX = 0, sumYY = 0, sumXY = 0;

    for (int m = 0; m < kernelSize; m++)
    {
        for (int n = 0; n < kernelSize; n++)
        {
            Rpp32f valX = (Rpp32f) *srcPtrWindowTempX;
            Rpp32f valY = (Rpp32f) *srcPtrWindowTempY;
            sumXX += (valX * valX);
            sumYY += (valY * valY);
            sumXY += (valX * valY);

            srcPtrWindowTempX++;
            srcPtrWindowTempY++;
        }
        srcPtrWindowTempX += remainingElementsInRow;
        srcPtrWindowTempY += remainingElementsInRow;
    }
    Rpp32f det = (sumXX * sumYY) - (sumXY * sumXY);
    Rpp32f trace = sumXX + sumYY;
    pixel = (det) - (kValue * trace * trace);

    if (pixel > threshold)
    {
        *dstPtrPixel = (U) pixel;
    }
    else
    {
        *dstPtrPixel = (U) 0;
    }
}

template<typename T>
inline void harris_corner_set_maximum_kernel_host(T* dstPtrWindow, Rpp32u kernelSize, Rpp32u remainingElementsInRow,
                                                  RppiChnFormat chnFormat, Rpp32u channel)
{
    T* dstPtrWindowTemp;
    dstPtrWindowTemp = dstPtrWindow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *dstPtrWindowTemp = (T) 255;
                dstPtrWindowTemp++;
            }
            dstPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *dstPtrWindowTemp = (T) 255;
                dstPtrWindowTemp += channel;
            }
            dstPtrWindowTemp += remainingElementsInRow;
        }
    }
}

template<typename T>
inline void harris_corner_set_minimum_kernel_host(T* dstPtrWindow, Rpp32u kernelSize, Rpp32u remainingElementsInRow,
                                                  RppiChnFormat chnFormat, Rpp32u channel)
{
    T* dstPtrWindowTemp;
    dstPtrWindowTemp = dstPtrWindow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *dstPtrWindowTemp = (T) 0;
                dstPtrWindowTemp++;
            }
            dstPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *dstPtrWindowTemp = (T) 0;
                dstPtrWindowTemp += channel;
            }
            dstPtrWindowTemp += remainingElementsInRow;
        }
    }
}

inline void tensor_index_exchange_kernel_host(Rpp32u *loopCount, Rpp32u *loopCountTransposed, Rpp32u tensorDimension, Rpp32u dimension1, Rpp32u dimension2)
{
    memcpy(loopCountTransposed, loopCount, tensorDimension * sizeof(Rpp32u));

    loopCountTransposed[dimension2] = loopCount[dimension1];
    loopCountTransposed[dimension1] = loopCount[dimension2];
}

template<typename T>
inline void tensor_transpose_iterate_kernel_host(T* srcPtr, T* dstPtr,
                                               Rpp32u tensorDimensionTemp, Rpp32u tensorDimension,
                                               Rpp32u *tensorDimensionValues, Rpp32u *tensorDimensionValuesProduct,
                                               Rpp32u *loopCount, Rpp32u *loopCountTransposed,
                                               Rpp32u dimension1, Rpp32u dimension2)
{
    if (tensorDimensionTemp >= tensorDimension)
    {
        Rpp32u dstPtrLoc = 0;
        for (int i = tensorDimension - 1; i > 0 ; i--)
        {
            dstPtrLoc = dstPtrLoc + (loopCount[i] * tensorDimensionValuesProduct[i - 1]);
        }
        dstPtrLoc += loopCount[0];

        tensor_index_exchange_kernel_host(loopCount, loopCountTransposed, tensorDimension, dimension1, dimension2);

        Rpp32u srcPtrLoc = 0;
        for (int i = tensorDimension - 1; i > 0 ; i--)
        {
            srcPtrLoc = srcPtrLoc + (loopCountTransposed[i] * tensorDimensionValuesProduct[i - 1]);
        }
        srcPtrLoc += loopCountTransposed[0];

        *(dstPtr + dstPtrLoc) = *(srcPtr + srcPtrLoc);
    }
    for (int i = 0; i < *(tensorDimensionValues + tensorDimensionTemp); i++)
    {
        *(loopCount + tensorDimensionTemp) = i;
        tensor_transpose_iterate_kernel_host(srcPtr, dstPtr,
                                             tensorDimensionTemp + 1, tensorDimension,
                                             tensorDimensionValues, tensorDimensionValuesProduct,
                                             loopCount, loopCountTransposed,
                                             dimension1, dimension2);
    }
}

template<typename T>
inline void fast_corner_detector_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                           Rpp32u* bresenhamCirclePositions, T threshold, Rpp32u numOfPixels)
{
    T centerPixel = *(srcPtrWindow + (3 * srcSize.width) + 3);
    T max = (T) (RPPPIXELCHECK((Rpp32s) centerPixel + (Rpp32s) threshold));
    T min = (T) (RPPPIXELCHECK((Rpp32s) centerPixel - (Rpp32s) threshold));

    // Find Bresenham Circle for the pixel

    Rpp32u *bresenhamCirclePositionsTemp;
    bresenhamCirclePositionsTemp = bresenhamCirclePositions;

    T *bresenhamCircle = (T*) calloc(16, sizeof(T));
    T *bresenhamCircleTemp;
    bresenhamCircleTemp = bresenhamCircle;

    T* bresenhamCircleOutput = (T*) calloc(16, sizeof(T));

    for (int i = 0; i < 16; i++)
    {
        *bresenhamCircleTemp = *(srcPtrWindow + *bresenhamCirclePositionsTemp);
        bresenhamCircleTemp++;
        bresenhamCirclePositionsTemp++;
    }

    Rpp32u flag = 0;

    *bresenhamCircleOutput = (T) RPPISLESSER(*bresenhamCircle, min);
    *(bresenhamCircleOutput + 8) = (T) RPPISLESSER(*(bresenhamCircle + 8), min);

    if (*bresenhamCircleOutput == 1)
    {
        *(bresenhamCircleOutput + 4) = (T) RPPISLESSER(*(bresenhamCircle + 4), min);
        *(bresenhamCircleOutput + 12) = (T) RPPISLESSER(*(bresenhamCircle + 12), min);
        if (*(bresenhamCircleOutput + 8) == 1)
        {
            if (*(bresenhamCircleOutput + 4) == 1 || *(bresenhamCircleOutput + 12) == 1)
            {
                flag = 1;
            }
        }
        else if (*(bresenhamCircleOutput + 4) == 1 && *(bresenhamCircleOutput + 12) == 1)
        {
            flag = 1;
        }
    }
    else if (*(bresenhamCircleOutput + 8) == 1)
    {
        *(bresenhamCircleOutput + 4) = (T) RPPISLESSER(*(bresenhamCircle + 4), min);
        *(bresenhamCircleOutput + 12) = (T) RPPISLESSER(*(bresenhamCircle + 12), min);
        if (*(bresenhamCircleOutput + 4) == 1 && *(bresenhamCircleOutput + 12) == 1)
        {
            flag = 1;
        }
    }
    if (flag == 0)
    {
        *bresenhamCircleOutput = (T) RPPISGREATER(*bresenhamCircle, max);
        *(bresenhamCircleOutput + 8) = (T) RPPISGREATER(*(bresenhamCircle + 8), max);

        if (*bresenhamCircleOutput == 1)
        {
            *(bresenhamCircleOutput + 4) = (T) RPPISGREATER(*(bresenhamCircle + 4), max);
            *(bresenhamCircleOutput + 12) = (T) RPPISGREATER(*(bresenhamCircle + 12), max);
            if (*(bresenhamCircleOutput + 8) == 1)
            {
                if (*(bresenhamCircleOutput + 4) == 1 || *(bresenhamCircleOutput + 12) == 1)
                {
                    flag = 2;
                }
            }
            else if (*(bresenhamCircleOutput + 4) == 1 && *(bresenhamCircleOutput + 12) == 1)
            {
                flag = 2;
            }
        }
        else if (*(bresenhamCircleOutput + 8) == 1)
        {
            *(bresenhamCircleOutput + 4) = (T) RPPISGREATER(*(bresenhamCircle + 4), max);
            *(bresenhamCircleOutput + 12) = (T) RPPISGREATER(*(bresenhamCircle + 12), max);
            if (*(bresenhamCircleOutput + 4) == 1 && *(bresenhamCircleOutput + 12) == 1)
            {
                flag = 2;
            }
        }
    }
    if (flag == 0)
    {
        *dstPtrPixel = (T) 0;
    }
    else if (flag == 1)
    {
        *(bresenhamCircleOutput + 1) = (T) RPPISLESSER(*(bresenhamCircle + 1), min);
        *(bresenhamCircleOutput + 2) = (T) RPPISLESSER(*(bresenhamCircle + 2), min);
        *(bresenhamCircleOutput + 3) = (T) RPPISLESSER(*(bresenhamCircle + 3), min);
        *(bresenhamCircleOutput + 5) = (T) RPPISLESSER(*(bresenhamCircle + 5), min);
        *(bresenhamCircleOutput + 6) = (T) RPPISLESSER(*(bresenhamCircle + 6), min);
        *(bresenhamCircleOutput + 7) = (T) RPPISLESSER(*(bresenhamCircle + 7), min);
        *(bresenhamCircleOutput + 9) = (T) RPPISLESSER(*(bresenhamCircle + 9), min);
        *(bresenhamCircleOutput + 10) = (T) RPPISLESSER(*(bresenhamCircle + 10), min);
        *(bresenhamCircleOutput + 11) = (T) RPPISLESSER(*(bresenhamCircle + 11), min);
        *(bresenhamCircleOutput + 13) = (T) RPPISLESSER(*(bresenhamCircle + 13), min);
        *(bresenhamCircleOutput + 14) = (T) RPPISLESSER(*(bresenhamCircle + 14), min);
        *(bresenhamCircleOutput + 15) = (T) RPPISLESSER(*(bresenhamCircle + 15), min);
    }
    else if (flag == 2)
    {
        *(bresenhamCircleOutput + 1) = (T) RPPISGREATER(*(bresenhamCircle + 1), max);
        *(bresenhamCircleOutput + 2) = (T) RPPISGREATER(*(bresenhamCircle + 2), max);
        *(bresenhamCircleOutput + 3) = (T) RPPISGREATER(*(bresenhamCircle + 3), max);
        *(bresenhamCircleOutput + 5) = (T) RPPISGREATER(*(bresenhamCircle + 5), max);
        *(bresenhamCircleOutput + 6) = (T) RPPISGREATER(*(bresenhamCircle + 6), max);
        *(bresenhamCircleOutput + 7) = (T) RPPISGREATER(*(bresenhamCircle + 7), max);
        *(bresenhamCircleOutput + 9) = (T) RPPISGREATER(*(bresenhamCircle + 9), max);
        *(bresenhamCircleOutput + 10) = (T) RPPISGREATER(*(bresenhamCircle + 10), max);
        *(bresenhamCircleOutput + 11) = (T) RPPISGREATER(*(bresenhamCircle + 11), max);
        *(bresenhamCircleOutput + 13) = (T) RPPISGREATER(*(bresenhamCircle + 13), max);
        *(bresenhamCircleOutput + 14) = (T) RPPISGREATER(*(bresenhamCircle + 14), max);
        *(bresenhamCircleOutput + 15) = (T) RPPISGREATER(*(bresenhamCircle + 15), max);
    }

    // Find maximum contiguous pixels in bresenhamCircleOutput with value 1

    Rpp32u count = 0;
    Rpp32u maxLength = 0;

    for (int i = 0; i < 32; i++)
    {
        if (*(bresenhamCircleOutput + (i % 16)) == 0)
        {
            count = 0;
            if (i >= 16)
            {
                break;
            }
        }
        else
        {
            count++;
            maxLength = RPPMAX2(maxLength, count);
        }
    }

    // Corner Classification

    if (maxLength >= numOfPixels)
    {
        *dstPtrPixel = (T) 255;
    }
    else
    {
        *dstPtrPixel = (T) 0;
    }

    free(bresenhamCircle);
    free(bresenhamCircleOutput);
}

template<typename T, typename U>
inline void fast_corner_detector_score_function_kernel_host(T* srcPtrWindow, U* dstPtrPixel, RppiSize srcSize,
                                                          Rpp32u* bresenhamCirclePositions, U centerPixel)
{
    U* bresenhamCircle = (U*) calloc(16, sizeof(U));
    U *bresenhamCircleTemp;
    bresenhamCircleTemp = bresenhamCircle;
    Rpp32u *bresenhamCirclePositionsTemp;
    bresenhamCirclePositionsTemp = bresenhamCirclePositions;

    for (int i = 0; i < 16; i++)
    {
        *bresenhamCircleTemp = (U) *(srcPtrWindow + *bresenhamCirclePositionsTemp);
        bresenhamCircleTemp++;
        bresenhamCirclePositionsTemp++;
    }

    U score = 0;
    bresenhamCircleTemp = bresenhamCircle;
    for (int i = 0; i < 16; i++)
    {
        score += RPPABS(centerPixel - *bresenhamCircleTemp);
        bresenhamCircleTemp++;
    }

    *dstPtrPixel = score;

    free(bresenhamCircle);
}

template<typename T, typename U, typename V>
inline void hog_single_channel_gradient_computations_kernel_host(T* srcPtr, RppiSize srcSize, U* gradientX, U* gradientY, U* gradientMagnitude, V* gradientDirection,
                                                               Rpp32f* gradientKernel, RppiSize rppiGradientKernelSizeX, RppiSize rppiGradientKernelSizeY)
{
    custom_convolve_image_host(srcPtr, srcSize, gradientX, gradientKernel, rppiGradientKernelSizeX, RPPI_CHN_PLANAR, 1);
    custom_convolve_image_host(srcPtr, srcSize, gradientY, gradientKernel, rppiGradientKernelSizeY, RPPI_CHN_PLANAR, 1);
    compute_magnitude_host(gradientX, gradientY, srcSize, gradientMagnitude, RPPI_CHN_PLANAR, 1);
    compute_gradient_direction_host(gradientX, gradientY, srcSize, gradientDirection, RPPI_CHN_PLANAR, 1);
}

template<typename T, typename U, typename V>
inline void hog_three_channel_gradient_computations_kernel_host(T* srcPtr, T* srcPtrSingleChannel, RppiSize srcSize,
                                                              U* gradientX0, U* gradientY0, U* gradientX1, U* gradientY1, U* gradientX2, U* gradientY2,
                                                              U* gradientX, U* gradientY,
                                                              U* gradientMagnitude, V* gradientDirection,
                                                              Rpp32f* gradientKernel, RppiSize rppiGradientKernelSizeX, RppiSize rppiGradientKernelSizeY,
                                                              RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u imageDim = srcSize.height * srcSize.width;

    compute_channel_extract_host(srcPtr, srcSize, srcPtrSingleChannel, 0, chnFormat, channel);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientX0, gradientKernel, rppiGradientKernelSizeX, RPPI_CHN_PLANAR, 1);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientY0, gradientKernel, rppiGradientKernelSizeY, RPPI_CHN_PLANAR, 1);

    compute_channel_extract_host(srcPtr, srcSize, srcPtrSingleChannel, 1, chnFormat, channel);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientX1, gradientKernel, rppiGradientKernelSizeX, RPPI_CHN_PLANAR, 1);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientY1, gradientKernel, rppiGradientKernelSizeY, RPPI_CHN_PLANAR, 1);

    compute_channel_extract_host(srcPtr, srcSize, srcPtrSingleChannel, 2, chnFormat, channel);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientX2, gradientKernel, rppiGradientKernelSizeX, RPPI_CHN_PLANAR, 1);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientY2, gradientKernel, rppiGradientKernelSizeY, RPPI_CHN_PLANAR, 1);

    compute_max_host(gradientX0, gradientX1, srcSize, gradientX, channel);
    memcpy(gradientX0, gradientX, imageDim * sizeof(Rpp32s));
    compute_max_host(gradientX0, gradientX2, srcSize, gradientX, channel);

    compute_max_host(gradientY0, gradientY1, srcSize, gradientY, channel);
    memcpy(gradientY0, gradientY, imageDim * sizeof(Rpp32s));
    compute_max_host(gradientY0, gradientY2, srcSize, gradientY, channel);

    compute_magnitude_host(gradientX, gradientY, srcSize, gradientMagnitude, RPPI_CHN_PLANAR, 1);
    compute_gradient_direction_host(gradientX, gradientY, srcSize, gradientDirection, RPPI_CHN_PLANAR, 1);
}



















// Convolution Functions

template<typename T>
inline void convolve_image_host_batch(T* srcPtrImage, RppiSize srcSize, RppiSize srcSizeMax, T* dstPtrImage,
                                           T* srcPtrBoundedROI, RppiSize srcSizeBoundedROI,
                                           Rpp32f* kernel, RppiSize kernelSize,
                                           Rpp32f x1, Rpp32f y1, Rpp32f x2, Rpp32f y2,
                                           RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u imageDimMax = srcSizeMax.height * srcSizeMax.width;
    Rpp32u imageDimROI = srcSizeBoundedROI.height * srcSizeBoundedROI.width;

    T maxVal = (T)(std::numeric_limits<T>::max());
    T minVal = (T)(std::numeric_limits<T>::min());

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSizeBoundedROI.width - kernelSize.width;

        for(int c = 0; c < channel; c++)
        {
            T *srcPtrBoundedROIChannel, *srcPtrChannel, *dstPtrChannel;
            srcPtrBoundedROIChannel = srcPtrBoundedROI + (c * imageDimROI);
            srcPtrChannel = srcPtrImage + (c * imageDimMax);
            dstPtrChannel = dstPtrImage + (c * imageDimMax);

            Rpp32u roiRowCount = 0;


            for(int i = 0; i < srcSize.height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrChannel + (i * srcSizeMax.width);
                dstPtrTemp = dstPtrChannel + (i * srcSizeMax.width);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, srcSize.width * sizeof(T));

                    dstPtrTemp += srcSizeMax.width;
                    srcPtrTemp += srcSizeMax.width;
                }
                else
                {
                    T *srcPtrWindow;
                    srcPtrWindow = srcPtrBoundedROIChannel + (roiRowCount * srcSizeBoundedROI.width);
                    for(int j = 0; j < srcSize.width; j++)
                    {
                        if((x1 <= j) && (j <= x2 ))
                        {
                            convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                    kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                    chnFormat, channel);

                            srcPtrWindow++;
                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        else
                        {
                            *dstPtrTemp = *srcPtrTemp;

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                    }
                    #pragma omp critical
                    roiRowCount++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = (srcSizeBoundedROI.width - kernelSize.width) * channel;
        Rpp32u elementsInRowBoundedROI = channel * srcSizeBoundedROI.width;
        Rpp32u elementsInRowMax = channel * srcSizeMax.width;
        Rpp32u elementsInRow = channel * srcSize.width;

        Rpp32u roiRowCount = 0;


        for(int i = 0; i < srcSize.height; i++)
        {
            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
            dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

            if (!((y1 <= i) && (i <= y2)))
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                dstPtrTemp += elementsInRowMax;
                srcPtrTemp += elementsInRowMax;
            }
            else
            {
                T *srcPtrWindow;
                srcPtrWindow = srcPtrBoundedROI + (roiRowCount * elementsInRowBoundedROI);
                for(int j = 0; j < srcSize.width; j++)
                {
                    if (!((x1 <= j) && (j <= x2 )))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));

                        dstPtrTemp += channel;
                        srcPtrTemp += channel;
                    }
                    else
                    {
                        for(int c = 0; c < channel; c++)
                        {

                            convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                    kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                    chnFormat, channel);

                            srcPtrWindow++;
                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                    }
                }
                #pragma omp critical
                roiRowCount++;
            }
        }
    }
}

template<typename T, typename U>
inline void convolve_image_host(T* srcPtrMod, RppiSize srcSizeMod, U* dstPtr, RppiSize srcSize,
                        Rpp32f* kernel, RppiSize kernelSize,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrWindow;
    U *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;

    U maxVal = (U)(std::numeric_limits<U>::max());
    U minVal = (U)(std::numeric_limits<U>::min());

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSizeMod.width - kernelSize.width;

        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                 kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += (kernelSize.width - 1);
            }
            srcPtrWindow += ((kernelSize.height - 1) * srcSizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = (srcSizeMod.width - kernelSize.width) * channel;

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                 kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += ((kernelSize.width - 1) * channel);
        }
    }
}

template<typename T>
inline void convolve_subimage_host(T* srcPtrMod, RppiSize srcSizeMod, T* dstPtr, RppiSize srcSizeSubImage, RppiSize srcSize,
                        Rpp32f* kernel, RppiSize kernelSize,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    int widthDiffPlanar = srcSize.width - srcSizeSubImage.width;
    int widthDiffPacked = (srcSize.width - srcSizeSubImage.width) * channel;

    T *srcPtrWindow, *dstPtrTemp;

    T maxVal = (T)(std::numeric_limits<T>::max());
    T minVal = (T)(std::numeric_limits<T>::min());

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSize.width - kernelSize.width;

        for (int c = 0; c < channel; c++)
        {
            srcPtrWindow = srcPtrMod + (c * srcSize.height * srcSize.width);
            dstPtrTemp = dstPtr + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSizeSubImage.height; i++)
            {
                for (int j = 0; j < srcSizeSubImage.width; j++)
                {
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                 kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += widthDiffPlanar;
                dstPtrTemp += widthDiffPlanar;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = (srcSize.width - kernelSize.width) * channel;

        srcPtrWindow = srcPtrMod;
        dstPtrTemp = dstPtr;
        for (int i = 0; i < srcSizeSubImage.height; i++)
        {
            for (int j = 0; j < srcSizeSubImage.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                 kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += widthDiffPacked;
            dstPtrTemp += widthDiffPacked;
        }
    }
}

template <typename T, typename U>
inline RppStatus custom_convolve_image_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                                  Rpp32f *kernel, RppiSize rppiKernelSize,
                                  RppiChnFormat chnFormat, Rpp32u channel)
{
    if (rppiKernelSize.height % 2 == 0 || rppiKernelSize.width % 2 == 0)
    {
        return RPP_ERROR;
    }

    int boundY = ((rppiKernelSize.height - 1) / 2);
    int boundX = ((rppiKernelSize.width - 1) / 2);

    RppiSize srcSizeMod1, srcSizeMod2;

    srcSizeMod1.height = srcSize.height + boundY;
    srcSizeMod1.width = srcSize.width + boundX;
    T *srcPtrMod1 = (T *)calloc(srcSizeMod1.height * srcSizeMod1.width * channel, sizeof(T));
    generate_corner_padded_image_host(srcPtr, srcSize, srcPtrMod1, srcSizeMod1, 1, chnFormat, channel);

    srcSizeMod2.height = srcSizeMod1.height + boundY;
    srcSizeMod2.width = srcSizeMod1.width + boundX;
    T *srcPtrMod2 = (T *)calloc(srcSizeMod2.height * srcSizeMod2.width * channel, sizeof(T));
    generate_corner_padded_image_host(srcPtrMod1, srcSizeMod1, srcPtrMod2, srcSizeMod2, 4, chnFormat, channel);

    convolve_image_host(srcPtrMod2, srcSizeMod2, dstPtr, srcSize, kernel, rppiKernelSize, chnFormat, channel);

    free(srcPtrMod1);
    free(srcPtrMod2);

    return RPP_SUCCESS;
}

// Compute Functions for RPP Tensor API

inline void compute_multiply_16_host(__m256 *p, __m256 *pMulParam)
{
    p[0] = _mm256_mul_ps(p[0], pMulParam[0]);    // multiply adjustment
    p[1] = _mm256_mul_ps(p[1], pMulParam[0]);    // multiply adjustment
}

inline void compute_subtract_16_host(__m256 *p, __m256 *pSubtractParam)
{
    p[0] = _mm256_sub_ps(p[0], pSubtractParam[0]);    // subtract adjustment
    p[1] = _mm256_sub_ps(p[1], pSubtractParam[0]);    // subtract adjustment
}

inline void compute_add_16_host(__m256 *p, __m256 *pAddParam)
{
    p[0] = _mm256_add_ps(p[0], pAddParam[0]);    // add adjustment
    p[1] = _mm256_add_ps(p[1], pAddParam[0]);    // add adjustment
}

inline void compute_rmn_24_host(__m256 *p, __m256 *pRMNParams)
{
    p[0] = _mm256_mul_ps(_mm256_sub_ps(p[0], pRMNParams[0]), pRMNParams[1]);
    p[1] = _mm256_mul_ps(_mm256_sub_ps(p[1], pRMNParams[2]), pRMNParams[3]);
    p[2] = _mm256_mul_ps(_mm256_sub_ps(p[2], pRMNParams[4]), pRMNParams[5]);
}

inline void compute_rmn_8_host(__m256 *p, __m256 *pRMNParams)
{
    p[0] = _mm256_mul_ps(_mm256_sub_ps(p[0], pRMNParams[0]), pRMNParams[1]);
}

inline void compute_color_48_to_greyscale_16_host(__m256 *p, __m256 *pChannelWeights)
{
    p[0] = _mm256_fmadd_ps(p[0], pChannelWeights[0], _mm256_fmadd_ps(p[2], pChannelWeights[1], _mm256_mul_ps(p[4], pChannelWeights[2])));
    p[1] = _mm256_fmadd_ps(p[1], pChannelWeights[0], _mm256_fmadd_ps(p[3], pChannelWeights[1], _mm256_mul_ps(p[5], pChannelWeights[2])));
}

inline void compute_color_48_to_greyscale_16_host(__m128 *p, __m128 *pChannelWeights)
{
    p[0] = _mm_fmadd_ps(p[0], pChannelWeights[0], _mm_fmadd_ps(p[4], pChannelWeights[1], _mm_mul_ps(p[8], pChannelWeights[2])));
    p[1] = _mm_fmadd_ps(p[1], pChannelWeights[0], _mm_fmadd_ps(p[5], pChannelWeights[1], _mm_mul_ps(p[9], pChannelWeights[2])));
    p[2] = _mm_fmadd_ps(p[2], pChannelWeights[0], _mm_fmadd_ps(p[6], pChannelWeights[1], _mm_mul_ps(p[10], pChannelWeights[2])));
    p[3] = _mm_fmadd_ps(p[3], pChannelWeights[0], _mm_fmadd_ps(p[7], pChannelWeights[1], _mm_mul_ps(p[11], pChannelWeights[2])));
}

inline void compute_color_24_to_greyscale_8_host(__m256 *p, __m256 *pChannelWeights)
{
    p[0] = _mm256_fmadd_ps(p[0], pChannelWeights[0], _mm256_fmadd_ps(p[1], pChannelWeights[1], _mm256_mul_ps(p[2], pChannelWeights[2])));
}

inline void compute_color_12_to_greyscale_4_host(__m128 *p, __m128 *pChannelWeights)
{
    p[0] = _mm_fmadd_ps(p[0], pChannelWeights[0], _mm_fmadd_ps(p[1], pChannelWeights[1], _mm_mul_ps(p[2], pChannelWeights[2])));
}

inline void compute_contrast_48_host(__m256 *p, __m256 *pContrastParams)
{
    p[0] = _mm256_fmadd_ps(_mm256_sub_ps(p[0], pContrastParams[1]), pContrastParams[0], pContrastParams[1]);    // contrast adjustment
    p[1] = _mm256_fmadd_ps(_mm256_sub_ps(p[1], pContrastParams[1]), pContrastParams[0], pContrastParams[1]);    // contrast adjustment
    p[2] = _mm256_fmadd_ps(_mm256_sub_ps(p[2], pContrastParams[1]), pContrastParams[0], pContrastParams[1]);    // contrast adjustment
    p[3] = _mm256_fmadd_ps(_mm256_sub_ps(p[3], pContrastParams[1]), pContrastParams[0], pContrastParams[1]);    // contrast adjustment
    p[4] = _mm256_fmadd_ps(_mm256_sub_ps(p[4], pContrastParams[1]), pContrastParams[0], pContrastParams[1]);    // contrast adjustment
    p[5] = _mm256_fmadd_ps(_mm256_sub_ps(p[5], pContrastParams[1]), pContrastParams[0], pContrastParams[1]);    // contrast adjustment
}

inline void compute_contrast_24_host(__m256 *p, __m256 *pContrastParams)
{
    p[0] = _mm256_fmadd_ps(_mm256_sub_ps(p[0], pContrastParams[1]), pContrastParams[0], pContrastParams[1]);    // contrast adjustment
    p[1] = _mm256_fmadd_ps(_mm256_sub_ps(p[1], pContrastParams[1]), pContrastParams[0], pContrastParams[1]);    // contrast adjustment
    p[2] = _mm256_fmadd_ps(_mm256_sub_ps(p[2], pContrastParams[1]), pContrastParams[0], pContrastParams[1]);    // contrast adjustment
}

inline void compute_contrast_16_host(__m256 *p, __m256 *pContrastParams)
{
    p[0] = _mm256_fmadd_ps(_mm256_sub_ps(p[0], pContrastParams[1]), pContrastParams[0], pContrastParams[1]);    // contrast adjustment
    p[1] = _mm256_fmadd_ps(_mm256_sub_ps(p[1], pContrastParams[1]), pContrastParams[0], pContrastParams[1]);    // contrast adjustment
}

inline void compute_contrast_8_host(__m256 *p, __m256 *pContrastParams)
{
    p[0] = _mm256_fmadd_ps(_mm256_sub_ps(p[0], pContrastParams[1]), pContrastParams[0], pContrastParams[1]);    // contrast adjustment
}

inline void compute_brightness_48_host(__m256 *p, __m256 *pBrightnessParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm256_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm256_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[3] = _mm256_fmadd_ps(p[3], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[4] = _mm256_fmadd_ps(p[4], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[5] = _mm256_fmadd_ps(p[5], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_48_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[3] = _mm_fmadd_ps(p[3], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[4] = _mm_fmadd_ps(p[4], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[5] = _mm_fmadd_ps(p[5], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[6] = _mm_fmadd_ps(p[6], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[7] = _mm_fmadd_ps(p[7], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[8] = _mm_fmadd_ps(p[8], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[9] = _mm_fmadd_ps(p[9], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[10] = _mm_fmadd_ps(p[10], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[11] = _mm_fmadd_ps(p[11], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_24_host(__m256 *p, __m256 *pBrightnessParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm256_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm256_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_24_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[3] = _mm_fmadd_ps(p[3], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[4] = _mm_fmadd_ps(p[4], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[5] = _mm_fmadd_ps(p[5], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_16_host(__m256 *p, __m256 *pBrightnessParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm256_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_16_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[3] = _mm_fmadd_ps(p[3], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_12_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_8_host(__m256 *p, __m256 *pBrightnessParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_8_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_brightness_4_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
}

inline void compute_exposure_48_host(__m256 *p, __m256 &pExposureParam)
{
    p[0] = _mm256_mul_ps(p[0], pExposureParam);    // exposure adjustment
    p[1] = _mm256_mul_ps(p[1], pExposureParam);    // exposure adjustment
    p[2] = _mm256_mul_ps(p[2], pExposureParam);    // exposure adjustment
    p[3] = _mm256_mul_ps(p[3], pExposureParam);    // exposure adjustment
    p[4] = _mm256_mul_ps(p[4], pExposureParam);    // exposure adjustment
    p[5] = _mm256_mul_ps(p[5], pExposureParam);    // exposure adjustment
}

inline void compute_exposure_24_host(__m256 *p, __m256 &pExposureParam)
{
    p[0] = _mm256_mul_ps(p[0], pExposureParam);    // exposure adjustment
    p[1] = _mm256_mul_ps(p[1], pExposureParam);    // exposure adjustment
    p[2] = _mm256_mul_ps(p[2], pExposureParam);    // exposure adjustment
}

inline void compute_exposure_16_host(__m256 *p, __m256 &pExposureParam)
{
    p[0] = _mm256_mul_ps(p[0], pExposureParam);    // exposure adjustment
    p[1] = _mm256_mul_ps(p[1], pExposureParam);    // exposure adjustment
}

inline void compute_exposure_8_host(__m256 *p, __m256 &pExposureParam)
{
    p[0] = _mm256_mul_ps(p[0], pExposureParam);    // exposure adjustment
}

inline void compute_spatter_48_host(__m256 *p, __m256 *pSpatterMaskInv, __m256 *pSpatterMask, __m256 *pSpatterValue)
{
    p[0] = _mm256_fmadd_ps(p[0], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm256_fmadd_ps(p[1], pSpatterMaskInv[1], _mm256_mul_ps(pSpatterValue[0], pSpatterMask[1]));    // spatter adjustment
    p[2] = _mm256_fmadd_ps(p[2], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[1], pSpatterMask[0]));    // spatter adjustment
    p[3] = _mm256_fmadd_ps(p[3], pSpatterMaskInv[1], _mm256_mul_ps(pSpatterValue[1], pSpatterMask[1]));    // spatter adjustment
    p[4] = _mm256_fmadd_ps(p[4], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[2], pSpatterMask[0]));    // spatter adjustment
    p[5] = _mm256_fmadd_ps(p[5], pSpatterMaskInv[1], _mm256_mul_ps(pSpatterValue[2], pSpatterMask[1]));    // spatter adjustment
}

inline void compute_spatter_24_host(__m256 *p, __m256 *pSpatterMaskInv, __m256 *pSpatterMask, __m256 *pSpatterValue)
{
    p[0] = _mm256_fmadd_ps(p[0], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm256_fmadd_ps(p[1], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[1], pSpatterMask[0]));    // spatter adjustment
    p[2] = _mm256_fmadd_ps(p[2], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[2], pSpatterMask[0]));    // spatter adjustment
}

inline void compute_spatter_16_host(__m256 *p, __m256 *pSpatterMaskInv, __m256 *pSpatterMask, __m256 pSpatterValue)
{
    p[0] = _mm256_fmadd_ps(p[0], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue, pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm256_fmadd_ps(p[1], pSpatterMaskInv[1], _mm256_mul_ps(pSpatterValue, pSpatterMask[1]));    // spatter adjustment
}

inline void compute_spatter_8_host(__m256 *p, __m256 *pSpatterMaskInv, __m256 *pSpatterMask, __m256 *pSpatterValue)
{
    p[0] = _mm256_fmadd_ps(p[0], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
}

inline void compute_spatter_48_host(__m128 *p, __m128 *pSpatterMaskInv, __m128 *pSpatterMask, __m128 *pSpatterValue)
{
    p[0] = _mm_fmadd_ps(p[0], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm_fmadd_ps(p[1], pSpatterMaskInv[1], _mm_mul_ps(pSpatterValue[0], pSpatterMask[1]));    // spatter adjustment
    p[2] = _mm_fmadd_ps(p[2], pSpatterMaskInv[2], _mm_mul_ps(pSpatterValue[0], pSpatterMask[2]));    // spatter adjustment
    p[3] = _mm_fmadd_ps(p[3], pSpatterMaskInv[3], _mm_mul_ps(pSpatterValue[0], pSpatterMask[3]));    // spatter adjustment
    p[4] = _mm_fmadd_ps(p[4], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[1], pSpatterMask[0]));    // spatter adjustment
    p[5] = _mm_fmadd_ps(p[5], pSpatterMaskInv[1], _mm_mul_ps(pSpatterValue[1], pSpatterMask[1]));    // spatter adjustment
    p[6] = _mm_fmadd_ps(p[6], pSpatterMaskInv[2], _mm_mul_ps(pSpatterValue[1], pSpatterMask[2]));    // spatter adjustment
    p[7] = _mm_fmadd_ps(p[7], pSpatterMaskInv[3], _mm_mul_ps(pSpatterValue[1], pSpatterMask[3]));    // spatter adjustment
    p[8] = _mm_fmadd_ps(p[8], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[2], pSpatterMask[0]));    // spatter adjustment
    p[9] = _mm_fmadd_ps(p[9], pSpatterMaskInv[1], _mm_mul_ps(pSpatterValue[2], pSpatterMask[1]));    // spatter adjustment
    p[10] = _mm_fmadd_ps(p[10], pSpatterMaskInv[2], _mm_mul_ps(pSpatterValue[2], pSpatterMask[2]));    // spatter adjustment
    p[11] = _mm_fmadd_ps(p[11], pSpatterMaskInv[3], _mm_mul_ps(pSpatterValue[2], pSpatterMask[3]));    // spatter adjustment
}

inline void compute_spatter_16_host(__m128 *p, __m128 *pSpatterMaskInv, __m128 *pSpatterMask, __m128 pSpatterValue)
{
    p[0] = _mm_fmadd_ps(p[0], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue, pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm_fmadd_ps(p[1], pSpatterMaskInv[1], _mm_mul_ps(pSpatterValue, pSpatterMask[1]));    // spatter adjustment
    p[2] = _mm_fmadd_ps(p[2], pSpatterMaskInv[2], _mm_mul_ps(pSpatterValue, pSpatterMask[2]));    // spatter adjustment
    p[3] = _mm_fmadd_ps(p[3], pSpatterMaskInv[3], _mm_mul_ps(pSpatterValue, pSpatterMask[3]));    // spatter adjustment
}

inline void compute_spatter_12_host(__m128 *p, __m128 *pSpatterMaskInv, __m128 *pSpatterMask, __m128 *pSpatterValue)
{
    p[0] = _mm_fmadd_ps(p[0], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm_fmadd_ps(p[1], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[1], pSpatterMask[0]));    // spatter adjustment
    p[2] = _mm_fmadd_ps(p[2], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[2], pSpatterMask[0]));    // spatter adjustment
}

inline void compute_spatter_4_host(__m128 *p, __m128 *pSpatterMaskInv, __m128 *pSpatterMask, __m128 *pSpatterValue)
{
    p[0] = _mm_fmadd_ps(p[0], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
}

inline void compute_cmn_48_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pCMNParams[0], pCMNParams[1]);
    p[1] = _mm256_fmadd_ps(p[1], pCMNParams[0], pCMNParams[1]);
    p[2] = _mm256_fmadd_ps(p[2], pCMNParams[2], pCMNParams[3]);
    p[3] = _mm256_fmadd_ps(p[3], pCMNParams[2], pCMNParams[3]);
    p[4] = _mm256_fmadd_ps(p[4], pCMNParams[4], pCMNParams[5]);
    p[5] = _mm256_fmadd_ps(p[5], pCMNParams[4], pCMNParams[5]);
}

inline void compute_cmn_48_rgb_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pCMNParams[0], pCMNParams[1]);
    p[1] = _mm256_fmadd_ps(p[1], pCMNParams[0], pCMNParams[1]);
    p[2] = _mm256_fmadd_ps(p[2], pCMNParams[0], pCMNParams[1]);
    p[3] = _mm256_fmadd_ps(p[3], pCMNParams[0], pCMNParams[1]);
    p[4] = _mm256_fmadd_ps(p[4], pCMNParams[0], pCMNParams[1]);
    p[5] = _mm256_fmadd_ps(p[5], pCMNParams[0], pCMNParams[1]);
    p[6] = _mm256_fmadd_ps(p[6], pCMNParams[0], pCMNParams[1]);
    p[7] = _mm256_fmadd_ps(p[7], pCMNParams[0], pCMNParams[1]);
}

inline void compute_cmn_24_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pCMNParams[0], pCMNParams[1]);
    p[1] = _mm256_fmadd_ps(p[1], pCMNParams[2], pCMNParams[3]);
    p[2] = _mm256_fmadd_ps(p[2], pCMNParams[4], pCMNParams[5]);
}

inline void compute_cmn_16_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pCMNParams[0], pCMNParams[1]);
    p[1] = _mm256_fmadd_ps(p[1], pCMNParams[0], pCMNParams[1]);
}

inline void compute_cmn_8_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pCMNParams[0], pCMNParams[1]);
}

inline void compute_gridmask_masks_16_host(__m128 *pCol, __m128 *pGridRowRatio, __m128 pCosRatio, __m128 pSinRatio, __m128 pGridRatio, __m128 *pMask)
{
    __m128 pCalc[2];

    pCalc[0] = _mm_fmadd_ps(pCol[0], pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol[0], pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask[0] = _mm_or_ps(pCalc[0], pCalc[1]);

    pCalc[0] = _mm_fmadd_ps(pCol[1], pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol[1], pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask[1] = _mm_or_ps(pCalc[0], pCalc[1]);

    pCalc[0] = _mm_fmadd_ps(pCol[2], pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol[2], pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask[2] = _mm_or_ps(pCalc[0], pCalc[1]);

    pCalc[0] = _mm_fmadd_ps(pCol[3], pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol[3], pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask[3] = _mm_or_ps(pCalc[0], pCalc[1]);

    pCol[0] = _mm_add_ps(pCol[0], xmm_p16);
    pCol[1] = _mm_add_ps(pCol[1], xmm_p16);
    pCol[2] = _mm_add_ps(pCol[2], xmm_p16);
    pCol[3] = _mm_add_ps(pCol[3], xmm_p16);
}

inline void compute_gridmask_masks_4_host(__m128 &pCol, __m128 *pGridRowRatio, __m128 pCosRatio, __m128 pSinRatio, __m128 pGridRatio, __m128 &pMask)
{
    __m128 pCalc[2];

    pCalc[0] = _mm_fmadd_ps(pCol, pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol, pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask = _mm_or_ps(pCalc[0], pCalc[1]);
    pCol = _mm_add_ps(pCol, xmm_p4);
}

inline void compute_gridmask_result_48_host(__m128 *p, __m128 *pMask)
{
    p[0] = _mm_and_ps(p[0], pMask[0]);
    p[1] = _mm_and_ps(p[1], pMask[1]);
    p[2] = _mm_and_ps(p[2], pMask[2]);
    p[3] = _mm_and_ps(p[3], pMask[3]);
    p[4] = _mm_and_ps(p[4], pMask[0]);
    p[5] = _mm_and_ps(p[5], pMask[1]);
    p[6] = _mm_and_ps(p[6], pMask[2]);
    p[7] = _mm_and_ps(p[7], pMask[3]);
    p[8] = _mm_and_ps(p[8], pMask[0]);
    p[9] = _mm_and_ps(p[9], pMask[1]);
    p[10] = _mm_and_ps(p[10], pMask[2]);
    p[11] = _mm_and_ps(p[11], pMask[3]);
}

inline void compute_gridmask_result_16_host(__m128 *p, __m128 *pMask)
{
    p[0] = _mm_and_ps(p[0], pMask[0]);
    p[1] = _mm_and_ps(p[1], pMask[1]);
    p[2] = _mm_and_ps(p[2], pMask[2]);
    p[3] = _mm_and_ps(p[3], pMask[3]);
}

inline void compute_gridmask_result_12_host(__m128 *p, __m128 pMask)
{
    p[0] = _mm_and_ps(p[0], pMask);
    p[1] = _mm_and_ps(p[1], pMask);
    p[2] = _mm_and_ps(p[2], pMask);
}

inline void compute_gridmask_result_4_host(__m128 *p, __m128 pMask)
{
    p[0] = _mm_and_ps(p[0], pMask);
}

inline void compute_color_twist_host(RpptFloatRGB *pixel, Rpp32f brightnessParam, Rpp32f contrastParam, Rpp32f hueParam, Rpp32f saturationParam)
{
    // RGB to HSV

    Rpp32f hue, sat, v, add;
    Rpp32f rf, gf, bf, cmax, cmin, delta;
    rf = pixel->R;
    gf = pixel->G;
    bf = pixel->B;
    cmax = RPPMAX3(rf, gf, bf);
    cmin = RPPMIN3(rf, gf, bf);
    delta = cmax - cmin;
    hue = 0.0f;
    sat = 0.0f;
    add = 0.0f;
    if ((delta != 0) && (cmax != 0))
    {
        sat = delta / cmax;
        if (cmax == rf)
        {
            hue = gf - bf;
            add = 0.0f;
        }
        else if (cmax == gf)
        {
            hue = bf - rf;
            add = 2.0f;
        }
        else
        {
            hue = rf - gf;
            add = 4.0f;
        }
        hue /= delta;
    }
    v = cmax;

    // Modify Hue and Saturation

    hue += hueParam + add;
    if (hue >= 6.0f) hue -= 6.0f;
    if (hue < 0) hue += 6.0f;
    sat *= saturationParam;
    sat = std::max(0.0f, std::min(1.0f, sat));

    // HSV to RGB with brightness/contrast adjustment

    Rpp32s hueIntegerPart = (Rpp32s) hue;
    Rpp32f hueFractionPart = hue - hueIntegerPart;
    Rpp32f vsat = v * sat;
    Rpp32f vsatf = vsat * hueFractionPart;
    Rpp32f p = v - vsat;
    Rpp32f q = v - vsatf;
    Rpp32f t = v - vsat + vsatf;
    switch (hueIntegerPart)
    {
        case 0: rf = v; gf = t; bf = p; break;
        case 1: rf = q; gf = v; bf = p; break;
        case 2: rf = p; gf = v; bf = t; break;
        case 3: rf = p; gf = q; bf = v; break;
        case 4: rf = t; gf = p; bf = v; break;
        case 5: rf = v; gf = p; bf = q; break;
    }
    pixel->R = std::fma(rf, brightnessParam, contrastParam);
    pixel->G = std::fma(gf, brightnessParam, contrastParam);
    pixel->B = std::fma(bf, brightnessParam, contrastParam);
}

inline void compute_color_twist_12_host(__m128 &pVecR, __m128 &pVecG, __m128 &pVecB, __m128 *pColorTwistParams)
{
    __m128 pA, pH, pS, pV, pDelta, pAdd, pIntH;
    __m128 pMask[4];
    __m128i pxIntH;

    // RGB to HSV
    pV = _mm_max_ps(pVecR, _mm_max_ps(pVecG, pVecB));                                                               // cmax = RPPMAX3(rf, gf, bf);
    pS = _mm_min_ps(pVecR, _mm_min_ps(pVecG, pVecB));                                                               // cmin = RPPMIN3(rf, gf, bf);
    pDelta = _mm_sub_ps(pV, pS);                                                                                    // delta = cmax - cmin;
    pH = xmm_p0;                                                                                                    // hue = 0.0f;
    pS = xmm_p0;                                                                                                    // sat = 0.0f;
    pAdd = xmm_p0;                                                                                                  // add = 0.0f;
    pMask[0] = _mm_and_ps(_mm_cmpneq_ps(pDelta, xmm_p0), _mm_cmpneq_ps(pV, xmm_p0));                                // if ((delta != 0) && (cmax != 0)) {
    pS = _mm_div_ps(_mm_and_ps(pMask[0], pDelta), pV);                                                              //     sat = delta / cmax;
    pMask[1] = _mm_cmpeq_ps(pV, pVecR);                                                                             //     Temporarily store cmax == rf comparison
    pMask[2] = _mm_and_ps(pMask[0], pMask[1]);                                                                      //     if (cmax == rf)
    pH = _mm_and_ps(pMask[2], _mm_sub_ps(pVecG, pVecB));                                                            //         hue = gf - bf;
    pAdd = _mm_and_ps(pMask[2], xmm_p0);                                                                            //         add = 0.0f;
    pMask[3] = _mm_cmpeq_ps(pV, pVecG);                                                                             //     Temporarily store cmax == gf comparison
    pMask[2] = _mm_andnot_ps(pMask[1], pMask[3]);                                                                   //     else if (cmax == gf)
    pH = _mm_or_ps(_mm_andnot_ps(pMask[2], pH), _mm_and_ps(pMask[2], _mm_sub_ps(pVecB, pVecR)));                    //         hue = bf - rf;
    pAdd = _mm_or_ps(_mm_andnot_ps(pMask[2], pAdd), _mm_and_ps(pMask[2], xmm_p2));                                  //         add = 2.0f;
    pMask[3] = _mm_andnot_ps(pMask[3], _mm_andnot_ps(pMask[1], pMask[0]));                                          //     else
    pH = _mm_or_ps(_mm_andnot_ps(pMask[3], pH), _mm_and_ps(pMask[3], _mm_sub_ps(pVecR, pVecG)));                    //         hue = rf - gf;
    pAdd = _mm_or_ps(_mm_andnot_ps(pMask[3], pAdd), _mm_and_ps(pMask[3], xmm_p4));                                  //         add = 4.0f;
    pH = _mm_or_ps(_mm_andnot_ps(pMask[0], pH), _mm_and_ps(pMask[0], _mm_div_ps(pH, pDelta)));                      //     hue /= delta; }

    // Modify Hue and Saturation
    pH = _mm_add_ps(pH, _mm_add_ps(pColorTwistParams[2], pAdd));                                                    // hue += hueParam + add;
    pH = _mm_sub_ps(pH, _mm_and_ps(_mm_cmpge_ps(pH, xmm_p6), xmm_p6));                                              // if (hue >= 6.0f) hue -= 6.0f;
    pH = _mm_add_ps(pH, _mm_and_ps(_mm_cmplt_ps(pH, xmm_p0), xmm_p6));                                              // if (hue < 0) hue += 6.0f;
    pS = _mm_mul_ps(pS, pColorTwistParams[3]);                                                                      // sat *= saturationParam;
    pS = _mm_max_ps(xmm_p0, _mm_min_ps(xmm_p1, pS));                                                                // sat = std::max(0.0f, std::min(1.0f, sat));

    // HSV to RGB with brightness/contrast adjustment
    pIntH = _mm_floor_ps(pH);                                                                                       // Rpp32s hueIntegerPart = (Rpp32s) hue;
    pxIntH = _mm_cvtps_epi32(pIntH);                                                                                // Convert to epi32
    pH = _mm_sub_ps(pH, pIntH);                                                                                     // Rpp32f hueFractionPart = hue - hueIntegerPart;
    pS = _mm_mul_ps(pV, pS);                                                                                        // Rpp32f vsat = v * sat;
    pAdd = _mm_mul_ps(pS, pH);                                                                                      // Rpp32f vsatf = vsat * hueFractionPart;
    pA = _mm_sub_ps(pV, pS);                                                                                        // Rpp32f p = v - vsat;
    pH = _mm_sub_ps(pV, pAdd);                                                                                      // Rpp32f q = v - vsatf;
    pS = _mm_add_ps(pA, pAdd);                                                                                      // Rpp32f t = v - vsat + vsatf;
    pVecR = xmm_p0;                                                                                                 // Reset dstPtrR
    pVecG = xmm_p0;                                                                                                 // Reset dstPtrG
    pVecB = xmm_p0;                                                                                                 // Reset dstPtrB
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px0));                                                  // switch (hueIntegerPart) {case 0:
    pVecR = _mm_and_ps(pMask[0], pV);                                                                               //     rf = v;
    pVecG = _mm_and_ps(pMask[0], pS);                                                                               //     gf = t;
    pVecB = _mm_and_ps(pMask[0], pA);                                                                               //     bf = p; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px1));                                                  // case 1:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pH));                                    //     rf = q;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pV));                                    //     gf = v;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pA));                                    //     bf = p; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px2));                                                  // case 2:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pA));                                    //     rf = p;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pV));                                    //     gf = v;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pS));                                    //     bf = t; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px3));                                                  // case 3:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pA));                                    //     rf = p;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pH));                                    //     gf = q;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pV));                                    //     bf = v; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px4));                                                  // case 4:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pS));                                    //     rf = t;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pA));                                    //     gf = p;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pV));                                    //     bf = v; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px5));                                                  // case 5:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pV));                                    //     rf = v;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pA));                                    //     gf = p;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pH));                                    //     bf = q; break;}
    pVecR = _mm_fmadd_ps(pVecR, pColorTwistParams[0], pColorTwistParams[1]);                                        // dstPtrR = rf * brightnessParam + contrastParam;
    pVecG = _mm_fmadd_ps(pVecG, pColorTwistParams[0], pColorTwistParams[1]);                                        // dstPtrG = gf * brightnessParam + contrastParam;
    pVecB = _mm_fmadd_ps(pVecB, pColorTwistParams[0], pColorTwistParams[1]);                                        // dstPtrB = bf * brightnessParam + contrastParam;
}

inline void compute_color_twist_24_host(__m256 &pVecR, __m256 &pVecG, __m256 &pVecB, __m256 *pColorTwistParams)
{
    __m256 pA, pH, pS, pV, pDelta, pAdd, pIntH;
    __m256 pMask[4];
    __m256i pxIntH;

    // RGB to HSV
    pV = _mm256_max_ps(pVecR, _mm256_max_ps(pVecG, pVecB));                                                            // cmax = RPPMAX3(rf, gf, bf);
    pS = _mm256_min_ps(pVecR, _mm256_min_ps(pVecG, pVecB));                                                            // cmin = RPPMIN3(rf, gf, bf);
    pDelta = _mm256_sub_ps(pV, pS);                                                                                    // delta = cmax - cmin;
    pH = avx_p0;                                                                                                       // hue = 0.0f;
    pS = avx_p0;                                                                                                       // sat = 0.0f;
    pAdd = avx_p0;                                                                                                     // add = 0.0f;
    pMask[0] = _mm256_and_ps(_mm256_cmp_ps(pDelta, avx_p0, _CMP_NEQ_OQ), _mm256_cmp_ps(pV, avx_p0, _CMP_NEQ_OQ));      // if ((delta != 0) && (cmax != 0)) {
    pS = _mm256_div_ps(_mm256_and_ps(pMask[0], pDelta), pV);                                                           //     sat = delta / cmax;
    pMask[1] = _mm256_cmp_ps(pV, pVecR, _CMP_EQ_OQ);                                                                   //     Temporarily store cmax == rf comparison
    pMask[2] = _mm256_and_ps(pMask[0], pMask[1]);                                                                      //     if (cmax == rf)
    pH = _mm256_and_ps(pMask[2], _mm256_sub_ps(pVecG, pVecB));                                                         //         hue = gf - bf;
    pAdd = _mm256_and_ps(pMask[2], avx_p0);                                                                            //         add = 0.0f;
    pMask[3] = _mm256_cmp_ps(pV, pVecG, _CMP_EQ_OQ);                                                                   //     Temporarily store cmax == gf comparison
    pMask[2] = _mm256_andnot_ps(pMask[1], pMask[3]);                                                                   //     else if (cmax == gf)
    pH = _mm256_or_ps(_mm256_andnot_ps(pMask[2], pH), _mm256_and_ps(pMask[2], _mm256_sub_ps(pVecB, pVecR)));           //         hue = bf - rf;
    pAdd = _mm256_or_ps(_mm256_andnot_ps(pMask[2], pAdd), _mm256_and_ps(pMask[2], avx_p2));                            //         add = 2.0f;
    pMask[3] = _mm256_andnot_ps(pMask[3], _mm256_andnot_ps(pMask[1], pMask[0]));                                       //     else
    pH = _mm256_or_ps(_mm256_andnot_ps(pMask[3], pH), _mm256_and_ps(pMask[3], _mm256_sub_ps(pVecR, pVecG)));           //         hue = rf - gf;
    pAdd = _mm256_or_ps(_mm256_andnot_ps(pMask[3], pAdd), _mm256_and_ps(pMask[3], avx_p4));                            //         add = 4.0f;
    pH = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pH), _mm256_and_ps(pMask[0], _mm256_div_ps(pH, pDelta)));             //     hue /= delta; }

    // Modify Hue and Saturation
    pH = _mm256_add_ps(pH, _mm256_add_ps(pColorTwistParams[2], pAdd));                                                 // hue += hueParam + add;
    pH = _mm256_sub_ps(pH, _mm256_and_ps(_mm256_cmp_ps(pH, avx_p6, _CMP_GE_OQ), avx_p6));                              // if (hue >= 6.0f) hue -= 6.0f;
    pH = _mm256_add_ps(pH, _mm256_and_ps(_mm256_cmp_ps(pH, avx_p0, _CMP_LT_OQ), avx_p6));                              // if (hue < 0) hue += 6.0f;
    pS = _mm256_mul_ps(pS, pColorTwistParams[3]);                                                                      // sat *= saturationParam;
    pS = _mm256_max_ps(avx_p0, _mm256_min_ps(avx_p1, pS));                                                             // sat = std::max(0.0f, std::min(1.0f, sat));

    // HSV to RGB with brightness/contrast adjustment
    pIntH = _mm256_floor_ps(pH);                                                                                       // Rpp32s hueIntegerPart = (Rpp32s) hue;
    pxIntH = _mm256_cvtps_epi32(pIntH);                                                                                // Convert to epi32
    pH = _mm256_sub_ps(pH, pIntH);                                                                                     // Rpp32f hueFractionPart = hue - hueIntegerPart;
    pS = _mm256_mul_ps(pV, pS);                                                                                        // Rpp32f vsat = v * sat;
    pAdd = _mm256_mul_ps(pS, pH);                                                                                      // Rpp32f vsatf = vsat * hueFractionPart;
    pA = _mm256_sub_ps(pV, pS);                                                                                        // Rpp32f p = v - vsat;
    pH = _mm256_sub_ps(pV, pAdd);                                                                                      // Rpp32f q = v - vsatf;
    pS = _mm256_add_ps(pA, pAdd);                                                                                      // Rpp32f t = v - vsat + vsatf;
    pVecR = avx_p0;                                                                                                    // Reset dstPtrR
    pVecG = avx_p0;                                                                                                    // Reset dstPtrG
    pVecB = avx_p0;                                                                                                    // Reset dstPtrB
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px0));                                               // switch (hueIntegerPart) {case 0:
    pVecR = _mm256_and_ps(pMask[0], pV);                                                                               //     rf = v;
    pVecG = _mm256_and_ps(pMask[0], pS);                                                                               //     gf = t;
    pVecB = _mm256_and_ps(pMask[0], pA);                                                                               //     bf = p; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px1));                                               // case 1:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pH));                              //     rf = q;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pV));                              //     gf = v;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pA));                              //     bf = p; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px2));                                               // case 2:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pA));                              //     rf = p;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pV));                              //     gf = v;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pS));                              //     bf = t; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px3));                                               // case 3:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pA));                              //     rf = p;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pH));                              //     gf = q;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pV));                              //     bf = v; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px4));                                               // case 4:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pS));                              //     rf = t;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pA));                              //     gf = p;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pV));                              //     bf = v; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px5));                                               // case 5:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pV));                              //     rf = v;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pA));                              //     gf = p;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pH));                              //     bf = q; break;}
    pVecR = _mm256_fmadd_ps(pVecR, pColorTwistParams[0], pColorTwistParams[1]);                                        // dstPtrR = rf * brightnessParam + contrastParam;
    pVecG = _mm256_fmadd_ps(pVecG, pColorTwistParams[0], pColorTwistParams[1]);                                        // dstPtrG = gf * brightnessParam + contrastParam;
    pVecB = _mm256_fmadd_ps(pVecB, pColorTwistParams[0], pColorTwistParams[1]);                                        // dstPtrB = bf * brightnessParam + contrastParam;
}

inline void compute_color_cast_48_host(__m128 *p, __m128 pMul, __m128 *pAdd)
{
    p[0] = _mm_fmadd_ps(_mm_sub_ps(p[0], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[1] = _mm_fmadd_ps(_mm_sub_ps(p[1], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[2] = _mm_fmadd_ps(_mm_sub_ps(p[2], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[3] = _mm_fmadd_ps(_mm_sub_ps(p[3], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[4] = _mm_fmadd_ps(_mm_sub_ps(p[4], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Gs
    p[5] = _mm_fmadd_ps(_mm_sub_ps(p[5], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Gs
    p[6] = _mm_fmadd_ps(_mm_sub_ps(p[6], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Gs
    p[7] = _mm_fmadd_ps(_mm_sub_ps(p[7], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Gs
    p[8] = _mm_fmadd_ps(_mm_sub_ps(p[8], pAdd[2]), pMul, pAdd[2]);    // color_cast adjustment Bs
    p[9] = _mm_fmadd_ps(_mm_sub_ps(p[9], pAdd[2]), pMul, pAdd[2]);    // color_cast adjustment Bs
    p[10] = _mm_fmadd_ps(_mm_sub_ps(p[10], pAdd[2]), pMul, pAdd[2]);    // color_cast adjustment Bs
    p[11] = _mm_fmadd_ps(_mm_sub_ps(p[11], pAdd[2]), pMul, pAdd[2]);    // color_cast adjustment Bs
}

inline void compute_color_cast_12_host(__m128 *p, __m128 pMul, __m128 *pAdd)
{
    p[0] = _mm_fmadd_ps(_mm_sub_ps(p[0], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[1] = _mm_fmadd_ps(_mm_sub_ps(p[1], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Rs
    p[2] = _mm_fmadd_ps(_mm_sub_ps(p[2], pAdd[2]), pMul, pAdd[2]);    // color_cast adjustment Rs
}

inline void compute_color_temperature_48_host(__m256 *p, __m256 pAdj)
{
    p[0] = _mm256_add_ps(p[0], pAdj);    // color_temperature adjustment Rs
    p[1] = _mm256_add_ps(p[1], pAdj);    // color_temperature adjustment Rs
    // no color_temperature adjustment Gs
    p[4] = _mm256_sub_ps(p[4], pAdj);    // color_temperature adjustment Bs
    p[5] = _mm256_sub_ps(p[5], pAdj);    // color_temperature adjustment Bs
}

inline void compute_color_temperature_24_host(__m256 *p, __m256 pAdj)
{
    p[0] = _mm256_add_ps(p[0], pAdj);    // color_temperature adjustment Rs
    // no color_temperature adjustment Gs
    p[2] = _mm256_sub_ps(p[2], pAdj);    // color_temperature adjustment Bs
}

inline void compute_xywh_from_ltrb_host(RpptROIPtr roiPtrInput, RpptROIPtr roiPtrImage)
{
    roiPtrImage->xywhROI.xy.x = roiPtrInput->ltrbROI.lt.x;
    roiPtrImage->xywhROI.xy.y = roiPtrInput->ltrbROI.lt.y;
    roiPtrImage->xywhROI.roiWidth = roiPtrInput->ltrbROI.rb.x - roiPtrInput->ltrbROI.lt.x + 1;
    roiPtrImage->xywhROI.roiHeight = roiPtrInput->ltrbROI.rb.y - roiPtrInput->ltrbROI.lt.y + 1;
}

inline void compute_xyzwhd_from_ltfrbb_host(RpptROI3DPtr roiPtrInput, RpptROI3DPtr roiPtrImage)
{
    roiPtrImage->xyzwhdROI.xyz.x = roiPtrInput->ltfrbbROI.ltf.x;
    roiPtrImage->xyzwhdROI.xyz.y = roiPtrInput->ltfrbbROI.ltf.y;
    roiPtrImage->xyzwhdROI.xyz.z = roiPtrInput->ltfrbbROI.ltf.z;
    roiPtrImage->xyzwhdROI.roiWidth = roiPtrInput->ltfrbbROI.rbb.x - roiPtrInput->ltfrbbROI.ltf.x + 1;
    roiPtrImage->xyzwhdROI.roiHeight = roiPtrInput->ltfrbbROI.rbb.y - roiPtrInput->ltfrbbROI.ltf.y + 1;
    roiPtrImage->xyzwhdROI.roiDepth = roiPtrInput->ltfrbbROI.rbb.z - roiPtrInput->ltfrbbROI.ltf.z + 1;
}

inline void compute_ltrb_from_xywh_host(RpptROIPtr roiPtrInput, RpptROIPtr roiPtrImage)
{
    roiPtrImage->ltrbROI.lt.x = roiPtrInput->xywhROI.xy.x;
    roiPtrImage->ltrbROI.lt.y = roiPtrInput->xywhROI.xy.y;
    roiPtrImage->ltrbROI.rb.x = roiPtrInput->xywhROI.xy.x + roiPtrInput->xywhROI.roiWidth - 1;
    roiPtrImage->ltrbROI.rb.y = roiPtrInput->xywhROI.xy.y + roiPtrInput->xywhROI.roiHeight - 1;
}

inline void compute_roi_boundary_check_host(RpptROIPtr roiPtrImage, RpptROIPtr roiPtr, RpptROIPtr roiPtrDefault)
{
    roiPtr->xywhROI.xy.x = std::max(roiPtrDefault->xywhROI.xy.x, roiPtrImage->xywhROI.xy.x);
    roiPtr->xywhROI.xy.y = std::max(roiPtrDefault->xywhROI.xy.y, roiPtrImage->xywhROI.xy.y);
    roiPtr->xywhROI.roiWidth = std::min(roiPtrDefault->xywhROI.roiWidth - roiPtrImage->xywhROI.xy.x, roiPtrImage->xywhROI.roiWidth);
    roiPtr->xywhROI.roiHeight = std::min(roiPtrDefault->xywhROI.roiHeight - roiPtrImage->xywhROI.xy.y, roiPtrImage->xywhROI.roiHeight);
}

inline void compute_roi3D_boundary_check_host(RpptROI3DPtr roiPtrImage, RpptROI3DPtr roiPtr, RpptROI3DPtr roiPtrDefault)
{
    roiPtr->xyzwhdROI.xyz.x = std::max(roiPtrDefault->xyzwhdROI.xyz.x, roiPtrImage->xyzwhdROI.xyz.x);
    roiPtr->xyzwhdROI.xyz.y = std::max(roiPtrDefault->xyzwhdROI.xyz.y, roiPtrImage->xyzwhdROI.xyz.y);
    roiPtr->xyzwhdROI.xyz.z = std::max(roiPtrDefault->xyzwhdROI.xyz.z, roiPtrImage->xyzwhdROI.xyz.z);
    roiPtr->xyzwhdROI.roiWidth = std::min(roiPtrDefault->xyzwhdROI.roiWidth - roiPtrImage->xyzwhdROI.xyz.x, roiPtrImage->xyzwhdROI.roiWidth);
    roiPtr->xyzwhdROI.roiHeight = std::min(roiPtrDefault->xyzwhdROI.roiHeight - roiPtrImage->xyzwhdROI.xyz.y, roiPtrImage->xyzwhdROI.roiHeight);
    roiPtr->xyzwhdROI.roiDepth = std::min(roiPtrDefault->xyzwhdROI.roiDepth - roiPtrImage->xyzwhdROI.xyz.z, roiPtrImage->xyzwhdROI.roiDepth);
}

inline void compute_roi_validation_host(RpptROIPtr roiPtrInput, RpptROIPtr roiPtr, RpptROIPtr roiPtrDefault, RpptRoiType roiType)
{
    if (roiPtrInput == NULL)
    {
        roiPtr = roiPtrDefault;
    }
    else
    {
        RpptROI roiImage;
        RpptROIPtr roiPtrImage = &roiImage;
        if (roiType == RpptRoiType::LTRB)
            compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
        else if (roiType == RpptRoiType::XYWH)
            roiPtrImage = roiPtrInput;
        compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
    }
}

inline void compute_roi3D_validation_host(RpptROI3DPtr roiPtrInput, RpptROI3DPtr roiPtr, RpptROI3DPtr roiPtrDefault, RpptRoi3DType roiType)
{
    if (roiPtrInput == NULL)
    {
        roiPtr = roiPtrDefault;
    }
    else
    {
        RpptROI3D roiImage;
        RpptROI3DPtr roiPtrImage = &roiImage;
        if (roiType == RpptRoi3DType::LTFRBB)
            compute_xyzwhd_from_ltfrbb_host(roiPtrInput, roiPtrImage);
        else if (roiType == RpptRoi3DType::XYZWHD)
            roiPtrImage = roiPtrInput;
        compute_roi3D_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
    }
}

inline void compute_color_jitter_ctm_host(Rpp32f brightnessParam, Rpp32f contrastParam, Rpp32f hueParam, Rpp32f saturationParam, Rpp32f *ctm)
{
    contrastParam += 1.0f;

    alignas(64) Rpp32f hue_saturation_matrix[16] = {0.299f, 0.299f, 0.299f, 0.0f, 0.587f, 0.587f, 0.587f, 0.0f, 0.114f, 0.114f, 0.114f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    alignas(64) Rpp32f brightness_contrast_matrix[16] = {contrastParam, 0.0f, 0.0f, 0.0f, 0.0f, contrastParam, 0.0f, 0.0f, 0.0f, 0.0f, contrastParam, 0.0f, brightnessParam, brightnessParam, brightnessParam, 1.0f};

    Rpp32f sch = saturationParam * cos(hueParam * PI_OVER_180);
    Rpp32f ssh = saturationParam * sin(hueParam * PI_OVER_180);

    __m128 psch = _mm_set1_ps(sch);
    __m128 pssh = _mm_set1_ps(ssh);
    __m128 p0, p1, p2;

    for (int i = 0; i < 16; i+=4)
    {
        p0 = _mm_loadu_ps(hue_saturation_matrix + i);
        p1 = _mm_loadu_ps(sch_mat + i);
        p2 = _mm_loadu_ps(ssh_mat + i);
        p0 = _mm_fmadd_ps(psch, p1, _mm_fmadd_ps(pssh, p2, p0));
        _mm_storeu_ps(hue_saturation_matrix + i, p0);
    }

    fast_matmul4x4_sse(hue_saturation_matrix, brightness_contrast_matrix, ctm);
}

inline void compute_color_jitter_48_host(__m128 *p, __m128 *pCtm)
{
    __m128 pResult[3];

    pResult[0] = _mm_round_ps(_mm_fmadd_ps(p[0], pCtm[0], _mm_fmadd_ps(p[4], pCtm[1], _mm_fmadd_ps(p[8], pCtm[2], pCtm[3]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment R0-R3
    pResult[1] = _mm_round_ps(_mm_fmadd_ps(p[0], pCtm[4], _mm_fmadd_ps(p[4], pCtm[5], _mm_fmadd_ps(p[8], pCtm[6], pCtm[7]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment G0-G3
    pResult[2] = _mm_round_ps(_mm_fmadd_ps(p[0], pCtm[8], _mm_fmadd_ps(p[4], pCtm[9], _mm_fmadd_ps(p[8], pCtm[10], pCtm[11]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment B0-B3
    p[0] = pResult[0];    // color_jitter adjustment R0-R3
    p[4] = pResult[1];    // color_jitter adjustment G0-G3
    p[8] = pResult[2];    // color_jitter adjustment B0-B3
    pResult[0] = _mm_round_ps(_mm_fmadd_ps(p[1], pCtm[0], _mm_fmadd_ps(p[5], pCtm[1], _mm_fmadd_ps(p[9], pCtm[2], pCtm[3]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment R4-R7
    pResult[1] = _mm_round_ps(_mm_fmadd_ps(p[1], pCtm[4], _mm_fmadd_ps(p[5], pCtm[5], _mm_fmadd_ps(p[9], pCtm[6], pCtm[7]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment G4-G7
    pResult[2] = _mm_round_ps(_mm_fmadd_ps(p[1], pCtm[8], _mm_fmadd_ps(p[5], pCtm[9], _mm_fmadd_ps(p[9], pCtm[10], pCtm[11]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment B4-B7
    p[1] = pResult[0];    // color_jitter adjustment R4-R7
    p[5] = pResult[1];    // color_jitter adjustment G4-G7
    p[9] = pResult[2];    // color_jitter adjustment B4-B7
    pResult[0] = _mm_round_ps(_mm_fmadd_ps(p[2], pCtm[0], _mm_fmadd_ps(p[6], pCtm[1], _mm_fmadd_ps(p[10], pCtm[2], pCtm[3]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment R8-R11
    pResult[1] = _mm_round_ps(_mm_fmadd_ps(p[2], pCtm[4], _mm_fmadd_ps(p[6], pCtm[5], _mm_fmadd_ps(p[10], pCtm[6], pCtm[7]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment G8-G11
    pResult[2] = _mm_round_ps(_mm_fmadd_ps(p[2], pCtm[8], _mm_fmadd_ps(p[6], pCtm[9], _mm_fmadd_ps(p[10], pCtm[10], pCtm[11]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment B8-B11
    p[2] = pResult[0];    // color_jitter adjustment R8-R11
    p[6] = pResult[1];    // color_jitter adjustment G8-G11
    p[10] = pResult[2];    // color_jitter adjustment B8-B11
    pResult[0] = _mm_round_ps(_mm_fmadd_ps(p[3], pCtm[0], _mm_fmadd_ps(p[7], pCtm[1], _mm_fmadd_ps(p[11], pCtm[2], pCtm[3]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment R12-R15
    pResult[1] = _mm_round_ps(_mm_fmadd_ps(p[3], pCtm[4], _mm_fmadd_ps(p[7], pCtm[5], _mm_fmadd_ps(p[11], pCtm[6], pCtm[7]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment G12-G15
    pResult[2] = _mm_round_ps(_mm_fmadd_ps(p[3], pCtm[8], _mm_fmadd_ps(p[7], pCtm[9], _mm_fmadd_ps(p[11], pCtm[10], pCtm[11]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment B12-B15
    p[3] = pResult[0];    // color_jitter adjustment R12-R15
    p[7] = pResult[1];    // color_jitter adjustment G12-G15
    p[11] = pResult[2];    // color_jitter adjustment B12-B15
}

inline void compute_color_jitter_12_host(__m128 *p, __m128 *pCtm)
{
    __m128 pResult[3];

    pResult[0] = _mm_fmadd_ps(p[0], pCtm[0], _mm_fmadd_ps(p[1], pCtm[1], _mm_fmadd_ps(p[2], pCtm[2], pCtm[3])));    // color_jitter adjustment R0-R3
    pResult[1] = _mm_fmadd_ps(p[0], pCtm[4], _mm_fmadd_ps(p[1], pCtm[5], _mm_fmadd_ps(p[2], pCtm[6], pCtm[7])));    // color_jitter adjustment G0-G3
    pResult[2] = _mm_fmadd_ps(p[0], pCtm[8], _mm_fmadd_ps(p[1], pCtm[9], _mm_fmadd_ps(p[2], pCtm[10], pCtm[11])));    // color_jitter adjustment B0-B3
    p[0] = pResult[0];    // color_jitter adjustment R0-R3
    p[1] = pResult[1];    // color_jitter adjustment G0-G3
    p[2] = pResult[2];    // color_jitter adjustment B0-B3
}

inline void compute_salt_and_pepper_noise_8_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pSaltAndPepperNoiseParams)
{
    __m256 pMask[3];
    __m256 pRandomNumbers = rpp_host_rng_xorwow_8_f32_avx(pxXorwowStateX, pxXorwowStateCounter);
    pMask[0] = _mm256_cmp_ps(pRandomNumbers, pSaltAndPepperNoiseParams[0], _CMP_GT_OQ);
    pMask[1] = _mm256_andnot_ps(pMask[0], _mm256_cmp_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1], _CMP_LE_OQ));
    pMask[2] = _mm256_andnot_ps(pMask[0], _mm256_cmp_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1], _CMP_GT_OQ));
    p[0] = _mm256_and_ps(pMask[0], p[0]);
    p[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[1], p[0]), _mm256_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    p[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[2], p[0]), _mm256_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
}

inline void compute_salt_and_pepper_noise_4_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pSaltAndPepperNoiseParams)
{
    __m128 pMask[3];
    __m128 pRandomNumbers = rpp_host_rng_xorwow_4_f32_sse(pxXorwowStateX, pxXorwowStateCounter);
    pMask[0] = _mm_cmpgt_ps(pRandomNumbers, pSaltAndPepperNoiseParams[0]);
    pMask[1] = _mm_andnot_ps(pMask[0], _mm_cmple_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1]));
    pMask[2] = _mm_andnot_ps(pMask[0], _mm_cmpgt_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1]));
    p[0] = _mm_and_ps(pMask[0], p[0]);
    p[0] = _mm_or_ps(_mm_andnot_ps(pMask[1], p[0]), _mm_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    p[0] = _mm_or_ps(_mm_andnot_ps(pMask[2], p[0]), _mm_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
}

inline void compute_salt_and_pepper_noise_16_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pSaltAndPepperNoiseParams)
{
    compute_salt_and_pepper_noise_8_host(p    , pxXorwowStateX, pxXorwowStateCounter, pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_8_host(p + 1, pxXorwowStateX, pxXorwowStateCounter, pSaltAndPepperNoiseParams);
}

inline void compute_salt_and_pepper_noise_16_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pSaltAndPepperNoiseParams)
{
    compute_salt_and_pepper_noise_4_host(p    , pxXorwowStateX, pxXorwowStateCounter, pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_4_host(p + 1, pxXorwowStateX, pxXorwowStateCounter, pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_4_host(p + 2, pxXorwowStateX, pxXorwowStateCounter, pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_4_host(p + 3, pxXorwowStateX, pxXorwowStateCounter, pSaltAndPepperNoiseParams);
}

inline void compute_salt_and_pepper_noise_24_host(__m256 *pR, __m256 *pG, __m256 *pB, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pSaltAndPepperNoiseParams)
{
    __m256 pMask[3];
    __m256 pRandomNumbers = rpp_host_rng_xorwow_8_f32_avx(pxXorwowStateX, pxXorwowStateCounter);
    pMask[0] = _mm256_cmp_ps(pRandomNumbers, pSaltAndPepperNoiseParams[0], _CMP_GT_OQ);
    pMask[1] = _mm256_andnot_ps(pMask[0], _mm256_cmp_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1], _CMP_LE_OQ));
    pMask[2] = _mm256_andnot_ps(pMask[0], _mm256_cmp_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1], _CMP_GT_OQ));
    pR[0] = _mm256_and_ps(pMask[0], pR[0]);
    pR[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[1], pR[0]), _mm256_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    pR[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[2], pR[0]), _mm256_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
    pG[0] = _mm256_and_ps(pMask[0], pG[0]);
    pG[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[1], pG[0]), _mm256_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    pG[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[2], pG[0]), _mm256_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
    pB[0] = _mm256_and_ps(pMask[0], pB[0]);
    pB[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[1], pB[0]), _mm256_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    pB[0] = _mm256_or_ps(_mm256_andnot_ps(pMask[2], pB[0]), _mm256_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
}

inline void compute_salt_and_pepper_noise_12_host(__m128 *pR, __m128 *pG, __m128 *pB, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pSaltAndPepperNoiseParams)
{
    __m128 pMask[3];
    __m128 pRandomNumbers = rpp_host_rng_xorwow_4_f32_sse(pxXorwowStateX, pxXorwowStateCounter);
    pMask[0] = _mm_cmpgt_ps(pRandomNumbers, pSaltAndPepperNoiseParams[0]);
    pMask[1] = _mm_andnot_ps(pMask[0], _mm_cmple_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1]));
    pMask[2] = _mm_andnot_ps(pMask[0], _mm_cmpgt_ps(pRandomNumbers, pSaltAndPepperNoiseParams[1]));
    pR[0] = _mm_and_ps(pMask[0], pR[0]);
    pR[0] = _mm_or_ps(_mm_andnot_ps(pMask[1], pR[0]), _mm_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    pR[0] = _mm_or_ps(_mm_andnot_ps(pMask[2], pR[0]), _mm_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
    pG[0] = _mm_and_ps(pMask[0], pG[0]);
    pG[0] = _mm_or_ps(_mm_andnot_ps(pMask[1], pG[0]), _mm_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    pG[0] = _mm_or_ps(_mm_andnot_ps(pMask[2], pG[0]), _mm_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
    pB[0] = _mm_and_ps(pMask[0], pB[0]);
    pB[0] = _mm_or_ps(_mm_andnot_ps(pMask[1], pB[0]), _mm_and_ps(pMask[1], pSaltAndPepperNoiseParams[2]));
    pB[0] = _mm_or_ps(_mm_andnot_ps(pMask[2], pB[0]), _mm_and_ps(pMask[2], pSaltAndPepperNoiseParams[3]));
}

inline void compute_salt_and_pepper_noise_48_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pSaltAndPepperNoiseParams)
{
    compute_salt_and_pepper_noise_24_host(p    , p + 2, p +  4, pxXorwowStateX, pxXorwowStateCounter, pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_24_host(p + 1, p + 3, p +  5, pxXorwowStateX, pxXorwowStateCounter, pSaltAndPepperNoiseParams);
}

inline void compute_salt_and_pepper_noise_48_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pSaltAndPepperNoiseParams)
{
    compute_salt_and_pepper_noise_12_host(p    , p + 4, p +  8, pxXorwowStateX, pxXorwowStateCounter, pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_12_host(p + 1, p + 5, p +  9, pxXorwowStateX, pxXorwowStateCounter, pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_12_host(p + 2, p + 6, p + 10, pxXorwowStateX, pxXorwowStateCounter, pSaltAndPepperNoiseParams);
    compute_salt_and_pepper_noise_12_host(p + 3, p + 7, p + 11, pxXorwowStateX, pxXorwowStateCounter, pSaltAndPepperNoiseParams);
}

inline void compute_shot_noise_8_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pShotNoiseFactorInv, __m256 *pShotNoiseFactor)
{
    __m256 pShotNoiseValue = avx_p0;                                                                                                                                // Rpp32u shotNoiseValue = 0;
    __m256 pFactValue = fast_exp_avx(_mm256_mul_ps(*p, *pShotNoiseFactorInv));                                                                                      // Rpp32f factValue = expf(lambda);
    __m256 pFactValueMask = avx_p1;                                                                                                                                 // Set mask for pFactValue computation to 1
    do
    {
        pShotNoiseValue = _mm256_or_ps(_mm256_andnot_ps(pFactValueMask, pShotNoiseValue), _mm256_and_ps(pFactValueMask, _mm256_add_ps(pShotNoiseValue, avx_p1)));   // shotNoiseValue++;
        pFactValue = _mm256_mul_ps(pFactValue, rpp_host_rng_xorwow_8_f32_avx(pxXorwowStateX, pxXorwowStateCounter));                                                // factValue *= rpp_host_rng_xorwow_f32(xorwowStatePtr);
        pFactValueMask = _mm256_cmp_ps(pFactValue, avx_p1, _CMP_GT_OQ);                                                                                             // compute new pFactValueMask for loop exit condition
    } while (_mm256_movemask_epi8(_mm256_cvtps_epi32(pFactValueMask)) != 0);                                                                                        // while (factValue > 1.0f);

    pShotNoiseValue = _mm256_sub_ps(pShotNoiseValue, avx_p1);                                                                                                       // shotNoiseValue -= 1;
    *p = _mm256_mul_ps(pShotNoiseValue, *pShotNoiseFactor);                                                                                                         // dst = pShotNoiseValue * shotNoiseFactor;
}

inline void compute_shot_noise_4_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pShotNoiseFactorInv, __m128 *pShotNoiseFactor)
{
    __m128 pShotNoiseValue = xmm_p0;                                                                                                                    // Rpp32u shotNoiseValue = 0;
    __m128 pFactValue = fast_exp_sse(_mm_mul_ps(*p, *pShotNoiseFactorInv));                                                                             // Rpp32f factValue = expf(lambda);
    __m128 pFactValueMask = xmm_p1;                                                                                                                     // Set mask for pFactValue computation to 1
    do
    {
        pShotNoiseValue = _mm_or_ps(_mm_andnot_ps(pFactValueMask, pShotNoiseValue), _mm_and_ps(pFactValueMask, _mm_add_ps(pShotNoiseValue, xmm_p1)));   // shotNoiseValue++;
        pFactValue = _mm_mul_ps(pFactValue, rpp_host_rng_xorwow_4_f32_sse(pxXorwowStateX, pxXorwowStateCounter));                                       // factValue *= rpp_host_rng_xorwow_f32(xorwowStatePtr);
        pFactValueMask = _mm_cmpgt_ps(pFactValue, xmm_p1);                                                                                              // compute new pFactValueMask for loop exit condition
    } while (_mm_movemask_epi8(_mm_cvtps_epi32(pFactValueMask)) != 0);                                                                                  // while (factValue > 1.0f);

    pShotNoiseValue = _mm_sub_ps(pShotNoiseValue, xmm_p1);                                                                                              // shotNoiseValue -= 1;
    *p = _mm_mul_ps(pShotNoiseValue, *pShotNoiseFactor);                                                                                                // dst = pShotNoiseValue * shotNoiseFactor;
}

inline void compute_shot_noise_48_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pShotNoiseFactorInv, __m256 *pShotNoiseFactor)
{
    compute_shot_noise_8_host(p     , pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  1, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  2, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  3, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  4, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  5, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
}

inline void compute_shot_noise_48_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pShotNoiseFactorInv, __m128 *pShotNoiseFactor)
{
    compute_shot_noise_4_host(p     , pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  1, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  2, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  3, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  4, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  5, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  6, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  7, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  8, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  9, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p + 10, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p + 11, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
}

inline void compute_shot_noise_24_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pShotNoiseFactorInv, __m256 *pShotNoiseFactor)
{
    compute_shot_noise_8_host(p     , pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  1, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  2, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
}

inline void compute_shot_noise_16_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pShotNoiseFactorInv, __m256 *pShotNoiseFactor)
{
    compute_shot_noise_8_host(p     , pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_8_host(p +  1, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
}

inline void compute_shot_noise_16_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pShotNoiseFactorInv, __m128 *pShotNoiseFactor)
{
    compute_shot_noise_4_host(p     , pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  1, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  2, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  3, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
}

inline void compute_shot_noise_12_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pShotNoiseFactorInv, __m128 *pShotNoiseFactor)
{
    compute_shot_noise_4_host(p     , pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  1, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
    compute_shot_noise_4_host(p +  2, pxXorwowStateX, pxXorwowStateCounter, pShotNoiseFactorInv, pShotNoiseFactor);
}

inline Rpp32u compute_shot_noise_1_host(RpptXorwowState *xorwowStatePtr, Rpp32f lambda)
{
    Rpp32u shotNoiseValue = 0;                                   // initialize shotNoiseValue to 0
    Rpp32f factValue = rpp_host_math_exp_lim256approx(lambda);   // initialize factValue to e^lambda
    do
    {
        shotNoiseValue++;                                        // additively cumulate shotNoiseValue by 1 until exit condition
        factValue *= rpp_host_rng_xorwow_f32(xorwowStatePtr);    // multiplicatively cumulate factValue by the next uniform random number until exit condition
    } while (factValue > 1.0f);                                  // loop while factValue >= 1.0f

    return shotNoiseValue - 1;
}

inline void compute_gaussian_noise_16_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pGaussianNoiseParams)
{
    __m256 pRngVals[2], pSqrt[2];

    rpp_host_rng_16_gaussian_f32_avx(pRngVals, pxXorwowStateX, pxXorwowStateCounter);               // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm256_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);   // rngVal = rngVal * stdDev + mean;
    pRngVals[1] = _mm256_fmadd_ps(pRngVals[1], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);   // rngVal = rngVal * stdDev + mean;
    pSqrt[0] = _mm256_sqrt_ps(p[0]);                                                                // pixSqrt = sqrt(pixVal);
    pSqrt[1] = _mm256_sqrt_ps(p[1]);                                                                // pixSqrt = sqrt(pixVal);
    p[0] = _mm256_fmadd_ps(pSqrt[0], pRngVals[0], p[0]);                                            // return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
    p[1] = _mm256_fmadd_ps(pSqrt[1], pRngVals[1], p[1]);                                            // return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
}

inline void compute_gaussian_noise_voxel_16_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pGaussianNoiseParams)
{
    __m256 pRngVals[2], pSqrt[2];

    rpp_host_rng_16_gaussian_f32_avx(pRngVals, pxXorwowStateX, pxXorwowStateCounter);               // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm256_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);   // rngVal = rngVal * stdDev + mean;
    pRngVals[1] = _mm256_fmadd_ps(pRngVals[1], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);   // rngVal = rngVal * stdDev + mean;
    p[0] = _mm256_add_ps(p[0], pRngVals[0]);                                                        // return pixVal + rngVal;
    p[1] = _mm256_add_ps(p[1], pRngVals[1]);                                                        // return pixVal + rngVal;
}

inline void compute_gaussian_noise_8_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    __m128 pRngVals[2], pSqrt[2];

    rpp_host_rng_8_gaussian_f32_sse(pRngVals, pxXorwowStateX, pxXorwowStateCounter);            // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);  // rngVal = rngVal * stdDev + mean;
    pRngVals[1] = _mm_fmadd_ps(pRngVals[1], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);  // rngVal = rngVal * stdDev + mean;
    pSqrt[0] = _mm_sqrt_ps(p[0]);                                                               // pixSqrt = sqrt(pixVal);
    pSqrt[1] = _mm_sqrt_ps(p[1]);                                                               // pixSqrt = sqrt(pixVal);
    p[0] = _mm_fmadd_ps(pSqrt[0], pRngVals[0], p[0]);                                           // return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
    p[1] = _mm_fmadd_ps(pSqrt[1], pRngVals[1], p[1]);                                           // return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
}

inline void compute_gaussian_noise_voxel_8_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    __m128 pRngVals[2], pSqrt[2];

    rpp_host_rng_8_gaussian_f32_sse(pRngVals, pxXorwowStateX, pxXorwowStateCounter);            // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);  // rngVal = rngVal * stdDev + mean;
    pRngVals[1] = _mm_fmadd_ps(pRngVals[1], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);  // rngVal = rngVal * stdDev + mean;
    p[0] = _mm_add_ps(p[0], pRngVals[0]);                                                       // return (pixVal + rngVal);
    p[1] = _mm_add_ps(p[1], pRngVals[1]);                                                       // return (pixVal + rngVal);
}

inline void compute_gaussian_noise_16_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    compute_gaussian_noise_8_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_8_host(&p[2], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
}

inline void compute_gaussian_noise_voxel_16_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    compute_gaussian_noise_voxel_8_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_voxel_8_host(&p[2], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
}

inline void compute_gaussian_noise_48_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pGaussianNoiseParams)
{
    compute_gaussian_noise_16_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_16_host(&p[2], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_16_host(&p[4], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
}

inline void compute_gaussian_noise_voxel_48_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pGaussianNoiseParams)
{
    compute_gaussian_noise_voxel_16_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_voxel_16_host(&p[2], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_voxel_16_host(&p[4], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
}

inline void compute_gaussian_noise_48_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    compute_gaussian_noise_16_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_16_host(&p[4], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_16_host(&p[8], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
}

inline void compute_gaussian_noise_voxel_48_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    compute_gaussian_noise_voxel_16_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_voxel_16_host(&p[4], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
    compute_gaussian_noise_voxel_16_host(&p[8], pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);
}

inline void compute_gaussian_noise_24_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pGaussianNoiseParams)
{
    compute_gaussian_noise_16_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);

    __m256 pRngVals[2], pSqrt;
    rpp_host_rng_16_gaussian_f32_avx(pRngVals, pxXorwowStateX, pxXorwowStateCounter);               // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm256_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);   // rngVal = rngVal * stdDev + mean;
    pSqrt = _mm256_sqrt_ps(p[2]);                                                                   // pixSqrt = sqrt(pixVal);
    p[2] = _mm256_fmadd_ps(pSqrt, pRngVals[0], p[2]);                                               // return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
}

inline void compute_gaussian_noise_voxel_24_host(__m256 *p, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 *pGaussianNoiseParams)
{
    compute_gaussian_noise_16_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);

    __m256 pRngVals[2];
    rpp_host_rng_16_gaussian_f32_avx(pRngVals, pxXorwowStateX, pxXorwowStateCounter);               // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm256_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);   // rngVal = rngVal * stdDev + mean;
    p[2] = _mm256_add_ps(p[2], pRngVals[0]);                                                        // return pixVal + rngVal;
}

inline void compute_gaussian_noise_12_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    compute_gaussian_noise_8_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);

    __m128 pRngVals[2], pSqrt;
    rpp_host_rng_8_gaussian_f32_sse(pRngVals, pxXorwowStateX, pxXorwowStateCounter);            // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);  // rngVal = rngVal * stdDev + mean;
    pSqrt = _mm_sqrt_ps(p[2]);                                                                  // pixSqrt = sqrt(pixVal);
    p[2] = _mm_fmadd_ps(pSqrt, pRngVals[0], p[2]);                                              // return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
}

inline void compute_gaussian_noise_voxel_12_host(__m128 *p, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter, __m128 *pGaussianNoiseParams)
{
    compute_gaussian_noise_8_host(p, pxXorwowStateX, pxXorwowStateCounter, pGaussianNoiseParams);

    __m128 pRngVals[2], pSqrt;
    rpp_host_rng_8_gaussian_f32_sse(pRngVals, pxXorwowStateX, pxXorwowStateCounter);            // rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    pRngVals[0] = _mm_fmadd_ps(pRngVals[0], pGaussianNoiseParams[1], pGaussianNoiseParams[0]);  // rngVal = rngVal * stdDev + mean;
    p[2] = _mm_add_ps(p[2], pRngVals[0]);                                                       // return pixVal + rngVal;
}

inline Rpp32f compute_gaussian_noise_1_host(Rpp32f pixVal, RpptXorwowStateBoxMuller *xorwowStatePtr, Rpp32f mean, Rpp32f stdDev)
{
    Rpp32f rngVal, pixSqrt;
    rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    rngVal = rngVal * stdDev + mean;
    pixSqrt = sqrt(pixVal);

    return RPPPIXELCHECKF32(pixSqrt * rngVal + pixVal);
}

inline Rpp32f compute_gaussian_noise_voxel_1_host(Rpp32f pixVal, RpptXorwowStateBoxMuller *xorwowStatePtr, Rpp32f mean, Rpp32f stdDev)
{
    Rpp32f rngVal, pixSqrt;
    rngVal = rpp_host_rng_1_gaussian_f32(xorwowStatePtr);
    rngVal = rngVal * stdDev + mean;
    return pixVal + rngVal;
}

inline void compute_offset_i8_1c_avx(__m256 &p)
{
    p = _mm256_add_ps(p, avx_p128);
}

inline void compute_offset_i8_3c_avx(__m256 *p)
{
    compute_offset_i8_1c_avx(p[0]);
    compute_offset_i8_1c_avx(p[1]);
    compute_offset_i8_1c_avx(p[2]);
}

// Compute Functions for RPP Image API

template<typename T>
inline RppStatus compute_subimage_location_host(T* ptr, T** ptrSubImage,
                                         RppiSize size, RppiSize *sizeSubImage,
                                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if ((RPPINRANGE(x1, 0, size.width - 1) == 0)
        || (RPPINRANGE(x2, 0, size.width - 1) == 0)
        || (RPPINRANGE(y1, 0, size.height - 1) == 0)
        || (RPPINRANGE(y2, 0, size.height - 1) == 0))
    {
        return RPP_ERROR;
    }

    int yDiff = (int) y2 - (int) y1;
    int xDiff = (int) x2 - (int) x1;

    sizeSubImage->height = (Rpp32u) RPPABS(yDiff) + 1;
    sizeSubImage->width = (Rpp32u) RPPABS(xDiff) + 1;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        *ptrSubImage = ptr + (RPPMIN2(y1, y2) * size.width) + RPPMIN2(x1, x2);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        *ptrSubImage = ptr + (RPPMIN2(y1, y2) * size.width * channel) + (RPPMIN2(x1, x2) * channel);
    }

    return RPP_SUCCESS;
}

template<typename T>
inline void compute_transpose_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                                 RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < dstSize.height; i++)
            {
                for (int j = 0; j < dstSize.width; j++)
                {
                    *dstPtrTemp = *(srcPtrTemp + (j * srcSize.width) + i);
                    dstPtrTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < dstSize.height; i++)
        {
            for (int j = 0; j < dstSize.width; j++)
            {
                srcPtrTemp = srcPtr + (channel * ((j * srcSize.width) + i));
                for (int c = 0; c < channel; c++)
                {
                    *dstPtrTemp = *(srcPtrTemp + c);
                    dstPtrTemp++;
                }
            }
        }
    }
}

template <typename T, typename U>
inline void compute_add_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32s pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_subtract_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                        Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32s pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) - ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_multiply_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32f) (*srcPtr1Temp)) * ((Rpp32f) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp = (T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_bitwise_AND_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                           Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp8u pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp8u) (*srcPtr1Temp)) & ((Rpp8u) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_inclusive_OR_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                            Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp8u pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp8u) (*srcPtr1Temp)) | ((Rpp8u) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_exclusive_OR_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                            Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp8u pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp8u) (*srcPtr1Temp)) ^ ((Rpp8u) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_min_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = RPPMIN2(*srcPtr1Temp, ((T)*srcPtr2Temp));
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_max_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = RPPMAX2(*srcPtr1Temp, ((T)*srcPtr2Temp));
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_rgb_to_hsv_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
    U *dstPtrTempH, *dstPtrTempS, *dstPtrTempV;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp64u imageDim = srcSize.height * srcSize.width;
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + (imageDim);
        srcPtrTempB = srcPtr + (2 * imageDim);
        dstPtrTempH = dstPtr;
        dstPtrTempS = dstPtr + (imageDim);
        dstPtrTempV = dstPtr + (2 * imageDim);

        Rpp64u bufferLength = srcSize.height * srcSize.width;
        Rpp64u alignedLength = bufferLength & ~3;

        __m128i const zero = _mm_setzero_si128();
        __m128 pDiv = _mm_set1_ps(255.0);
        __m128 pMul = _mm_set1_ps(360.0);
        __m128i px0, px1, px2;
        __m128 xR, xG, xB;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
        {
            px0 =  _mm_loadu_si128((__m128i *)srcPtrTempR);
            px1 =  _mm_loadu_si128((__m128i *)srcPtrTempG);
            px2 =  _mm_loadu_si128((__m128i *)srcPtrTempB);

            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
            px1 = _mm_unpacklo_epi8(px1, zero);    // pixels 0-7
            px2 = _mm_unpacklo_epi8(px2, zero);    // pixels 0-7

            xR = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
            xG = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 0-3
            xB = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px2, zero));    // pixels 0-3

            xR = _mm_div_ps(xR, pDiv);
            xG = _mm_div_ps(xG, pDiv);
            xB = _mm_div_ps(xB, pDiv);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]

            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction. If H >= 1 then H - 1
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Multiply by 360
            xG = _mm_mul_ps(xG, pMul);

            _mm_storeu_ps(dstPtrTempH, xG);
            _mm_storeu_ps(dstPtrTempS, xS);
            _mm_storeu_ps(dstPtrTempV, xV);

            srcPtrTempR += 4;
            srcPtrTempG += 4;
            srcPtrTempB += 4;
            dstPtrTempH += 4;
            dstPtrTempS += 4;
            dstPtrTempV += 4;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                *dstPtrTempH = 0;
            }
            else if (cmax == rf)
            {
                *dstPtrTempH = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                *dstPtrTempH = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                *dstPtrTempH = round(60 * (((rf - gf) / delta) + 4));
            }

            while (*dstPtrTempH > 360)
            {
                *dstPtrTempH = *dstPtrTempH - 360;
            }
            while (*dstPtrTempH < 0)
            {
                *dstPtrTempH = 360 + *dstPtrTempH;
            }

            if (cmax == 0)
            {
                *dstPtrTempS = 0;
            }
            else
            {
                *dstPtrTempS = delta / cmax;
            }

            *dstPtrTempV = cmax;

            srcPtrTempR++;
            srcPtrTempG++;
            srcPtrTempB++;
            dstPtrTempH++;
            dstPtrTempS++;
            dstPtrTempV++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + 1;
        srcPtrTempB = srcPtr + 2;
        dstPtrTempH = dstPtr;
        dstPtrTempS = dstPtr + 1;
        dstPtrTempV = dstPtr + 2;

        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                *dstPtrTempH = 0;
            }
            else if (cmax == rf)
            {
                *dstPtrTempH = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                *dstPtrTempH = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                *dstPtrTempH = round(60 * (((rf - gf) / delta) + 4));
            }

            while (*dstPtrTempH > 360)
            {
                *dstPtrTempH = *dstPtrTempH - 360;
            }
            while (*dstPtrTempH < 0)
            {
                *dstPtrTempH = 360 + *dstPtrTempH;
            }

            if (cmax == 0)
            {
                *dstPtrTempS = 0;
            }
            else
            {
                *dstPtrTempS = delta / cmax;
            }

            *dstPtrTempV = cmax;

            srcPtrTempR += 3;
            srcPtrTempG += 3;
            srcPtrTempB += 3;
            dstPtrTempH += 3;
            dstPtrTempS += 3;
            dstPtrTempV += 3;
        }
    }
}

template <typename T, typename U>
inline void compute_hsv_to_rgb_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTempH, *srcPtrTempS, *srcPtrTempV;
    U *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp64u imageDim = srcSize.height * srcSize.width;
        srcPtrTempH = srcPtr;
        srcPtrTempS = srcPtr + (imageDim);
        srcPtrTempV = srcPtr + (2 * imageDim);
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + (imageDim);
        dstPtrTempB = dstPtr + (2 * imageDim);

        Rpp64u bufferLength = srcSize.height * srcSize.width;
        Rpp64u alignedLength = bufferLength & ~3;

        __m128 pDiv = _mm_set1_ps(360.0);
        __m128 pMul = _mm_set1_ps(255.0);

        __m128 h0, h1, h2, h3;
        h0 = _mm_set1_ps(1.0);
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        __m128i px1, px2, px3;

        Rpp8u arrayR[4];
        Rpp8u arrayG[4];
        Rpp8u arrayB[4];

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
        {
            // Load
            h1 =  _mm_loadu_si128((__m128i *)srcPtrTempH);
            h2 =  _mm_loadu_si128((__m128i *)srcPtrTempS);
            h3 =  _mm_loadu_si128((__m128i *)srcPtrTempV);

            h1 = _mm_div_ps(h1, pDiv);

            _MM_TRANSPOSE4_PS (h0, h1, h2, h3);

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            _MM_TRANSPOSE4_PS (x0, x1, x2, x3);

            x1 = _mm_mul_ps(x1, pMul);
            x2 = _mm_mul_ps(x2, pMul);
            x3 = _mm_mul_ps(x3, pMul);

            px1 = _mm_cvtps_epi32(x1);
            px2 = _mm_cvtps_epi32(x2);
            px3 = _mm_cvtps_epi32(x3);

            px1 = _mm_packs_epi32(px1, px1);
            px1 = _mm_packus_epi16(px1, px1);
            *((int*)arrayR) = _mm_cvtsi128_si32(px1);

            px2 = _mm_packs_epi32(px2, px2);
            px2 = _mm_packus_epi16(px2, px2);
            *((int*)arrayG) = _mm_cvtsi128_si32(px2);

            px3 = _mm_packs_epi32(px3, px3);
            px3 = _mm_packus_epi16(px3, px3);
            *((int*)arrayB) = _mm_cvtsi128_si32(px3);

            memcpy(dstPtrTempR, arrayR, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempG, arrayG, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempB, arrayB, 4 * sizeof(Rpp8u));

            srcPtrTempH += 4;
            srcPtrTempS += 4;
            srcPtrTempV += 4;
            dstPtrTempR += 4;
            dstPtrTempG += 4;
            dstPtrTempB += 4;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            Rpp32f c, x, m, rf, gf, bf;
            c = *srcPtrTempV * *srcPtrTempS;
            x = c * (1 - abs(int(fmod((*srcPtrTempH / 60), 2)) - 1));
            m = *srcPtrTempV - c;

            if ((0 <= *srcPtrTempH) && (*srcPtrTempH < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= *srcPtrTempH) && (*srcPtrTempH < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= *srcPtrTempH) && (*srcPtrTempH < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= *srcPtrTempH) && (*srcPtrTempH < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= *srcPtrTempH) && (*srcPtrTempH < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= *srcPtrTempH) && (*srcPtrTempH < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempH++;
            srcPtrTempS++;
            srcPtrTempV++;
            dstPtrTempR++;
            dstPtrTempG++;
            dstPtrTempB++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrTempH = srcPtr;
        srcPtrTempS = srcPtr + 1;
        srcPtrTempV = srcPtr + 2;
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + 1;
        dstPtrTempB = dstPtr + 2;

        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f c, x, m, rf, gf, bf;
            c = *srcPtrTempV * *srcPtrTempS;
            x = c * (1 - abs(int(fmod((*srcPtrTempH / 60), 2)) - 1));
            m = *srcPtrTempV - c;

            if ((0 <= *srcPtrTempH) && (*srcPtrTempH < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= *srcPtrTempH) && (*srcPtrTempH < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= *srcPtrTempH) && (*srcPtrTempH < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= *srcPtrTempH) && (*srcPtrTempH < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= *srcPtrTempH) && (*srcPtrTempH < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= *srcPtrTempH) && (*srcPtrTempH < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempH += 3;
            srcPtrTempS += 3;
            srcPtrTempV += 3;
            dstPtrTempR += 3;
            dstPtrTempG += 3;
            dstPtrTempB += 3;
        }
    }
}

template <typename T, typename U>
inline void compute_magnitude_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, U* dstPtr,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr1Temp, *srcPtr2Temp;
    U *dstPtrTemp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;
    Rpp32s srcPtr1Value, srcPtr2Value;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        srcPtr1Value = (Rpp32s) *srcPtr1Temp;
        srcPtr2Value = (Rpp32s) *srcPtr2Temp;
        pixel = sqrt((srcPtr1Value * srcPtr1Value) + (srcPtr2Value * srcPtr2Value));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(U) round(pixel);
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T>
inline void compute_magnitude_ROI_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                                            Rpp32f x1, Rpp32f y1, Rpp32f x2, Rpp32f y2,
                                            RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u imageDim = srcSize.height * srcSize.width;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            T *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
            srcPtr1Channel = srcPtr1 + (c * imageDim);
            srcPtr2Channel = srcPtr2 + (c * imageDim);
            dstPtrChannel = dstPtr + (c * imageDim);


            for(int i = 0; i < srcSize.height; i++)
            {
                Rpp32f pixel;

                T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Channel + (i * srcSize.width);
                srcPtr2Temp = srcPtr2Channel + (i * srcSize.width);
                dstPtrTemp = dstPtrChannel + (i * srcSize.width);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtr1Temp, srcSize.width * sizeof(T));

                    srcPtr1Temp += srcSize.width;
                    srcPtr2Temp += srcSize.width;
                    dstPtrTemp += srcSize.width;
                }
                else
                {
                    for(int j = 0; j < srcSize.width; j++)
                    {
                        if((x1 <= j) && (j <= x2 ))
                        {
                            Rpp32s srcPtr1Value = (Rpp32s) *srcPtr1Temp;
                            Rpp32s srcPtr2Value = (Rpp32s) *srcPtr2Temp;
                            pixel = sqrt((srcPtr1Value * srcPtr1Value) + (srcPtr2Value * srcPtr2Value));
                            pixel = RPPPIXELCHECK(pixel);
                            *dstPtrTemp =(T) round(pixel);

                            srcPtr1Temp++;
                            srcPtr2Temp++;
                            dstPtrTemp++;
                        }
                        else
                        {
                            *dstPtrTemp = *srcPtr1Temp;

                            srcPtr1Temp++;
                            srcPtr2Temp++;
                            dstPtrTemp++;
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;


        for(int i = 0; i < srcSize.height; i++)
        {
            Rpp32f pixel;

            T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
            srcPtr1Temp = srcPtr1 + (i * elementsInRow);
            srcPtr2Temp = srcPtr2 + (i * elementsInRow);
            dstPtrTemp = dstPtr + (i * elementsInRow);

            if (!((y1 <= i) && (i <= y2)))
            {
                memcpy(dstPtrTemp, srcPtr1Temp, elementsInRow * sizeof(T));

                srcPtr1Temp += elementsInRow;
                srcPtr2Temp += elementsInRow;
                dstPtrTemp += elementsInRow;
            }
            else
            {
                for(int j = 0; j < srcSize.width; j++)
                {
                    if (!((x1 <= j) && (j <= x2 )))
                    {
                        memcpy(dstPtrTemp, srcPtr1Temp, channel * sizeof(T));

                        srcPtr1Temp += channel;
                        srcPtr2Temp += channel;
                        dstPtrTemp += channel;
                    }
                    else
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            Rpp32s srcPtr1Value = (Rpp32s) *srcPtr1Temp;
                            Rpp32s srcPtr2Value = (Rpp32s) *srcPtr2Temp;
                            pixel = sqrt((srcPtr1Value * srcPtr1Value) + (srcPtr2Value * srcPtr2Value));
                            pixel = RPPPIXELCHECK(pixel);
                            *dstPtrTemp =(T) round(pixel);

                            srcPtr1Temp++;
                            srcPtr2Temp++;
                            dstPtrTemp++;
                        }
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
inline void compute_threshold_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                                 U min, U max, Rpp32u type,
                                 RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp;
    U *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (type == 1)
    {
        for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
        {
            if (*srcPtrTemp < min)
            {
                *dstPtrTemp = (U) 0;
            }
            else if (*srcPtrTemp <= max)
            {
                *dstPtrTemp = (U) 255;
            }
            else
            {
                *dstPtrTemp = (U) 0;
            }

            srcPtrTemp++;
            dstPtrTemp++;
        }
    }
    else if (type == 2)
    {
        for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
        {
            if (RPPABS(*srcPtrTemp) < min)
            {
                *dstPtrTemp = (U) 0;
            }
            else if (RPPABS(*srcPtrTemp) <= max)
            {
                *dstPtrTemp = (U) 255;
            }
            else
            {
                *dstPtrTemp = (U) 0;
            }

            srcPtrTemp++;
            dstPtrTemp++;
        }
    }
}

template <typename T>
inline void compute_data_object_copy_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    memcpy(dstPtr, srcPtr, srcSize.height * srcSize.width * channel * sizeof(T));

}

template <typename T>
inline void compute_downsampled_image_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                                         RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp8u checkEven;
    checkEven = (Rpp8u) RPPISEVEN(srcSize.width);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);

            for (int i = 0; i < dstSize.height; i++)
            {
                for (int j = 0; j < dstSize.width; j++)
                {
                    *dstPtrTemp = *srcPtrTemp;
                    srcPtrTemp += 2;
                    dstPtrTemp++;
                }
                if (checkEven == 0)
                {
                    srcPtrTemp--;
                }
                srcPtrTemp += srcSize.width;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = srcSize.width * channel;

        for (int i = 0; i < dstSize.height; i++)
        {
            for (int j = 0; j < dstSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *dstPtrTemp = *srcPtrTemp;
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtrTemp += channel;
            }
            if (checkEven == 0)
            {
                srcPtrTemp -= channel;
            }
            srcPtrTemp += elementsInRow;
        }
    }
}

template <typename T>
inline RppStatus compute_channel_extract_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                       Rpp32u extractChannelNumber,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    if (extractChannelNumber != 0 && extractChannelNumber != 1 && extractChannelNumber != 2)
    {
        return RPP_ERROR;
    }

    T *srcPtrTemp, *dstPtrTemp;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        srcPtrTemp = srcPtr + (extractChannelNumber * srcSize.height * srcSize.width);
        memcpy(dstPtrTemp, srcPtrTemp, srcSize.height * srcSize.width * sizeof(T));
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrTemp = srcPtr + extractChannelNumber;
        for (int i = 0; i < srcSize.height * srcSize.width; i++)
        {
            *dstPtrTemp = *srcPtrTemp;
            srcPtrTemp = srcPtrTemp + channel;
            dstPtrTemp++;
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
inline void compute_gradient_direction_host(T* gradientX, T* gradientY, RppiSize srcSize, U* gradientDirection,
                                          RppiChnFormat chnFormat, Rpp32u channel)
{
    T *gradientXTemp, *gradientYTemp;
    U *gradientDirectionTemp;
    gradientXTemp = gradientX;
    gradientYTemp = gradientY;
    gradientDirectionTemp = gradientDirection;

    Rpp32f pixel;

    for (int i = 0; i < (srcSize.height * srcSize.width * channel); i++)
    {
        if (*gradientXTemp != 0)
        {
            *gradientDirectionTemp = atan((Rpp32f) *gradientYTemp / (Rpp32f) *gradientXTemp);
        }
        else
        {
            if (*gradientYTemp > 0)
            {
                *gradientDirectionTemp = ((Rpp32f) PI) / 2.0;
            }
            else if (*gradientYTemp < 0)
            {
                *gradientDirectionTemp = ((Rpp32f) PI) / 2.0 * -1.0;
            }
            else if (*gradientYTemp == 0)
            {
                *gradientDirectionTemp = 0.0;
            }
        }
        gradientDirectionTemp++;
        gradientXTemp++;
        gradientYTemp++;
    }
}

inline Rpp32u fogGenerator(Rpp32u srcPtr, Rpp32f fogValue, int colour, int check)
{
    int fog = 0;
    int range = 3;
    if(check >= (240) && fogValue!=0);
    else if(check>=(170))
        range = 1;
    else if(check<=(85))
        range = 2;
    else
        range = 3;
    switch(range)
    {
        case 1:
            if(colour==1)
            {
                fog = srcPtr * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
            }
            else if(colour==2)
            {
                fog = srcPtr * (1.5 + fogValue) + (7*fogValue);
            }
            else
            {
                fog = srcPtr * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
            }
            break;
        case 2:
            if(colour==1)
            {
                fog = srcPtr * (1.5 + pow(fogValue,2)) - (fogValue*4) + (130*fogValue);
            }
            else if(colour==2)
            {
                fog = srcPtr * (1.5 + pow(fogValue,2)) + (130*fogValue);
            }
            else
            {
                fog = srcPtr * (1.5 + pow(fogValue,2)) + (fogValue*4) + 130*fogValue;
            }
            break;
        case 3:
            if(colour==1)
            {
                fog = srcPtr * (1.5 + pow(fogValue,1.5)) - (fogValue*4) + 20 + (100*fogValue);
            }
            else if(colour==2)
            {
                fog = srcPtr * (1.5 + pow(fogValue,1.5)) + 20 + (100*fogValue);
            }
            else
            {
                fog = srcPtr * (1.5 + pow(fogValue,1.5)) + (fogValue*4) + (100*fogValue);
            }
            break;
    }
    fog = RPPPIXELCHECK(fog);
    return fog;
}

inline void compute_image_location_host(RppiSize *batch_srcSizeMax, int batchCount, Rpp32u *loc, Rpp32u channel)
{
    for (int m = 0; m < batchCount; m++)
    {
        *loc += (batch_srcSizeMax[m].height * batch_srcSizeMax[m].width);
    }
    *loc *= channel;
}

template <typename T>
inline void compute_1_channel_minmax_host(T *srcPtr, RppiSize srcSize, RppiSize srcSizeMax,
                                                    T *min, T *max,
                                                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp;
    srcPtrTemp = srcPtr;

    __m128i pMin, pMax;

    for (int i = 0; i < srcSize.height; i++)
    {
        pMin = _mm_set1_epi8(*min);
        pMax = _mm_set1_epi8(*max);

        int bufferLength = srcSize.width;
        int alignedLength = bufferLength & ~15;

        __m128i px0;

        int vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
        {
            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
            pMin = _mm_min_epu8(px0, pMin);
            pMax = _mm_max_epu8(px0, pMax);
            srcPtrTemp +=16;
        }
        *min = (T) HorMin(pMin);
        *max = (T) HorMax(pMax);
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            if (*srcPtrTemp < *min)
            {
                *min = *srcPtrTemp;
            }
            if (*srcPtrTemp > *max)
            {
                *max = *srcPtrTemp;
            }
            srcPtrTemp++;
        }
        srcPtrTemp += (srcSizeMax.width - srcSize.width);
    }
}

template <typename T>
inline void compute_3_channel_minmax_host(T *srcPtr, RppiSize srcSize, RppiSize srcSizeMax,
                                               T *min, T *max,
                                               RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp;
    srcPtrTemp = srcPtr;

    __m128i pMin, pMax;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            T minTemp, maxTemp;
            minTemp = *(min + c);
            maxTemp = *(max + c);
            for (int i = 0; i < srcSize.height; i++)
            {
                pMin = _mm_set1_epi8(minTemp);
                pMax = _mm_set1_epi8(maxTemp);

                int bufferLength = srcSize.width;
                int alignedLength = bufferLength & ~15;

                __m128i px0;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                    pMin = _mm_min_epu8(px0, pMin);
                    pMax = _mm_max_epu8(px0, pMax);
                    srcPtrTemp +=16;
                }
                minTemp = (T) HorMin(pMin);
                maxTemp = (T) HorMax(pMax);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if (*srcPtrTemp < minTemp)
                    {
                        minTemp = *srcPtrTemp;
                    }
                    if (*srcPtrTemp > maxTemp)
                    {
                        maxTemp = *srcPtrTemp;
                    }
                    srcPtrTemp++;
                }
                srcPtrTemp += (srcSizeMax.width - srcSize.width);
            }
            *(min + c) = minTemp;
            *(max + c) = maxTemp;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u elementsInRowMax = channel * srcSizeMax.width;

        T minRTemp, maxRTemp, minGTemp, maxGTemp, minBTemp, maxBTemp;
        minRTemp = *min;
        maxRTemp = *max;
        minGTemp = *(min + 1);
        maxGTemp = *(max + 1);
        minBTemp = *(min + 2);
        maxBTemp = *(max + 2);

        pMin = _mm_set1_epi8(minRTemp);
        pMax = _mm_set1_epi8(maxRTemp);

        for (int i = 0; i < srcSize.height; i++)
        {
            int bufferLength = elementsInRow;
            int alignedLength = bufferLength & ~14;

            __m128i px0;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                pMin = _mm_min_epu8(px0, pMin);
                pMax = _mm_max_epu8(px0, pMax);
                srcPtrTemp +=15;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
            {
                if (*srcPtrTemp < minRTemp)
                {
                    minRTemp = *srcPtrTemp;
                }
                if (*srcPtrTemp > maxRTemp)
                {
                    maxRTemp = *srcPtrTemp;
                }
                srcPtrTemp++;
                if (*srcPtrTemp < minGTemp)
                {
                    minGTemp = *srcPtrTemp;
                }
                if (*srcPtrTemp > maxGTemp)
                {
                    maxGTemp = *srcPtrTemp;
                }
                srcPtrTemp++;
                if (*srcPtrTemp < minBTemp)
                {
                    minBTemp = *srcPtrTemp;
                }
                if (*srcPtrTemp > maxBTemp)
                {
                    maxBTemp = *srcPtrTemp;
                }
                srcPtrTemp++;
            }
            srcPtrTemp += (elementsInRowMax - elementsInRow);
        }

        T minVector[16], maxVector[16];
        _mm_storeu_si128((__m128i *)minVector, pMin);
        _mm_storeu_si128((__m128i *)maxVector, pMax);

        minRTemp = RPPMIN2(RPPMIN3(minVector[0], minVector[3], minVector[6]), RPPMIN3(minVector[9], minVector[12], minRTemp));
        minGTemp = RPPMIN2(RPPMIN3(minVector[1], minVector[4], minVector[7]), RPPMIN3(minVector[10], minVector[13], minGTemp));
        minBTemp = RPPMIN2(RPPMIN3(minVector[2], minVector[5], minVector[8]), RPPMIN3(minVector[11], minVector[14], minBTemp));

        maxRTemp = RPPMAX2(RPPMAX3(maxVector[0], maxVector[3], maxVector[6]), RPPMAX3(maxVector[9], maxVector[12], maxRTemp));
        maxGTemp = RPPMAX2(RPPMAX3(maxVector[1], maxVector[4], maxVector[7]), RPPMAX3(maxVector[10], maxVector[13], maxGTemp));
        maxBTemp = RPPMAX2(RPPMAX3(maxVector[2], maxVector[5], maxVector[8]), RPPMAX3(maxVector[11], maxVector[14], maxBTemp));

        *min = minRTemp;
        *max = maxRTemp;
        *(min + 1) = minGTemp;
        *(max + 1) = maxGTemp;
        *(min + 2) = minBTemp;
        *(max + 2) = maxBTemp;
    }
}

inline void compute_histogram_location_host(Rpp32u *batch_bins, int batchCount, Rpp32u *locHist)
{
    for (int m = 0; m < batchCount; m++)
    {
        *locHist += batch_bins[m];
    }
}

template <typename T>
inline void compute_unpadded_from_padded_host(T* srcPtrPadded, RppiSize srcSize, RppiSize srcSizeMax, T* dstPtrUnpadded,
                                                   RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrPaddedChannel, *srcPtrPaddedRow, *dstPtrUnpaddedRow;
    Rpp32u imageDimMax = srcSizeMax.height * srcSizeMax.width;
    dstPtrUnpaddedRow = dstPtrUnpadded;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrPaddedChannel = srcPtrPadded + (c * imageDimMax);
            for (int i = 0; i < srcSize.height; i++)
            {
                srcPtrPaddedRow = srcPtrPaddedChannel + (i * srcSizeMax.width);
                memcpy(dstPtrUnpaddedRow, srcPtrPaddedRow, srcSize.width * sizeof(T));
                dstPtrUnpaddedRow += srcSize.width;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRowMax = channel * srcSizeMax.width;
        Rpp32u elementsInRow = channel * srcSize.width;
        for (int i = 0; i < srcSize.height; i++)
        {
            srcPtrPaddedRow = srcPtrPadded + (i * elementsInRowMax);
            memcpy(dstPtrUnpaddedRow, srcPtrPaddedRow, elementsInRow * sizeof(T));
            dstPtrUnpaddedRow += elementsInRow;
        }
    }
}

template <typename T>
inline void compute_padded_from_unpadded_host(T* srcPtrUnpadded, RppiSize srcSize, RppiSize dstSizeMax, T* dstPtrPadded,
                                                   RppiChnFormat chnFormat, Rpp32u channel)
{
    T *dstPtrPaddedChannel, *dstPtrPaddedRow, *srcPtrUnpaddedRow;
    Rpp32u imageDimMax = dstSizeMax.height * dstSizeMax.width;
    srcPtrUnpaddedRow = srcPtrUnpadded;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            dstPtrPaddedChannel = dstPtrPadded + (c * imageDimMax);
            for (int i = 0; i < srcSize.height; i++)
            {
                dstPtrPaddedRow = dstPtrPaddedChannel + (i * dstSizeMax.width);
                memcpy(dstPtrPaddedRow, srcPtrUnpaddedRow, srcSize.width * sizeof(T));
                srcPtrUnpaddedRow += srcSize.width;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRowMax = channel * dstSizeMax.width;
        Rpp32u elementsInRow = channel * srcSize.width;
        for (int i = 0; i < srcSize.height; i++)
        {
            dstPtrPaddedRow = dstPtrPadded + (i * elementsInRowMax);
            memcpy(dstPtrPaddedRow, srcPtrUnpaddedRow, elementsInRow * sizeof(T));
            srcPtrUnpaddedRow += elementsInRow;
        }
    }
}

template <typename T>
inline void compute_planar_to_packed_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                        Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    for (int c = 0; c < channel; c++)
    {
        dstPtrTemp += c;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                *dstPtrTemp = *srcPtrTemp;
                srcPtrTemp++;
                dstPtrTemp += 3;
            }
        }
        dstPtrTemp = dstPtr;
    }
}

template <typename T>
inline void compute_packed_to_planar_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                        Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    for (int c = 0; c < channel; c++)
    {
        srcPtrTemp += c;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                *dstPtrTemp = *srcPtrTemp;
                dstPtrTemp++;
                srcPtrTemp += 3;
            }
        }
        srcPtrTemp = srcPtr;
    }
}

/* Generic interpolation helper functions */

template <typename T>
inline void compute_generic_bilinear_srclocs_and_interpolate(T *srcPtrChannel, RpptDescPtr srcDescPtr, Rpp32f &srcY, Rpp32f &srcX, RpptROI* roiLTRB, T *dst)
{
    RppiPoint srcLT, srcRB;
    Rpp32f weightParams[4], bilinearCoeffs[4];
    Rpp32s srcLoc[4];
    srcLT.y = (Rpp32s) srcY;                                    // Bilinear LT point y value
    srcLT.y = std::min(srcLT.y, roiLTRB->ltrbROI.rb.y - 1);
    srcRB.y = std::min(srcLT.y + 1, roiLTRB->ltrbROI.rb.y - 1); // Bilinear RB point y value
    srcLT.x = (Rpp32s) srcX;                                    // Bilinear LT point x value
    srcLT.x = std::min(srcLT.x, roiLTRB->ltrbROI.rb.x - 1);
    srcRB.x = std::min(srcLT.x + 1, roiLTRB->ltrbROI.rb.x - 1); // Bilinear RB point x value
    weightParams[0] = srcY - srcLT.y;                           // weightedHeight
    weightParams[1] = 1 - weightParams[0];                      // 1 - weightedHeight
    weightParams[2] = srcX - srcLT.x;                           // weightedWidth
    weightParams[3] = 1 - weightParams[2];                      // 1 - weightedWidth
    bilinearCoeffs[0] = weightParams[1] * weightParams[3];      // (1 - weightedHeight) * (1 - weightedWidth)
    bilinearCoeffs[1] = weightParams[1] * weightParams[2];      // (1 - weightedHeight) * weightedWidth
    bilinearCoeffs[2] = weightParams[0] * weightParams[3];      // weightedHeight * (1 - weightedWidth)
    bilinearCoeffs[3] = weightParams[0] * weightParams[2];      // weightedHeight * weightedWidth
    srcLT.y *= srcDescPtr->strides.hStride;                     // LT Row * hStride
    srcRB.y *= srcDescPtr->strides.hStride;                     // RB Row * hStride
    srcLT.x *= srcDescPtr->strides.wStride;                     // LT Col * wStride
    srcRB.x *= srcDescPtr->strides.wStride;                     // LT Col * wStride
    srcLoc[0] = srcLT.y + srcLT.x;                              // Left-Top pixel memory location
    srcLoc[1] = srcLT.y + srcRB.x;                              // Right-Top pixel memory location
    srcLoc[2] = srcRB.y + srcLT.x;                              // Left-Bottom pixel memory location
    srcLoc[3] = srcRB.y + srcRB.x;                              // Right-Bottom pixel memory location

    for (int c = 0; c < srcDescPtr->c; c++)
    {
        dst[c] = (T)std::nearbyintf(((*(srcPtrChannel + srcLoc[0]) * bilinearCoeffs[0]) +        // TopRow R01 Pixel * coeff0
                    (*(srcPtrChannel + srcLoc[1]) * bilinearCoeffs[1]) +        // TopRow R02 Pixel * coeff1
                    (*(srcPtrChannel + srcLoc[2]) * bilinearCoeffs[2]) +        // BottomRow R01 Pixel * coeff2
                    (*(srcPtrChannel + srcLoc[3]) * bilinearCoeffs[3])));        // BottomRow R02 Pixel * coeff3
        srcPtrChannel += srcDescPtr->strides.cStride;
    }
}

inline void compute_generic_bilinear_srclocs_1c_avx(__m256 &pSrcY, __m256 &pSrcX, RpptBilinearNbhoodLocsVecLen8 &srcLocs, __m256 *pBilinearCoeffs, __m256 &pSrcStrideH, __m256i *pxSrcStridesCHW, __m256 *pRoiLTRB)
{
    __m256 pWeightParams[4], pSrcBilinearLTyx[4];
    pSrcBilinearLTyx[0] = _mm256_floor_ps(pSrcY);                               // srcLT->y = (Rpp32s) srcY;
    pSrcBilinearLTyx[1] = _mm256_floor_ps(pSrcX);                               // srcLT->x = (Rpp32s) srcX;
    pWeightParams[0] = _mm256_sub_ps(pSrcY, pSrcBilinearLTyx[0]);               // weightParams[0] = srcY - srcLT->y;
    pWeightParams[1] = _mm256_sub_ps(avx_p1, pWeightParams[0]);                 // weightParams[1] = 1 - weightParams[0];
    pWeightParams[2] = _mm256_sub_ps(pSrcX, pSrcBilinearLTyx[1]);               // weightParams[2] = srcX - srcLT->x;
    pWeightParams[3] = _mm256_sub_ps(avx_p1, pWeightParams[2]);                 // weightParams[3] = 1 - weightParams[2]
    pBilinearCoeffs[0] = _mm256_mul_ps(pWeightParams[1], pWeightParams[3]);     // (1 - weightedHeight) * (1 - weightedWidth)
    pBilinearCoeffs[1] = _mm256_mul_ps(pWeightParams[1], pWeightParams[2]);     // (1 - weightedHeight) * weightedWidth
    pBilinearCoeffs[2] = _mm256_mul_ps(pWeightParams[0], pWeightParams[3]);     // weightedHeight * (1 - weightedWidth)
    pBilinearCoeffs[3] = _mm256_mul_ps(pWeightParams[0], pWeightParams[2]);     // weightedHeight * weightedWidth
    pSrcBilinearLTyx[0] = _mm256_min_ps(_mm256_max_ps(pSrcBilinearLTyx[0], pRoiLTRB[1]), _mm256_sub_ps(pRoiLTRB[3], avx_p1));
    pSrcBilinearLTyx[1] = _mm256_min_ps(_mm256_max_ps(pSrcBilinearLTyx[1], pRoiLTRB[0]), _mm256_sub_ps(pRoiLTRB[2], avx_p1));
    pSrcBilinearLTyx[2] = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(pSrcBilinearLTyx[0], avx_p1), pRoiLTRB[1]), _mm256_sub_ps(pRoiLTRB[3], avx_p1));
    pSrcBilinearLTyx[3] = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(pSrcBilinearLTyx[1], avx_p1), pRoiLTRB[0]), _mm256_sub_ps(pRoiLTRB[2], avx_p1));
    __m256i pxSrcLocsTL =  _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[0], pSrcStrideH, pSrcBilinearLTyx[1]));     // 8 Top-Left memory locations = 8 Top-Left srcYs * hStride + 8 Top-Left srcXs
    __m256i pxSrcLocsTR =  _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[0], pSrcStrideH, pSrcBilinearLTyx[3]));     // 8 Top-Right memory locations = 8 Top-Left srcYs * hStride + 8 Bottom-right srcXs
    __m256i pxSrcLocsBL = _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[2], pSrcStrideH, pSrcBilinearLTyx[1]));      // 8 Bottom-Left memory locations = 8 Bottom-right srcYs * hStride + 8 Top-Left srcXs
    __m256i pxSrcLocsBR = _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[2], pSrcStrideH, pSrcBilinearLTyx[3]));      // 8 Bottom-Right memory locations = 8 Bottom-right srcYs * hStride + 8 Bottom-right srcXs
    _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsTL.data[0], pxSrcLocsTL);    // Store precomputed bilinear Top-Left locations
    _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsTR.data[0], pxSrcLocsTR);    // Store precomputed bilinear Top-Right locations
    _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsBL.data[0], pxSrcLocsBL);    // Store precomputed bilinear Bottom-Left locations
    _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsBR.data[0], pxSrcLocsBR);    // Store precomputed bilinear Bottom-Right locations
}

inline void compute_generic_bilinear_srclocs_3c_avx(__m256 &pSrcY, __m256 &pSrcX, RpptBilinearNbhoodLocsVecLen8 &srcLocs, __m256 *pBilinearCoeffs, __m256 &pSrcStrideH, __m256i *pxSrcStridesCHW, Rpp32s srcChannels, __m256 *pRoiLTRB, bool isSrcPKD3 = false)
{
    __m256 pWeightParams[4], pSrcBilinearLTyx[4];
    pSrcBilinearLTyx[0] = _mm256_floor_ps(pSrcY);                               // srcLT->y = (Rpp32s) srcY;
    pSrcBilinearLTyx[1] = _mm256_floor_ps(pSrcX);                               // srcLT->x = (Rpp32s) srcX;
    pWeightParams[0] = _mm256_sub_ps(pSrcY, pSrcBilinearLTyx[0]);               // weightParams[0] = srcY - srcLT->y;
    pWeightParams[1] = _mm256_sub_ps(avx_p1, pWeightParams[0]);                 // weightParams[1] = 1 - weightParams[0];
    pWeightParams[2] = _mm256_sub_ps(pSrcX, pSrcBilinearLTyx[1]);               // weightParams[2] = srcX - srcLT->x;
    pWeightParams[3] = _mm256_sub_ps(avx_p1, pWeightParams[2]);                 // weightParams[3] = 1 - weightParams[2]
    pBilinearCoeffs[0] = _mm256_mul_ps(pWeightParams[1], pWeightParams[3]);     // (1 - weightedHeight) * (1 - weightedWidth)
    pBilinearCoeffs[1] = _mm256_mul_ps(pWeightParams[1], pWeightParams[2]);     // (1 - weightedHeight) * weightedWidth
    pBilinearCoeffs[2] = _mm256_mul_ps(pWeightParams[0], pWeightParams[3]);     // weightedHeight * (1 - weightedWidth)
    pBilinearCoeffs[3] = _mm256_mul_ps(pWeightParams[0], pWeightParams[2]);     // weightedHeight * weightedWidth
    pSrcBilinearLTyx[0] = _mm256_min_ps(_mm256_max_ps(pSrcBilinearLTyx[0], pRoiLTRB[1]), _mm256_sub_ps(pRoiLTRB[3], avx_p1));
    pSrcBilinearLTyx[1] = _mm256_min_ps(_mm256_max_ps(pSrcBilinearLTyx[1], pRoiLTRB[0]), _mm256_sub_ps(pRoiLTRB[2], avx_p1));
    pSrcBilinearLTyx[2] = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(pSrcBilinearLTyx[0], avx_p1), pRoiLTRB[1]), _mm256_sub_ps(pRoiLTRB[3], avx_p1));
    pSrcBilinearLTyx[3] = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(pSrcBilinearLTyx[1], avx_p1), pRoiLTRB[0]), _mm256_sub_ps(pRoiLTRB[2], avx_p1));
    if(isSrcPKD3)
    {
        pSrcBilinearLTyx[1] = _mm256_mul_ps(pSrcBilinearLTyx[1], avx_p3);       // if pkd3, multiply Left-Top column location by 3
        pSrcBilinearLTyx[3] = _mm256_mul_ps(pSrcBilinearLTyx[3], avx_p3);       // if pkd3, multiply Right-Top column location by 3
    }
    __m256i pxSrcLocsTL =  _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[0], pSrcStrideH, pSrcBilinearLTyx[1]));    // 8 Top-Left memory locations = 8 Top-Left srcYs * hStride + 8 Top-Left srcXs
    __m256i pxSrcLocsTR =  _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[0], pSrcStrideH, pSrcBilinearLTyx[3]));    // 8 Top-Right memory locations = 8 Top-Left srcYs * hStride + 8 Bottom-right srcXs
    __m256i pxSrcLocsBL = _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[2], pSrcStrideH, pSrcBilinearLTyx[1]));     // 8 Bottom-Left memory locations = 8 Bottom-right srcYs * hStride + 8 Top-Left srcXs
    __m256i pxSrcLocsBR = _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[2], pSrcStrideH, pSrcBilinearLTyx[3]));     // 8 Bottom-Right memory locations = 8 Bottom-right srcYs * hStride + 8 Bottom-right srcXs
    for (int c = 0; c < srcChannels * 8; c += 8)
    {
        _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsTL.data[c], pxSrcLocsTL);    // Store precomputed bilinear Top-Left locations
        _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsTR.data[c], pxSrcLocsTR);    // Store precomputed bilinear Top-Right locations
        _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsBL.data[c], pxSrcLocsBL);    // Store precomputed bilinear Bottom-Left locations
        _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsBR.data[c], pxSrcLocsBR);    // Store precomputed bilinear Bottom-Right locations
        pxSrcLocsTL = _mm256_add_epi32(pxSrcLocsTL, pxSrcStridesCHW[0]);            // Increment Top-Left locations by cStride
        pxSrcLocsTR = _mm256_add_epi32(pxSrcLocsTR, pxSrcStridesCHW[0]);            // Increment Top-Right locations by cStride
        pxSrcLocsBL = _mm256_add_epi32(pxSrcLocsBL, pxSrcStridesCHW[0]);            // Increment Bottom-Left locations by cStride
        pxSrcLocsBR = _mm256_add_epi32(pxSrcLocsBR, pxSrcStridesCHW[0]);            // Increment Bottom-Right locations by cStride
    }
}

template <typename T>
inline void compute_generic_bilinear_interpolation_pkd3_to_pln3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTempR, T *dstPtrTempG, T *dstPtrTempB, T *srcPtrChannel, RpptDescPtr srcDescPtr)
{
    if ((srcX < roiLTRB->ltrbROI.lt.x) || (srcY < roiLTRB->ltrbROI.lt.y) || (srcX > roiLTRB->ltrbROI.rb.x) || (srcY > roiLTRB->ltrbROI.rb.y))
    {
        *dstPtrTempR = 0;
        *dstPtrTempG = 0;
        *dstPtrTempB = 0;
    }
    else
    {
        T dst[3];
        compute_generic_bilinear_srclocs_and_interpolate(srcPtrChannel, srcDescPtr, srcY, srcX, roiLTRB, dst);
        *dstPtrTempR = dst[0];
        *dstPtrTempG = dst[1];
        *dstPtrTempB = dst[2];
    }
}

template <typename T>
inline void compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTemp, T *srcPtrChannel, RpptDescPtr srcDescPtr)
{
    Rpp32s srcXFloor = std::floor(srcX);
    Rpp32s srcYFloor = std::floor(srcY);
    if ((srcXFloor < roiLTRB->ltrbROI.lt.x) || (srcYFloor < roiLTRB->ltrbROI.lt.y) || (srcXFloor > roiLTRB->ltrbROI.rb.x) || (srcYFloor > roiLTRB->ltrbROI.rb.y))
    {
        memset(dstPtrTemp, 0, 3 * sizeof(T));
    }
    else
    {
        compute_generic_bilinear_srclocs_and_interpolate(srcPtrChannel, srcDescPtr, srcY, srcX, roiLTRB, dstPtrTemp);
    }
}

template <typename T>
inline void compute_generic_bilinear_interpolation_pln_to_pln(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTemp, T *srcPtrChannel, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    Rpp32s srcXFloor = std::floor(srcX);
    Rpp32s srcYFloor = std::floor(srcY);
    if ((srcXFloor < roiLTRB->ltrbROI.lt.x) || (srcYFloor < roiLTRB->ltrbROI.lt.y) || (srcXFloor > roiLTRB->ltrbROI.rb.x) || (srcYFloor > roiLTRB->ltrbROI.rb.y))
    {
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            *dstPtrTemp = 0;
            dstPtrTemp += dstDescPtr->strides.cStride;
        }
    }
    else
    {
        T dst[3];
        compute_generic_bilinear_srclocs_and_interpolate(srcPtrChannel, srcDescPtr, srcY, srcX, roiLTRB, dst);
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            *dstPtrTemp = dst[c];
            dstPtrTemp += dstDescPtr->strides.cStride;
        }
    }
}

inline void compute_generic_nn_srclocs_and_validate_sse(__m128 pSrcY, __m128 pSrcX, __m128 *pRoiLTRB, __m128 pSrcStrideH, Rpp32s *srcLoc, Rpp32s *invalidLoad, bool hasRGBChannels = false)
{
    pSrcY = _mm_round_ps(pSrcY, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));        // Nearest Neighbor Y location vector
    pSrcX = _mm_round_ps(pSrcX, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));        // Nearest Neighbor X location vector
    _mm_storeu_si128((__m128i*) invalidLoad, _mm_cvtps_epi32(_mm_or_ps(                 // Vectorized ROI boundary check
        _mm_or_ps(_mm_cmplt_ps(pSrcX, pRoiLTRB[0]), _mm_cmplt_ps(pSrcY, pRoiLTRB[1])),
        _mm_or_ps(_mm_cmpgt_ps(pSrcX, pRoiLTRB[2]), _mm_cmpgt_ps(pSrcY, pRoiLTRB[3]))
    )));
    if (hasRGBChannels)
        pSrcX = _mm_mul_ps(pSrcX, xmm_p3);
    __m128i pxSrcLoc = _mm_cvtps_epi32(_mm_fmadd_ps(pSrcY, pSrcStrideH, pSrcX));
    _mm_storeu_si128((__m128i*) srcLoc, pxSrcLoc);
}

inline void compute_generic_nn_srclocs_and_validate_avx(__m256 pSrcY, __m256 pSrcX, __m256 *pRoiLTRB, __m256 pSrcStrideH, Rpp32s *srcLoc, Rpp32s *invalidLoad, bool hasRGBChannels = false)
{
    pSrcY = _mm256_round_ps(pSrcY, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));              // Nearest Neighbor Y location vector
    pSrcX = _mm256_round_ps(pSrcX, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));              // Nearest Neighbor X location vector
    _mm256_storeu_si256((__m256i*) invalidLoad, _mm256_cvtps_epi32(_mm256_or_ps(                 // Vectorized ROI boundary check
        _mm256_or_ps(_mm256_cmp_ps(pSrcX, pRoiLTRB[0], _CMP_LT_OQ), _mm256_cmp_ps(pSrcY, pRoiLTRB[1],_CMP_LT_OQ)),
        _mm256_or_ps(_mm256_cmp_ps(pSrcX, pRoiLTRB[2], _CMP_GT_OQ), _mm256_cmp_ps(pSrcY, pRoiLTRB[3], _CMP_GT_OQ))
    )));
    if (hasRGBChannels)
        pSrcX = _mm256_mul_ps(pSrcX, avx_p3);
    __m256i pxSrcLoc = _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcY, pSrcStrideH, pSrcX));
    _mm256_storeu_si256((__m256i*) srcLoc, pxSrcLoc);
}

template <typename T>
inline void compute_generic_nn_interpolation_pkd3_to_pln3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTempR, T *dstPtrTempG, T *dstPtrTempB, T *srcPtrChannel, RpptDescPtr srcDescPtr)
{
    srcY = std::round(srcY);    // Nearest Neighbor Y location
    srcX = std::round(srcX);    // Nearest Neighbor X location
    if ((srcX < roiLTRB->ltrbROI.lt.x) || (srcY < roiLTRB->ltrbROI.lt.y) || (srcX > roiLTRB->ltrbROI.rb.x) || (srcY > roiLTRB->ltrbROI.rb.y))
    {
        *dstPtrTempR = 0;
        *dstPtrTempG = 0;
        *dstPtrTempB = 0;
    }
    else
    {
        T *srcPtrTemp;
        srcPtrTemp = srcPtrChannel + ((Rpp32s)srcY * srcDescPtr->strides.hStride) + ((Rpp32s)srcX * srcDescPtr->strides.wStride);
        *dstPtrTempR = *srcPtrTemp++;
        *dstPtrTempG = *srcPtrTemp++;
        *dstPtrTempB = *srcPtrTemp;
    }
}

template <typename T>
inline void compute_generic_nn_interpolation_pkd3_to_pkd3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTemp, T *srcPtrChannel, RpptDescPtr srcDescPtr)
{
    srcY = std::round(srcY);    // Nearest Neighbor Y location
    srcX = std::round(srcX);    // Nearest Neighbor X location
    if ((srcX < roiLTRB->ltrbROI.lt.x) || (srcY < roiLTRB->ltrbROI.lt.y) || (srcX > roiLTRB->ltrbROI.rb.x) || (srcY > roiLTRB->ltrbROI.rb.y))
    {
        memset(dstPtrTemp, 0, 3 * sizeof(T));
    }
    else
    {
        T *srcPtrTemp;
        srcPtrTemp = srcPtrChannel + ((Rpp32s)srcY * srcDescPtr->strides.hStride) + ((Rpp32s)srcX * srcDescPtr->strides.wStride);
        memcpy(dstPtrTemp, srcPtrTemp, 3 * sizeof(T));
    }
}

template <typename T>
inline void compute_generic_nn_interpolation_pln3_to_pkd3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTemp, T *srcPtrChannel, RpptDescPtr srcDescPtr)
{
    srcY = std::round(srcY);    // Nearest Neighbor Y location
    srcX = std::round(srcX);    // Nearest Neighbor X location
    if ((srcX < roiLTRB->ltrbROI.lt.x) || (srcY < roiLTRB->ltrbROI.lt.y) || (srcX > roiLTRB->ltrbROI.rb.x) || (srcY > roiLTRB->ltrbROI.rb.y))
    {
        memset(dstPtrTemp, 0, 3 * sizeof(T));
    }
    else
    {
        T *srcPtrTemp;
        srcPtrTemp = srcPtrChannel + ((Rpp32s)srcY * srcDescPtr->strides.hStride) + ((Rpp32s)srcX * srcDescPtr->strides.wStride);
        *dstPtrTemp++ = *srcPtrTemp;
        srcPtrTemp += srcDescPtr->strides.cStride;
        *dstPtrTemp++ = *srcPtrTemp;
        srcPtrTemp += srcDescPtr->strides.cStride;
        *dstPtrTemp = *srcPtrTemp;
    }
}

template <typename T>
inline void compute_generic_nn_interpolation_pln_to_pln(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTemp, T *srcPtrChannel, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    srcY = std::round(srcY);    // Nearest Neighbor Y location
    srcX = std::round(srcX);    // Nearest Neighbor X location
    if ((srcX < roiLTRB->ltrbROI.lt.x) || (srcY < roiLTRB->ltrbROI.lt.y) || (srcX > roiLTRB->ltrbROI.rb.x) || (srcY > roiLTRB->ltrbROI.rb.y))
    {
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            *dstPtrTemp = 0;
            dstPtrTemp += dstDescPtr->strides.cStride;
        }
    }
    else
    {
        T *srcPtrTemp;
        srcPtrTemp = srcPtrChannel + ((Rpp32s)srcY * srcDescPtr->strides.hStride) + ((Rpp32s)srcX * srcDescPtr->strides.wStride);
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            *dstPtrTemp = *srcPtrTemp;
            srcPtrTemp += srcDescPtr->strides.cStride;
            dstPtrTemp += dstDescPtr->strides.cStride;
        }
    }
}

/* Resize helper functions */
inline void compute_dst_size_cap_host(RpptImagePatchPtr dstImgSize, RpptDescPtr dstDescPtr)
{
    dstImgSize->width = std::min(dstImgSize->width, dstDescPtr->w);
    dstImgSize->height = std::min(dstImgSize->height, dstDescPtr->h);
}

inline void compute_resize_src_loc(Rpp32s dstLocation, Rpp32f scale, Rpp32s &srcLoc, Rpp32f &weight, Rpp32f offset = 0, Rpp32u srcStride = 1)
{
    Rpp32f srcLocationFloat = ((Rpp32f) dstLocation) * scale + offset;
    Rpp32s srcLocation = (Rpp32s) std::ceil(srcLocationFloat);
    weight = srcLocation - srcLocationFloat;
    srcLoc = srcLocation * srcStride;
}

inline void compute_resize_nn_src_loc(Rpp32s dstLocation, Rpp32f scale, Rpp32u limit, Rpp32s &srcLoc, Rpp32f offset = 0, Rpp32u srcStride = 1)
{
    Rpp32f srcLocation = ((Rpp32f) dstLocation) * scale + offset;
    Rpp32s srcLocationFloor = (Rpp32s) RPPFLOOR(srcLocation);
    srcLoc = ((srcLocationFloor > limit) ? limit : srcLocationFloor) * srcStride;
}

inline void compute_resize_bilinear_src_loc_and_weights(Rpp32s dstLocation, Rpp32f scale, Rpp32s &srcLoc, Rpp32f *weight, Rpp32f offset = 0, Rpp32u srcStride = 1)
{
    compute_resize_src_loc(dstLocation, scale, srcLoc, weight[1], offset, srcStride);
    weight[0] = 1 - weight[1];
}

inline void compute_resize_nn_src_loc_sse(__m128 &pDstLoc, __m128 &pScale, __m128 &pLimit, Rpp32s *srcLoc, __m128 pOffset = xmm_p0, bool hasRGBChannels = false)
{
    __m128 pLoc = _mm_fmadd_ps(pDstLoc, pScale, pOffset);
    pDstLoc = _mm_add_ps(pDstLoc, xmm_p4);
    __m128 pLocFloor = _mm_floor_ps(pLoc);
    pLocFloor = _mm_max_ps(_mm_min_ps(pLocFloor, pLimit), xmm_p0);
    if(hasRGBChannels)
        pLocFloor = _mm_mul_ps(pLocFloor, xmm_p3);
    __m128i pxLocFloor = _mm_cvtps_epi32(pLocFloor);
    _mm_storeu_si128((__m128i*) srcLoc, pxLocFloor);
}

inline void compute_resize_bilinear_src_loc_and_weights_avx(__m256 &pDstLoc, __m256 &pScale, Rpp32s *srcLoc, __m256 *pWeight, __m256i &pxLoc, __m256 pOffset = avx_p0, bool hasRGBChannels = false)
{
    __m256 pLocFloat = _mm256_fmadd_ps(pDstLoc, pScale, pOffset);
    pDstLoc = _mm256_add_ps(pDstLoc, avx_p8);
    __m256 pLoc = _mm256_ceil_ps(pLocFloat);
    pWeight[1] = _mm256_sub_ps(pLoc, pLocFloat);
    pWeight[0] = _mm256_sub_ps(avx_p1, pWeight[1]);
    if(hasRGBChannels)
        pLoc = _mm256_mul_ps(pLoc, avx_p3);
    pxLoc = _mm256_cvtps_epi32(pLoc);
    _mm256_storeu_si256((__m256i*) srcLoc, pxLoc);
}

inline void compute_resize_bilinear_src_loc_and_weights_mirror_avx(__m256 &pDstLoc, __m256 &pScale, Rpp32s *srcLoc, __m256 *pWeight, __m256i &pxLoc, __m256 pOffset = avx_p0, bool hasRGBChannels = false)
{
    __m256 pLocFloat = _mm256_fmadd_ps(pDstLoc, pScale, pOffset);
    pDstLoc = _mm256_sub_ps(pDstLoc, avx_p8);
    __m256 pLoc = _mm256_ceil_ps(pLocFloat);
    pWeight[1] = _mm256_sub_ps(pLoc, pLocFloat);
    pWeight[0] = _mm256_sub_ps(avx_p1, pWeight[1]);
    if (hasRGBChannels)
        pLoc = _mm256_mul_ps(pLoc, avx_p3);
    pxLoc = _mm256_cvtps_epi32(pLoc);
    _mm256_storeu_si256((__m256i *)srcLoc, pxLoc);
}

inline void compute_bicubic_coefficient(Rpp32f weight, Rpp32f &coeff)
{
    Rpp32f x = fabsf(weight);
    coeff = (x >= 2) ? 0 : ((x > 1) ? (x * x * (-0.5f * x + 2.5f) - 4.0f * x + 2.0f) : (x * x * (1.5f * x - 2.5f) + 1.0f));
}

inline Rpp32f sinc(Rpp32f x)
{
    x *= M_PI;
    return (std::abs(x) < 1e-5f) ? (1.0f - x * x * ONE_OVER_6) : std::sin(x) / x;
}

inline void compute_lanczos3_coefficient(Rpp32f weight, Rpp32f &coeff)
{
    coeff = fabs(weight) >= 3 ? 0.0f : (sinc(weight) * sinc(weight * 0.333333f));
}

inline void compute_gaussian_coefficient(Rpp32f weight, Rpp32f &coeff)
{
    coeff = expf(weight * weight * -4.0f);
}

inline void compute_triangular_coefficient(Rpp32f weight, Rpp32f &coeff)
{
    coeff = 1 - std::fabs(weight);
    coeff = coeff < 0 ? 0 : coeff;
}

inline void compute_coefficient(RpptInterpolationType interpolationType, Rpp32f weight, Rpp32f &coeff)
{
    switch (interpolationType)
    {
    case RpptInterpolationType::BICUBIC:
    {
        compute_bicubic_coefficient(weight, coeff);
        break;
    }
    case RpptInterpolationType::LANCZOS:
    {
        compute_lanczos3_coefficient(weight, coeff);
        break;
    }
    case RpptInterpolationType::GAUSSIAN:
    {
        compute_gaussian_coefficient(weight, coeff);
        break;
    }
    case RpptInterpolationType::TRIANGULAR:
    {
        compute_triangular_coefficient(weight, coeff);
        break;
    }
    default:
        break;
    }
}

// Computes the row coefficients for separable resampling
inline void compute_row_coefficients(RpptInterpolationType interpolationType, GenericFilter &filter , Rpp32f weight, Rpp32f *coeffs, Rpp32u srcStride = 1)
{
    Rpp32f sum = 0;
    weight = weight - filter.radius;
    for(int k = 0; k < filter.size; k++)
    {
        compute_coefficient(interpolationType, (weight + k) * filter.scale, coeffs[k]);
        sum += coeffs[k];
    }
    if(sum)
    {
        sum = 1 / sum;
        for(int k = 0; k < filter.size; k++)
            coeffs[k] = coeffs[k] * sum;
    }
}

// Computes the column coefficients for separable resampling
inline void compute_col_coefficients(RpptInterpolationType interpolationType, GenericFilter &filter, Rpp32f weight, Rpp32f *coeffs, Rpp32u srcStride = 1)
{
    Rpp32f sum = 0;
    weight = weight - filter.radius;

    // The coefficients are computed for 4 dst locations and stored consecutively for ease of access
    for(int k = 0, kPos = 0; k < filter.size; k++, kPos += 4)
    {
        compute_coefficient(interpolationType, (weight + k) * filter.scale, coeffs[kPos]);
        sum += coeffs[kPos];
    }
    if(sum)
    {
        sum = 1 / sum;
        for(int k = 0, kPos = 0; k < filter.size; k++, kPos += 4)
            coeffs[kPos] = coeffs[kPos] * sum;
    }
}

inline void set_zeros(__m128 *pVecs, Rpp32s numVecs)
{
    for(int i = 0; i < numVecs; i++)
        pVecs[i] = xmm_p0;
}

inline void set_zeros_avx(__m256 *pVecs, Rpp32s numVecs)
{
    for(int i = 0; i < numVecs; i++)
        pVecs[i] = avx_p0;
}

inline void compute_bilinear_coefficients(Rpp32f *weightParams, Rpp32f *bilinearCoeffs)
{
    bilinearCoeffs[0] = weightParams[1] * weightParams[3];    // (1 - weightedHeight) * (1 - weightedWidth)
    bilinearCoeffs[1] = weightParams[1] * weightParams[2];    // (1 - weightedHeight) * weightedWidth
    bilinearCoeffs[2] = weightParams[0] * weightParams[3];    // weightedHeight * (1 - weightedWidth)
    bilinearCoeffs[3] = weightParams[0] * weightParams[2];    // weightedHeight * weightedWidth
}

inline void compute_bilinear_coefficients_avx(__m256 *pWeightParams, __m256 *pBilinearCoeffs)
{
    pBilinearCoeffs[0] = _mm256_mul_ps(pWeightParams[1], pWeightParams[3]);    // (1 - weightedHeight) * (1 - weightedWidth)
    pBilinearCoeffs[1] = _mm256_mul_ps(pWeightParams[1], pWeightParams[2]);    // (1 - weightedHeight) * weightedWidth
    pBilinearCoeffs[2] = _mm256_mul_ps(pWeightParams[0], pWeightParams[3]);    // weightedHeight * (1 - weightedWidth)
    pBilinearCoeffs[3] = _mm256_mul_ps(pWeightParams[0], pWeightParams[2]);    // weightedHeight * weightedWidth
}

template <typename T, typename U>
inline void compute_bilinear_interpolation_1c(T **srcRowPtrsForInterp, Rpp32s loc, Rpp32s limit, Rpp32f *bilinearCoeffs, U *dstPtr)
{
    Rpp32s loc1 = std::min(std::max(loc, 0), limit);
    Rpp32s loc2 = std::min(std::max(loc + 1, 0), limit);
    *dstPtr = (U)(((*(srcRowPtrsForInterp[0] + loc1)) * bilinearCoeffs[0]) +     // TopRow 1st Pixel * coeff0
                  ((*(srcRowPtrsForInterp[0] + loc2)) * bilinearCoeffs[1]) +     // TopRow 2nd Pixel * coeff1
                  ((*(srcRowPtrsForInterp[1] + loc1)) * bilinearCoeffs[2]) +     // BottomRow 1st Pixel * coeff2
                  ((*(srcRowPtrsForInterp[1] + loc2)) * bilinearCoeffs[3]));    // BottomRow 2nd Pixel * coeff3
}

template <typename T, typename U>
inline void compute_bilinear_interpolation_3c_pkd(T **srcRowPtrsForInterp, Rpp32s loc, Rpp32s limit, Rpp32f *bilinearCoeffs, U *dstPtrR, U *dstPtrG, U *dstPtrB)
{
    Rpp32s loc1 = std::min(std::max(loc, 0), limit);
    Rpp32s loc2 = std::min(std::max(loc + 3, 0), limit);
    *dstPtrR = (U)(((*(srcRowPtrsForInterp[0] + loc1)) * bilinearCoeffs[0]) +        // TopRow R01 Pixel * coeff0
                   ((*(srcRowPtrsForInterp[0] + loc2)) * bilinearCoeffs[1]) +        // TopRow R02 Pixel * coeff1
                   ((*(srcRowPtrsForInterp[1] + loc1)) * bilinearCoeffs[2]) +        // BottomRow R01 Pixel * coeff2
                   ((*(srcRowPtrsForInterp[1] + loc2)) * bilinearCoeffs[3]));       // BottomRow R02 Pixel * coeff3
    *dstPtrG = (U)(((*(srcRowPtrsForInterp[0] + loc1 + 1)) * bilinearCoeffs[0]) +    // TopRow G01 Pixel * coeff0
                   ((*(srcRowPtrsForInterp[0] + loc2 + 1)) * bilinearCoeffs[1]) +    // TopRow G02 Pixel * coeff1
                   ((*(srcRowPtrsForInterp[1] + loc1 + 1)) * bilinearCoeffs[2]) +    // BottomRow G01 Pixel * coeff2
                   ((*(srcRowPtrsForInterp[1] + loc2 + 1)) * bilinearCoeffs[3]));   // BottomRow G02 Pixel * coeff3
    *dstPtrB = (U)(((*(srcRowPtrsForInterp[0] + loc1 + 2)) * bilinearCoeffs[0]) +    // TopRow B01 Pixel * coeff0
                   ((*(srcRowPtrsForInterp[0] + loc2 + 2)) * bilinearCoeffs[1]) +    // TopRow B02 Pixel * coeff1
                   ((*(srcRowPtrsForInterp[1] + loc1 + 2)) * bilinearCoeffs[2]) +    // BottomRow B01 Pixel * coeff2
                   ((*(srcRowPtrsForInterp[1] + loc2 + 2)) * bilinearCoeffs[3]));   // BottomRow B02 Pixel * coeff3
}

template <typename T, typename U>
inline void compute_bilinear_interpolation_3c_pln(T **srcRowPtrsForInterp, Rpp32s loc, Rpp32s limit, Rpp32f *bilinearCoeffs, U *dstPtrR, U *dstPtrG, U *dstPtrB)
{
    compute_bilinear_interpolation_1c(srcRowPtrsForInterp, loc, limit, bilinearCoeffs, dstPtrR);
    compute_bilinear_interpolation_1c(srcRowPtrsForInterp + 2, loc, limit, bilinearCoeffs, dstPtrG);
    compute_bilinear_interpolation_1c(srcRowPtrsForInterp + 4, loc, limit, bilinearCoeffs, dstPtrB);
}

inline void compute_bilinear_interpolation_1c_avx(__m256 *pSrcPixels, __m256 *pBilinearCoeffs, __m256 &pDstPixels)
{
    pDstPixels = _mm256_fmadd_ps(pSrcPixels[3], pBilinearCoeffs[3], _mm256_fmadd_ps(pSrcPixels[2], pBilinearCoeffs[2],
                 _mm256_fmadd_ps(pSrcPixels[1], pBilinearCoeffs[1], _mm256_mul_ps(pSrcPixels[0], pBilinearCoeffs[0]))));
}

inline void compute_bilinear_interpolation_3c_avx(__m256 *pSrcPixels, __m256 *pBilinearCoeffs, __m256 *pDstPixels)
{
    compute_bilinear_interpolation_1c_avx(pSrcPixels, pBilinearCoeffs, pDstPixels[0]);
    compute_bilinear_interpolation_1c_avx(pSrcPixels + 4, pBilinearCoeffs, pDstPixels[1]);
    compute_bilinear_interpolation_1c_avx(pSrcPixels + 8, pBilinearCoeffs, pDstPixels[2]);
}

template <typename T>
inline void compute_src_row_ptrs_for_bilinear_interpolation(T **rowPtrsForInterp, T *srcPtr, Rpp32s loc, Rpp32s limit, RpptDescPtr descPtr)
{
    rowPtrsForInterp[0] = srcPtr + std::min(std::max(loc, 0), limit) * descPtr->strides.hStride;          // TopRow for bilinear interpolation
    rowPtrsForInterp[1]  = srcPtr + std::min(std::max(loc + 1, 0), limit) * descPtr->strides.hStride;     // BottomRow for bilinear interpolation
}

template <typename T>
inline void compute_src_row_ptrs_for_bilinear_interpolation_pln(T **rowPtrsForInterp, T *srcPtr, Rpp32s loc, Rpp32s limit, RpptDescPtr descPtr)
{
    rowPtrsForInterp[0] = srcPtr + std::min(std::max(loc, 0), limit) * descPtr->strides.hStride;          // TopRow for bilinear interpolation (R channel)
    rowPtrsForInterp[1] = srcPtr + std::min(std::max(loc + 1, 0), limit) * descPtr->strides.hStride;      // BottomRow for bilinear interpolation (R channel)
    rowPtrsForInterp[2] = rowPtrsForInterp[0] + descPtr->strides.cStride;   // TopRow for bilinear interpolation (G channel)
    rowPtrsForInterp[3] = rowPtrsForInterp[1] + descPtr->strides.cStride;   // BottomRow for bilinear interpolation (G channel)
    rowPtrsForInterp[4] = rowPtrsForInterp[2] + descPtr->strides.cStride;   // TopRow for bilinear interpolation (B channel)
    rowPtrsForInterp[5] = rowPtrsForInterp[3] + descPtr->strides.cStride;   // BottomRow for bilinear interpolation (B channel)
}

// Perform resampling along the rows
template <typename T>
inline void compute_separable_vertical_resample(T *inputPtr, Rpp32f *outputPtr, RpptDescPtr inputDescPtr, RpptDescPtr outputDescPtr,
                                                RpptImagePatch inputImgSize, RpptImagePatch outputImgSize, Rpp32s *index, Rpp32f *coeffs, GenericFilter &filter)
{

    static constexpr Rpp32s maxNumLanes = 16;                                  // Maximum number of pixels that can be present in a vector for U8 type
    static constexpr Rpp32s loadLanes = maxNumLanes / sizeof(T);
    static constexpr Rpp32s storeLanes = maxNumLanes / sizeof(Rpp32f);
    static constexpr Rpp32s numLanes = std::max(loadLanes, storeLanes);        // No of pixels that can be present in a vector wrt data type
    static constexpr Rpp32s numVecs = numLanes * sizeof(Rpp32f) / maxNumLanes; // No of float vectors required to process numLanes pixels

    Rpp32s inputHeightLimit = inputImgSize.height - 1;
    Rpp32s outPixelsPerIter = 4;

    // For PLN3 inputs/outputs
    if (inputDescPtr->c == 3 && inputDescPtr->layout == RpptLayout::NCHW)
    {
        T *inRowPtrR[filter.size];
        T *inRowPtrG[filter.size];
        T *inRowPtrB[filter.size];
        for (int outLocRow = 0; outLocRow < outputImgSize.height; outLocRow++)
        {
            Rpp32f *outRowPtrR = outputPtr + outLocRow * outputDescPtr->strides.hStride;
            Rpp32f *outRowPtrG = outRowPtrR + outputDescPtr->strides.cStride;
            Rpp32f *outRowPtrB = outRowPtrG + outputDescPtr->strides.cStride;
            Rpp32s k0 = outLocRow * filter.size;
            __m128 pCoeff[filter.size];

            // Determine the input row pointers and coefficients to be used for interpolation
            for (int k = 0; k < filter.size; k++)
            {
                Rpp32s inLocRow = index[outLocRow] + k;
                inLocRow = std::min(std::max(inLocRow, 0), inputHeightLimit);
                inRowPtrR[k] = inputPtr + inLocRow * inputDescPtr->strides.hStride;
                inRowPtrG[k] = inRowPtrR[k] + inputDescPtr->strides.cStride;
                inRowPtrB[k] = inRowPtrG[k] + inputDescPtr->strides.cStride;
                pCoeff[k] = _mm_set1_ps(coeffs[k0 + k]);    // Each row is associated with a single coeff
            }
            Rpp32s bufferLength = inputImgSize.width;
            Rpp32s alignedLength = bufferLength &~ (numLanes-1);
            Rpp32s outLocCol = 0;

            // Load the input pixels from filter.size rows
            // Multiply input vec from each row with it's correspondig coefficient
            // Add the results from filter.size rows to obtain the pixels of an output row
            for (; outLocCol + numLanes <= alignedLength; outLocCol += numLanes)
            {
                __m128 pTempR[numVecs], pTempG[numVecs], pTempB[numVecs];
                set_zeros(pTempR, numVecs);
                set_zeros(pTempG, numVecs);
                set_zeros(pTempB, numVecs);
                for (int k = 0; k < filter.size; k++)
                {
                    __m128 pInputR[numVecs], pInputG[numVecs], pInputB[numVecs];

                    // Load numLanes input pixels from each row
                    rpp_resize_load(inRowPtrR[k] + outLocCol, pInputR);
                    rpp_resize_load(inRowPtrG[k] + outLocCol, pInputG);
                    rpp_resize_load(inRowPtrB[k] + outLocCol, pInputB);
                    for (int v = 0; v < numVecs; v++)
                    {
                        pTempR[v] = _mm_fmadd_ps(pCoeff[k], pInputR[v], pTempR[v]);
                        pTempG[v] = _mm_fmadd_ps(pCoeff[k], pInputG[v], pTempG[v]);
                        pTempB[v] = _mm_fmadd_ps(pCoeff[k], pInputB[v], pTempB[v]);
                    }
                }
                for(int vec = 0, outStoreStride = 0; vec < numVecs; vec++, outStoreStride += outPixelsPerIter)    // Since 4 output pixels are stored per iteration
                {
                    rpp_simd_store(rpp_store4_f32_to_f32, outRowPtrR + outLocCol + outStoreStride, pTempR + vec);
                    rpp_simd_store(rpp_store4_f32_to_f32, outRowPtrG + outLocCol + outStoreStride, pTempG + vec);
                    rpp_simd_store(rpp_store4_f32_to_f32, outRowPtrB + outLocCol + outStoreStride, pTempB + vec);
                }
            }

            for (; outLocCol < bufferLength; outLocCol++)
            {
                Rpp32f tempR, tempG, tempB;
                tempR = tempG = tempB = 0;
                for (int k = 0; k < filter.size; k++)
                {
                    Rpp32f coefficient = coeffs[k0 + k];
                    tempR += (inRowPtrR[k][outLocCol] * coefficient);
                    tempG += (inRowPtrG[k][outLocCol] * coefficient);
                    tempB += (inRowPtrB[k][outLocCol] * coefficient);
                }
                outRowPtrR[outLocCol] = tempR;
                outRowPtrG[outLocCol] = tempG;
                outRowPtrB[outLocCol] = tempB;
            }
        }
    }
    // For PKD3 and PLN1 inputs/outputs
    else
    {
        T *inRowPtr[filter.size];
        for (int outLocRow = 0; outLocRow < outputImgSize.height; outLocRow++)
        {
            __m128 pCoeff[filter.size];
            Rpp32s k0 = outLocRow * filter.size;
            Rpp32f *outRowPtr = outputPtr + outLocRow * outputDescPtr->strides.hStride;

            // Determine the input row pointers and coefficients to be used for interpolation
            for (int k = 0; k < filter.size; k++)
            {
                Rpp32s inLocRow = index[outLocRow] + k;
                inLocRow = std::min(std::max(inLocRow, 0), inputHeightLimit);
                inRowPtr[k] = inputPtr + inLocRow * inputDescPtr->strides.hStride;
                pCoeff[k] = _mm_set1_ps(coeffs[k0 + k]);    // Each row is associated with a single coeff
            }
            Rpp32s bufferLength = inputImgSize.width * inputDescPtr->strides.wStride;
            Rpp32s alignedLength = bufferLength &~ (numLanes-1);
            Rpp32s outLocCol = 0;

            // Load the input pixels from filter.size rows
            // Multiply input vec from each row with it's correspondig coefficient
            // Add the results from filter.size rows to obtain the pixels of an output row
            for (; outLocCol + numLanes <= alignedLength; outLocCol += numLanes)
            {
                __m128 pTemp[numVecs];
                set_zeros(pTemp, numVecs);
                for (int k = 0; k < filter.size; k++)
                {
                    __m128 pInput[numVecs];
                    rpp_resize_load(inRowPtr[k] + outLocCol, pInput);   // Load numLanes input pixels from each row
                    for (int v = 0; v < numVecs; v++)
                        pTemp[v] = _mm_fmadd_ps(pInput[v], pCoeff[k], pTemp[v]);
                }
                for(int vec = 0, outStoreStride = 0; vec < numVecs; vec++, outStoreStride += outPixelsPerIter)     // Since 4 output pixels are stored per iteration
                    rpp_simd_store(rpp_store4_f32_to_f32, outRowPtr + outLocCol + outStoreStride, &pTemp[vec]);
            }

            for (; outLocCol < bufferLength; outLocCol++)
            {
                Rpp32f temp = 0;
                for (int k = 0; k < filter.size; k++)
                    temp += (inRowPtr[k][outLocCol] * coeffs[k0 + k]);
                outRowPtr[outLocCol] = temp;
            }
        }
    }
}

// Perform resampling along the columns
template <typename T>
inline void compute_separable_horizontal_resample(Rpp32f *inputPtr, T *outputPtr, RpptDescPtr inputDescPtr, RpptDescPtr outputDescPtr,
                        RpptImagePatch inputImgSize, RpptImagePatch outputImgSize, Rpp32s *index, Rpp32f *coeffs, GenericFilter &filter)
{
    static constexpr Rpp32s maxNumLanes = 16;                                   // Maximum number of pixels that can be present in a vector
    static constexpr Rpp32s numLanes = maxNumLanes / sizeof(T);                 // No of pixels that can be present in a vector wrt data type
    static constexpr Rpp32s numVecs = numLanes * sizeof(Rpp32f) / maxNumLanes;  // No of float vectors required to process numLanes pixels
    Rpp32s numOutPixels, filterKernelStride;
    numOutPixels = filterKernelStride = 4;
    Rpp32s filterKernelSizeOverStride = filter.size % filterKernelStride;
    Rpp32s filterKernelRadiusWStrided = (Rpp32s)(filter.radius) * inputDescPtr->strides.wStride;

    Rpp32s inputWidthLimit = (inputImgSize.width - 1) * inputDescPtr->strides.wStride;
    __m128i pxInputWidthLimit = _mm_set1_epi32(inputWidthLimit);

    // For PLN3 inputs
    if(inputDescPtr->c == 3 && inputDescPtr->layout == RpptLayout::NCHW)
    {
        for (int outLocRow = 0; outLocRow < outputImgSize.height; outLocRow++)
        {
            T *outRowPtrR = outputPtr + outLocRow * outputDescPtr->strides.hStride;
            T *outRowPtrG = outRowPtrR + outputDescPtr->strides.cStride;
            T *outRowPtrB = outRowPtrG + outputDescPtr->strides.cStride;
            Rpp32f *inRowPtrR = inputPtr + outLocRow * inputDescPtr->strides.hStride;
            Rpp32f *inRowPtrG = inRowPtrR + inputDescPtr->strides.cStride;
            Rpp32f *inRowPtrB = inRowPtrG + inputDescPtr->strides.cStride;
            Rpp32s bufferLength = outputImgSize.width;
            Rpp32s alignedLength = bufferLength &~ (numLanes-1);
            __m128 pFirstValR = _mm_set1_ps(inRowPtrR[0]);
            __m128 pFirstValG = _mm_set1_ps(inRowPtrG[0]);
            __m128 pFirstValB = _mm_set1_ps(inRowPtrB[0]);
            bool breakLoop = false;
            Rpp32s outLocCol = 0;

            // Load filter.size consecutive pixels from a location in the row
            // Multiply with corresponding coeffs and add together to obtain the output pixel
            for (; outLocCol + numLanes <= alignedLength; outLocCol += numLanes)
            {
                __m128 pOutputChannel[numVecs * 3];
                set_zeros(pOutputChannel, numVecs * 3);
                __m128 *pOutputR = pOutputChannel;
                __m128 *pOutputG = pOutputChannel + numVecs;
                __m128 *pOutputB = pOutputChannel + (numVecs * 2);
                for(int vec = 0, x = outLocCol; vec < numVecs; vec++, x += numOutPixels)
                {
                    Rpp32s coeffIdx = (x * filter.size);
                    if(index[x] < 0)
                    {
                        __m128i pxIdx[numOutPixels];
                        pxIdx[0] = _mm_set1_epi32(index[x]);
                        pxIdx[1] = _mm_set1_epi32(index[x + 1]);
                        pxIdx[2] = _mm_set1_epi32(index[x + 2]);
                        pxIdx[3] = _mm_set1_epi32(index[x + 3]);
                        for(int k = 0; k < filter.size; k += filterKernelStride)
                        {
                            // Generate mask to determine the negative indices in the iteration
                            __m128i pxNegativeIndexMask[numOutPixels];
                            __m128i pxKernelIdx = _mm_set1_epi32(k);
                            __m128 pInputR[numOutPixels], pInputG[numOutPixels], pInputB[numOutPixels], pCoeffs[numOutPixels];
                            Rpp32s kernelAdd = (k + filterKernelStride) > filter.size ? filterKernelSizeOverStride : filterKernelStride;
                            set_zeros(pInputR, numOutPixels);
                            set_zeros(pInputG, numOutPixels);
                            set_zeros(pInputB, numOutPixels);
                            set_zeros(pCoeffs, numOutPixels);

                            for(int l = 0; l < numOutPixels; l++)
                            {
                                pxNegativeIndexMask[l] = _mm_cmplt_epi32(_mm_add_epi32(_mm_add_epi32(pxIdx[l], pxKernelIdx), xmm_pDstLocInit), xmm_px0);    // Generate mask to determine the negative indices in the iteration
                                Rpp32s srcx = index[x + l] + k;

                                // Load filterKernelStride(4) consecutive pixels
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtrR + srcx, pInputR + l);
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtrG + srcx, pInputG + l);
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtrB + srcx, pInputB + l);
                                pCoeffs[l] = _mm_loadu_ps(&(coeffs[coeffIdx + ((l + k) * 4)]));        // Load coefficients

                                // If negative index is present replace the input pixel value with first value in the row
                                pInputR[l] = _mm_blendv_ps(pInputR[l], pFirstValR, pxNegativeIndexMask[l]);
                                pInputG[l] = _mm_blendv_ps(pInputG[l], pFirstValG, pxNegativeIndexMask[l]);
                                pInputB[l] = _mm_blendv_ps(pInputB[l], pFirstValB, pxNegativeIndexMask[l]);
                            }

                            // Perform transpose operation to arrange input pixels from different output locations in each vector
                            _MM_TRANSPOSE4_PS(pInputR[0], pInputR[1], pInputR[2], pInputR[3]);
                            _MM_TRANSPOSE4_PS(pInputG[0], pInputG[1], pInputG[2], pInputG[3]);
                            _MM_TRANSPOSE4_PS(pInputB[0], pInputB[1], pInputB[2], pInputB[3]);
                            for (int l = 0; l < kernelAdd; l++)
                            {
                                pOutputR[vec] = _mm_fmadd_ps(pCoeffs[l], pInputR[l], pOutputR[vec]);
                                pOutputG[vec] = _mm_fmadd_ps(pCoeffs[l], pInputG[l], pOutputG[vec]);
                                pOutputB[vec] = _mm_fmadd_ps(pCoeffs[l], pInputB[l], pOutputB[vec]);
                            }
                        }
                    }
                    else if(index[x + 3] >= (inputWidthLimit - filterKernelRadiusWStrided))    // If the index value exceeds the limit, break the loop
                    {
                        breakLoop = true;
                        break;
                    }
                    else
                    {
                        // Considers a 4x1 window for computation each time
                        for(int k = 0; k < filter.size; k += filterKernelStride)
                        {
                            __m128 pInputR[numOutPixels], pInputG[numOutPixels], pInputB[numOutPixels];
                            __m128 pCoeffs[numOutPixels];
                            Rpp32s kernelAdd = (k + filterKernelStride) > filter.size ? filterKernelSizeOverStride : filterKernelStride;
                            for (int l = 0; l < numOutPixels; l++)
                            {
                                pInputR[l] = pInputG[l] = pInputB[l] = pCoeffs[l] = xmm_p0;
                                pCoeffs[l] = _mm_loadu_ps(&(coeffs[coeffIdx + ((l + k) * 4)])); // Load coefficients
                                Rpp32s srcx = index[x + l] + k;
                                srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                                // Load filterKernelStride(4) consecutive pixels
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtrR + srcx, pInputR + l);
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtrG + srcx, pInputG + l);
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtrB + srcx, pInputB + l);
                            }

                            // Perform transpose operation to arrange input pixels from different output locations in each vector
                            _MM_TRANSPOSE4_PS(pInputR[0], pInputR[1], pInputR[2], pInputR[3]);
                            _MM_TRANSPOSE4_PS(pInputG[0], pInputG[1], pInputG[2], pInputG[3]);
                            _MM_TRANSPOSE4_PS(pInputB[0], pInputB[1], pInputB[2], pInputB[3]);
                            for (int l = 0; l < kernelAdd; l++)
                            {
                                pOutputR[vec] = _mm_fmadd_ps(pCoeffs[l], pInputR[l], pOutputR[vec]);
                                pOutputG[vec] = _mm_fmadd_ps(pCoeffs[l], pInputG[l], pOutputG[vec]);
                                pOutputB[vec] = _mm_fmadd_ps(pCoeffs[l], pInputB[l], pOutputB[vec]);
                            }
                        }
                    }
                }
                if(breakLoop) break;
                Rpp32s xStride = outLocCol * outputDescPtr->strides.wStride;
                if(outputDescPtr->layout == RpptLayout::NCHW)       // For PLN3 outputs
                    rpp_resize_store_pln3(outRowPtrR + xStride, outRowPtrG + xStride, outRowPtrB + xStride, pOutputChannel);
                else if(outputDescPtr->layout == RpptLayout::NHWC)  // For PKD3 outputs
                    rpp_resize_store_pkd3(outRowPtrR + xStride, pOutputChannel);
            }
            Rpp32s k0 = 0;
            for (; outLocCol < outputImgSize.width; outLocCol++)
            {
                Rpp32s x0 = index[outLocCol];
                k0 = outLocCol % 4 == 0 ? outLocCol * filter.size : k0 + 1; // Since coeffs are stored in continuously for 4 dst locations
                Rpp32f sumR, sumG, sumB;
                sumR = sumG = sumB = 0;
                for (int k = 0; k < filter.size; k++)
                {
                    Rpp32s srcx = x0 + k;
                    srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                    Rpp32s kPos = (k * 4);      // Since coeffs are stored in continuously for 4 dst locations
                    sumR += (coeffs[k0 + kPos] * inRowPtrR[srcx]);
                    sumG += (coeffs[k0 + kPos] * inRowPtrG[srcx]);
                    sumB += (coeffs[k0 + kPos] * inRowPtrB[srcx]);
                }
                Rpp32s xStride = outLocCol * outputDescPtr->strides.wStride;
                saturate_pixel(sumR, outRowPtrR + xStride);
                saturate_pixel(sumG, outRowPtrG + xStride);
                saturate_pixel(sumB, outRowPtrB + xStride);
            }
        }
    }
    // For PKD3 inputs
    else if(inputDescPtr->c == 3 && inputDescPtr->layout == RpptLayout::NHWC)
    {
        for (int outLocRow = 0; outLocRow < outputImgSize.height; outLocRow++)
        {
            T *outRowPtrR = outputPtr + outLocRow * outputDescPtr->strides.hStride;
            T *outRowPtrG = outRowPtrR + outputDescPtr->strides.cStride;
            T *outRowPtrB = outRowPtrG + outputDescPtr->strides.cStride;
            Rpp32f *inRowPtr = inputPtr + outLocRow * inputDescPtr->strides.hStride;
            Rpp32s bufferLength = outputImgSize.width;
            Rpp32s alignedLength = bufferLength &~ (numLanes-1);
            Rpp32s outLocCol = 0;

            // Load filter.size consecutive pixels from a location in the row
            // Multiply with corresponding coeffs and add together to obtain the output pixel
            for (; outLocCol + numLanes <= alignedLength; outLocCol += numLanes)
            {
                __m128 pOutputChannel[numVecs * 3];
                set_zeros(pOutputChannel, numVecs * 3);
                __m128 *pOutputR = pOutputChannel;
                __m128 *pOutputG = pOutputChannel + numVecs;
                __m128 *pOutputB = pOutputChannel + (numVecs * 2);
                for(int vec = 0, x = outLocCol; vec < numVecs; vec++, x += numOutPixels)   // 4 dst pixels processed per iteration
                {
                    Rpp32s coeffIdx = (x * filter.size);
                    for(int k = 0, kStrided = 0; k < filter.size; k ++, kStrided = k * 3)
                    {
                        __m128 pInput[numOutPixels];
                        __m128 pCoeffs = _mm_loadu_ps(&(coeffs[coeffIdx + (k * numOutPixels)]));
                        for (int l = 0; l < numOutPixels; l++)
                        {
                            Rpp32s srcx = index[x + l] + kStrided;
                            srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                            rpp_simd_load(rpp_load4_f32_to_f32, &inRowPtr[srcx], &pInput[l]);   // Load RGB pixel from a src location
                        }

                        // Perform transpose operation to arrange input pixels by R,G and B separately in each vector
                        _MM_TRANSPOSE4_PS(pInput[0], pInput[1], pInput[2], pInput[3]);
                        pOutputR[vec] = _mm_fmadd_ps(pCoeffs, pInput[0], pOutputR[vec]);
                        pOutputG[vec] = _mm_fmadd_ps(pCoeffs, pInput[1], pOutputG[vec]);
                        pOutputB[vec] = _mm_fmadd_ps(pCoeffs, pInput[2], pOutputB[vec]);
                    }
                }

                Rpp32s xStride = outLocCol * outputDescPtr->strides.wStride;
                if(outputDescPtr->layout == RpptLayout::NCHW)       // For PLN3 outputs
                    rpp_resize_store_pln3(outRowPtrR + xStride, outRowPtrG + xStride, outRowPtrB + xStride, pOutputChannel);
                else if(outputDescPtr->layout == RpptLayout::NHWC)  // For PKD3 outputs
                    rpp_resize_store_pkd3(outRowPtrR + xStride, pOutputChannel);
            }
            Rpp32s k0 = 0;
            for (; outLocCol < outputImgSize.width; outLocCol++)
            {
                Rpp32s x0 = index[outLocCol];
                k0 = outLocCol % 4 == 0 ? outLocCol * filter.size : k0 + 1;  // Since coeffs are stored in continuously for 4 dst locations
                Rpp32f sumR, sumG, sumB;
                sumR = sumG = sumB = 0;
                for (int k = 0; k < filter.size; k++)
                {
                    Rpp32s srcx = x0 + (k * 3);
                    srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                    Rpp32s kPos = (k * 4);      // Since coeffs are stored in continuously for 4 dst locations
                    sumR += (coeffs[k0 + kPos] * inRowPtr[srcx]);
                    sumG += (coeffs[k0 + kPos] * inRowPtr[srcx + 1]);
                    sumB += (coeffs[k0 + kPos] * inRowPtr[srcx + 2]);
                }
                Rpp32s xStride = outLocCol * outputDescPtr->strides.wStride;
                saturate_pixel(sumR, outRowPtrR + xStride);
                saturate_pixel(sumG, outRowPtrG + xStride);
                saturate_pixel(sumB, outRowPtrB + xStride);
            }
        }
    }
    else
    {
        for (int outLocRow = 0; outLocRow < outputImgSize.height; outLocRow++)
        {
            T *out_row = outputPtr + outLocRow * outputDescPtr->strides.hStride;
            Rpp32f *inRowPtr = inputPtr + outLocRow * inputDescPtr->strides.hStride;
            Rpp32s bufferLength = outputImgSize.width;
            Rpp32s alignedLength = bufferLength &~ (numLanes-1);
            __m128 pFirstVal = _mm_set1_ps(inRowPtr[0]);
            bool breakLoop = false;
            Rpp32s outLocCol = 0;

            // Load filter.size consecutive pixels from a location in the row
            // Multiply with corresponding coeffs and add together to obtain the output pixel
            for (; outLocCol + numLanes <= alignedLength; outLocCol += numLanes)
            {
                __m128 pOutput[numVecs];
                set_zeros(pOutput, numVecs);
                for(int vec = 0, x = outLocCol; vec < numVecs; vec++, x += numOutPixels)
                {
                    Rpp32s coeffIdx = (x * filter.size);
                    if(index[x] < 0)
                    {
                        __m128i pxIdx[numOutPixels];
                        pxIdx[0] = _mm_set1_epi32(index[x]);
                        pxIdx[1] = _mm_set1_epi32(index[x + 1]);
                        pxIdx[2] = _mm_set1_epi32(index[x + 2]);
                        pxIdx[3] = _mm_set1_epi32(index[x + 3]);
                        for(int k = 0; k < filter.size; k += filterKernelStride)
                        {
                            __m128i pxNegativeIndexMask[numOutPixels];
                            __m128i pxKernelIdx = _mm_set1_epi32(k);
                            __m128 pInput[numOutPixels], pCoeffs[numOutPixels];
                            Rpp32s kernelAdd = (k + filterKernelStride) > filter.size ? filterKernelSizeOverStride : filterKernelStride;
                            set_zeros(pInput, numOutPixels);
                            set_zeros(pCoeffs, numOutPixels);
                            for(int l = 0; l < numOutPixels; l++)
                            {
                                pxNegativeIndexMask[l] = _mm_cmplt_epi32(_mm_add_epi32(_mm_add_epi32(pxIdx[l], pxKernelIdx), xmm_pDstLocInit), xmm_px0);    // Generate mask to determine the negative indices in the iteration
                                rpp_simd_load(rpp_load4_f32_to_f32, &inRowPtr[index[x + l] + k], &pInput[l]);   // Load filterKernelStride(4) consecutive pixels
                                pCoeffs[l] = _mm_loadu_ps(&(coeffs[coeffIdx + ((l + k) * 4)]));                 // Load coefficients
                                pInput[l] = _mm_blendv_ps(pInput[l], pFirstVal, pxNegativeIndexMask[l]);        // If negative index is present replace the pixel value with first value in the row
                            }
                            _MM_TRANSPOSE4_PS(pInput[0], pInput[1], pInput[2], pInput[3]);  // Perform transpose operation to arrange input pixels from different output locations in each vector
                            for (int l = 0; l < kernelAdd; l++)
                                pOutput[vec] = _mm_fmadd_ps(pCoeffs[l], pInput[l], pOutput[vec]);
                        }
                    }
                    else if(index[x + 3] >= (inputWidthLimit - filterKernelRadiusWStrided))   // If the index value exceeds the limit, break the loop
                    {
                        breakLoop = true;
                        break;
                    }
                    else
                    {
                        for(int k = 0; k < filter.size; k += filterKernelStride)
                        {
                            __m128 pInput[numOutPixels], pCoeffs[numOutPixels];
                            Rpp32s kernelAdd = (k + filterKernelStride) > filter.size ? filterKernelSizeOverStride : filterKernelStride;
                            for (int l = 0; l < numOutPixels; l++)
                            {
                                pInput[l] = pCoeffs[l] = xmm_p0;
                                pCoeffs[l] = _mm_loadu_ps(&(coeffs[coeffIdx + ((l + k) * 4)]));     // Load coefficients
                                Rpp32s srcx = index[x + l] + k;
                                srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtr + srcx, pInput + l);   // Load filterKernelStride(4) consecutive pixels
                            }
                            _MM_TRANSPOSE4_PS(pInput[0], pInput[1], pInput[2], pInput[3]);  // Perform transpose operation to arrange input pixels from different output locations in each vector
                            for (int l = 0; l < kernelAdd; l++)
                                pOutput[vec] = _mm_fmadd_ps(pCoeffs[l], pInput[l], pOutput[vec]);
                        }
                    }
                }
                if(breakLoop) break;
                rpp_resize_store(out_row + outLocCol, pOutput);
            }
            Rpp32s k0 = 0;
            for (; outLocCol < bufferLength; outLocCol++)
            {
                Rpp32s x0 = index[outLocCol];
                k0 = outLocCol % 4 == 0 ? outLocCol * filter.size : k0 + 1;  // Since coeffs are stored in continuously for 4 dst locations
                Rpp32f sum = 0;
                for (int k = 0; k < filter.size; k++)
                {
                    Rpp32s srcx = x0 + k;
                    srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                    sum += (coeffs[k0 + (k * 4)] * inRowPtr[srcx]);
                }
                saturate_pixel(sum, out_row + outLocCol);
            }
        }
    }
}

inline void compute_jitter_src_loc_avx(__m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 &pRow, __m256 &pCol, __m256 &pKernelSize, __m256 &pBound, __m256 &pHeightLimit, __m256 &pWidthLimit, __m256 &pStride, __m256 &pChannel, Rpp32s *srcLoc)
{
    __m256 pRngX = rpp_host_rng_xorwow_8_f32_avx(pxXorwowStateX, pxXorwowStateCounter);
    __m256 pRngY = rpp_host_rng_xorwow_8_f32_avx(pxXorwowStateX, pxXorwowStateCounter);
    __m256 pX = _mm256_mul_ps(pRngX, pKernelSize);
    __m256 pY = _mm256_mul_ps(pRngY, pKernelSize);
    pX = _mm256_max_ps(_mm256_min_ps(_mm256_floor_ps(_mm256_add_ps(pRow, _mm256_sub_ps(pX, pBound))), pHeightLimit), avx_p0);
    pY = _mm256_max_ps(_mm256_min_ps(_mm256_floor_ps(_mm256_add_ps(pCol, _mm256_sub_ps(pY, pBound))), pWidthLimit), avx_p0);
    __m256i pxSrcLoc = _mm256_cvtps_epi32(_mm256_fmadd_ps(pX, pStride, _mm256_mul_ps(pY, pChannel)));
    _mm256_storeu_si256((__m256i*) srcLoc, pxSrcLoc);
}

inline void compute_jitter_src_loc(RpptXorwowStateBoxMuller *xorwowState, Rpp32s row, Rpp32s col, Rpp32s kSize, Rpp32s heightLimit, Rpp32s widthLimit, Rpp32s stride, Rpp32s bound, Rpp32s channels, Rpp32s &loc)
{
    Rpp32u heightIncrement = rpp_host_rng_xorwow_f32(xorwowState) * kSize;
    Rpp32u widthIncrement = rpp_host_rng_xorwow_f32(xorwowState) * kSize;
    loc = std::max(std::min(static_cast<int>(row + heightIncrement - bound), heightLimit), 0) * stride;
    loc += std::max(std::min(static_cast<int>(col + widthIncrement  - bound), (widthLimit - 1)), 0) * channels;
}
inline void compute_sum_16_host(__m256i *p, __m256i *pSum)
{
    pSum[0] = _mm256_add_epi32(_mm256_add_epi32(p[0], p[1]), pSum[0]); //add 16 values to 8
}

inline void compute_sum_48_host(__m256i *p, __m256i *pSumR, __m256i *pSumG, __m256i *pSumB)
{
    pSumR[0] = _mm256_add_epi32(_mm256_add_epi32(p[0], p[1]), pSumR[0]); //add 16R values and bring it down to 8
    pSumG[0] = _mm256_add_epi32(_mm256_add_epi32(p[2], p[3]), pSumG[0]); //add 16G values and bring it down to 8
    pSumB[0] = _mm256_add_epi32(_mm256_add_epi32(p[4], p[5]), pSumB[0]); //add 16B values and bring it down to 8
}

inline void compute_sum_8_host(__m256d *p, __m256d *pSum)
{
    pSum[0] = _mm256_add_pd(_mm256_add_pd(p[0], p[1]), pSum[0]); //add 8 values and bring it down to 4
}

inline void compute_sum_24_host(__m256d *p, __m256d *pSumR, __m256d *pSumG, __m256d *pSumB)
{
    pSumR[0] = _mm256_add_pd(_mm256_add_pd(p[0], p[1]), pSumR[0]); //add 8R values and bring it down to 4
    pSumG[0] = _mm256_add_pd(_mm256_add_pd(p[2], p[3]), pSumG[0]); //add 8G values and bring it down to 4
    pSumB[0] = _mm256_add_pd(_mm256_add_pd(p[4], p[5]), pSumB[0]); //add 8B values and bring it down to 4
}

inline void compute_variance_8_host(__m256d *p1, __m256d *pMean, __m256d *pVar)
{
    __m256d pSub = _mm256_sub_pd(p1[0], pMean[0]);
    pVar[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVar[0]);
    pSub = _mm256_sub_pd(p1[1], pMean[0]);
    pVar[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVar[0]);
}

inline void compute_variance_channel_pln3_24_host(__m256d *p1, __m256d *pMeanR, __m256d *pMeanG, __m256d *pMeanB, __m256d *pVarR, __m256d *pVarG, __m256d *pVarB)
{
    __m256d pSub = _mm256_sub_pd(p1[0], pMeanR[0]);
    pVarR[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarR[0]);
    pSub = _mm256_sub_pd(p1[1], pMeanR[0]);
    pVarR[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarR[0]);
    pSub = _mm256_sub_pd(p1[2], pMeanG[0]);
    pVarG[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarG[0]);
    pSub = _mm256_sub_pd(p1[3], pMeanG[0]);
    pVarG[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarG[0]);
    pSub = _mm256_sub_pd(p1[4], pMeanB[0]);
    pVarB[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarB[0]);
    pSub = _mm256_sub_pd(p1[5], pMeanB[0]);
    pVarB[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarB[0]);
}

inline void compute_variance_image_pln3_24_host(__m256d *p1, __m256d *pMean, __m256d *pVarR, __m256d *pVarG, __m256d *pVarB)
{
    __m256d pSub = _mm256_sub_pd(p1[0], pMean[0]);
    pVarR[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarR[0]);
    pSub = _mm256_sub_pd(p1[1], pMean[0]);
    pVarR[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarR[0]);
    pSub = _mm256_sub_pd(p1[2], pMean[0]);
    pVarG[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarG[0]);
    pSub = _mm256_sub_pd(pMean[0], p1[3]);
    pVarG[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarG[0]);
    pSub = _mm256_sub_pd(p1[4], pMean[0]);
    pVarB[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarB[0]);
    pSub = _mm256_sub_pd(p1[5], pMean[0]);
    pVarB[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarB[0]);
}

inline void compute_vignette_48_host(__m256 *p, __m256 &pMultiplier, __m256 &pILocComponent, __m256 &pJLocComponent)
{
    __m256 pGaussianValue;
    pGaussianValue = fast_exp_avx(_mm256_mul_ps(_mm256_fmadd_ps(pJLocComponent, pJLocComponent, pILocComponent), pMultiplier));
    p[0] = _mm256_mul_ps(p[0], pGaussianValue);    // vignette adjustment
    p[2] = _mm256_mul_ps(p[2], pGaussianValue);    // vignette adjustment
    p[4] = _mm256_mul_ps(p[4], pGaussianValue);    // vignette adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
    pGaussianValue = fast_exp_avx(_mm256_mul_ps(_mm256_fmadd_ps(pJLocComponent, pJLocComponent, pILocComponent), pMultiplier));
    p[1] = _mm256_mul_ps(p[1], pGaussianValue);    // vignette adjustment
    p[3] = _mm256_mul_ps(p[3], pGaussianValue);    // vignette adjustment
    p[5] = _mm256_mul_ps(p[5], pGaussianValue);    // vignette adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
}

inline void compute_vignette_24_host(__m256 *p, __m256 &pMultiplier, __m256 &pILocComponent, __m256 &pJLocComponent)
{
    __m256 pGaussianValue;
    pGaussianValue = fast_exp_avx(_mm256_mul_ps(_mm256_fmadd_ps(pJLocComponent, pJLocComponent, pILocComponent), pMultiplier));
    p[0] = _mm256_mul_ps(p[0], pGaussianValue);    // vignette adjustment
    p[1] = _mm256_mul_ps(p[1], pGaussianValue);    // vignette adjustment
    p[2] = _mm256_mul_ps(p[2], pGaussianValue);    // vignette adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
}

inline void compute_vignette_16_host(__m256 *p, __m256 &pMultiplier, __m256 &pILocComponent, __m256 &pJLocComponent)
{
    __m256 pGaussianValue;
    pGaussianValue = fast_exp_avx(_mm256_mul_ps(_mm256_fmadd_ps(pJLocComponent, pJLocComponent, pILocComponent), pMultiplier));
    p[0] = _mm256_mul_ps(p[0], pGaussianValue);    // vignette adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
    pGaussianValue = fast_exp_avx(_mm256_mul_ps(_mm256_fmadd_ps(pJLocComponent, pJLocComponent, pILocComponent), pMultiplier));
    p[1] = _mm256_mul_ps(p[1], pGaussianValue);    // vignette adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
}

inline void compute_vignette_8_host(__m256 *p, __m256 &pMultiplier, __m256 &pILocComponent, __m256 &pJLocComponent)
{
    __m256 pGaussianValue;
    pGaussianValue = fast_exp_avx(_mm256_mul_ps(_mm256_fmadd_ps(pJLocComponent, pJLocComponent, pILocComponent), pMultiplier));
    p[0] = _mm256_mul_ps(p[0], pGaussianValue);    // vignette adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
}

inline void reduce_min_32_host(__m256i *pMin, __m128i *result)
{
    __m128i px[2];
    __m128i zero = _mm_setzero_si128();
    __m128i mask = _mm_set_epi8(0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,7);
    px[0] = _mm256_castsi256_si128(pMin[0]);
    px[1] = _mm256_extracti128_si256(pMin[0], 1);
    px[0] = _mm_min_epu8(px[0], px[1]);
    px[1] = _mm_unpacklo_epi8(zero, px[0]);
    px[0] = _mm_unpackhi_epi8(zero, px[0]);
    px[0] = _mm_min_epu8(px[0], px[1]);
    px[1] = _mm_unpacklo_epi16(zero, px[0]);
    px[0] = _mm_unpackhi_epi16(zero, px[0]);
    px[0] = _mm_min_epu16(px[0], px[1]);
    px[1] = _mm_unpacklo_epi32(zero, px[0]);
    px[0] = _mm_unpackhi_epi32(zero, px[0]);
    px[0] = _mm_min_epu32(px[0], px[1]);
    result[0] = _mm_shuffle_epi8(px[0], mask);
}

inline void compute_min_96_host(__m256i *p1, __m256i *pMinR, __m256i *pMinG, __m256i *pMinB)
{
    pMinR[0] = _mm256_min_epu8(p1[0], pMinR[0]); //compare and store min of 32 R values into global min
    pMinG[0] = _mm256_min_epu8(p1[1], pMinG[0]); //compare and store min of 32 G values into global min
    pMinB[0] = _mm256_min_epu8(p1[2], pMinB[0]); //compare and store min of 32 B values into global min
}

inline void reduce_min_96_host(__m256i *pMinR, __m256i *pMinG, __m256i *pMinB, __m128i *result)
{
    __m128i px[4];
    __m128i zero = _mm_setzero_si128();
    px[0] = _mm_min_epu8(_mm256_castsi256_si128(pMinR[0]), _mm256_extracti128_si256(pMinR[0], 1));
    px[1] = _mm_min_epu8(_mm256_castsi256_si128(pMinG[0]), _mm256_extracti128_si256(pMinG[0], 1));
    px[1] = _mm_min_epu8(_mm_unpacklo_epi8(px[0], px[1]), _mm_unpackhi_epi8(px[0], px[1]));
    px[0] = _mm_min_epu8(_mm256_castsi256_si128(pMinB[0]), _mm256_extracti128_si256(pMinB[0], 1));
    px[0] = _mm_min_epu8(_mm_unpacklo_epi8(px[0], zero), _mm_unpackhi_epi8(px[0], zero));
    px[1] = _mm_min_epu8(_mm_unpacklo_epi16(px[1], px[0]), _mm_unpackhi_epi16(px[1], px[0]));
    px[0] = _mm_min_epu8(_mm_unpacklo_epi32(px[1], zero), _mm_unpackhi_epi32(px[1], zero));
    result[0] = _mm_min_epu8(_mm_unpacklo_epi64(px[0], zero), _mm_unpackhi_epi64(px[0], zero));
}

inline void compute_min_48_host(__m128i *p1, __m128i *pMinR, __m128i *pMinG, __m128i *pMinB)
{
    pMinR[0] = _mm_min_epu8(p1[0], pMinR[0]); //compare and store min of 16 R values into global min
    pMinG[0] = _mm_min_epu8(p1[1], pMinG[0]); //compare and store min of 16 G values into global min
    pMinB[0] = _mm_min_epu8(p1[2], pMinB[0]); //compare and store min of 16 B values into global min
}

inline void reduce_min_48_host(__m128i *pMinR, __m128i *pMinG, __m128i *pMinB, __m128i *result)
{
    __m128i px[2];
    __m128i zero = _mm_setzero_si128();
    px[1] = _mm_min_epu8(_mm_unpacklo_epi8(pMinR[0], pMinG[0]), _mm_unpackhi_epi8(pMinR[0], pMinG[0]));
    px[0] = _mm_min_epu8(_mm_unpacklo_epi8(pMinB[0], zero), _mm_unpackhi_epi8(pMinB[0], zero));
    px[1] = _mm_min_epu8(_mm_unpacklo_epi16(px[1], px[0]), _mm_unpackhi_epi16(px[1], px[0]));
    px[0] = _mm_min_epu8(_mm_unpacklo_epi32(px[1], zero), _mm_unpackhi_epi32(px[1], zero));
    result[0] = _mm_min_epu8(_mm_unpacklo_epi64(px[0], zero), _mm_unpackhi_epi64(px[0], zero));
}

inline void reduce_max_32_host(__m256i *pMax, __m128i *result)
{
    __m128i px;
    __m128i zero = _mm_setzero_si128();
    __m128i mask = _mm_set_epi8(0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,7);
    px = _mm_max_epu8(_mm256_castsi256_si128(pMax[0]), _mm256_extracti128_si256(pMax[0], 1));
    px = _mm_max_epu8(_mm_unpacklo_epi8(zero, px), _mm_unpackhi_epi8(zero, px));
    px = _mm_max_epu16(_mm_unpacklo_epi16(zero, px), _mm_unpackhi_epi16(zero, px));
    px = _mm_max_epu32(_mm_unpacklo_epi32(zero, px), _mm_unpackhi_epi32(zero, px));
    result[0] = _mm_shuffle_epi8(px, mask);
}

inline void compute_max_96_host(__m256i *p1, __m256i *pMaxR, __m256i *pMaxG, __m256i *pMaxB)
{
    pMaxR[0] = _mm256_max_epu8(p1[0], pMaxR[0]); //compare and store max of 32 R values into global max
    pMaxG[0] = _mm256_max_epu8(p1[1], pMaxG[0]); //compare and store max of 32 G values into global max
    pMaxB[0] = _mm256_max_epu8(p1[2], pMaxB[0]); //compare and store max of 32 B values into global max
}

inline void reduce_max_96_host(__m256i *pMaxR, __m256i *pMaxG, __m256i *pMaxB, __m128i *result)
{
    __m128i px[4];
    __m128i zero = _mm_setzero_si128();
    px[0] = _mm_max_epu8(_mm256_castsi256_si128(pMaxR[0]), _mm256_extracti128_si256(pMaxR[0], 1));
    px[1] = _mm_max_epu8(_mm256_castsi256_si128(pMaxG[0]), _mm256_extracti128_si256(pMaxG[0], 1));
    px[1] = _mm_max_epu8(_mm_unpacklo_epi8(px[0], px[1]), _mm_unpackhi_epi8(px[0], px[1]));
    px[0] = _mm_max_epu8(_mm256_castsi256_si128(pMaxB[0]), _mm256_extracti128_si256(pMaxB[0], 1));
    px[0] = _mm_max_epu8(_mm_unpacklo_epi8(px[0], zero), _mm_unpackhi_epi8(px[0], zero));
    px[1] = _mm_max_epu8(_mm_unpacklo_epi16(px[1], px[0]), _mm_unpackhi_epi16(px[1], px[0]));
    px[0] = _mm_max_epu8(_mm_unpacklo_epi32(px[1], zero), _mm_unpackhi_epi32(px[1], zero));
    result[0] = _mm_max_epu8(_mm_unpacklo_epi64(px[0], zero), _mm_unpackhi_epi64(px[0], zero));
}

inline void compute_max_48_host(__m128i *p1, __m128i *pMaxR, __m128i *pMaxG, __m128i *pMaxB)
{
    pMaxR[0] = _mm_max_epu8(p1[0], pMaxR[0]); //compare and store max of 16 R values into global max
    pMaxG[0] = _mm_max_epu8(p1[1], pMaxG[0]); //compare and store max of 16 G values into global max
    pMaxB[0] = _mm_max_epu8(p1[2], pMaxB[0]); //compare and store max of 16 B values into global max
}

inline void reduce_max_48_host(__m128i *pMaxR, __m128i *pMaxG, __m128i *pMaxB, __m128i *result)
{
    __m128i px[2];
    __m128i zero = _mm_setzero_si128();
    px[1] = _mm_max_epi8(_mm_unpacklo_epi8(pMaxR[0], pMaxG[0]), _mm_unpackhi_epi8(pMaxR[0], pMaxG[0]));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi8(pMaxB[0], zero), _mm_unpackhi_epi8(pMaxB[0], zero));
    px[1] = _mm_max_epi8(_mm_unpacklo_epi16(px[1], px[0]), _mm_unpackhi_epi16(px[1], px[0]));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi32(px[1], zero), _mm_unpackhi_epi32(px[1], zero));
    result[0] = _mm_max_epi8(_mm_unpacklo_epi64(px[0], zero), _mm_unpackhi_epi64(px[0], zero));
}

inline void compute_min_float8_host(__m256 *p1, __m256 *pMin)
{
    pMin[0] = _mm256_min_ps(p1[0], pMin[0]); //compare and store min of 8 values into global min
}

inline void reduce_min_float8_host(__m256 *pMin, __m128 *result)
{
    __m128 px;
    px = _mm_min_ps(_mm256_castps256_ps128(pMin[0]), _mm256_extractf128_ps(pMin[0], 1));
    px = _mm_min_ps(_mm_unpacklo_ps(xmm_p0, px), _mm_unpackhi_ps(xmm_p0, px));
    result[0] = _mm_shuffle_ps(px, px, 39);
}

inline void compute_min_float24_host(__m256 *p1, __m256 *pMinR, __m256 *pMinG, __m256 *pMinB)
{
    pMinR[0] = _mm256_min_ps(p1[0], pMinR[0]); //compare and store min of 8 R values into global min
    pMinG[0] = _mm256_min_ps(p1[1], pMinG[0]); //compare and store min of 8 G values into global min
    pMinB[0] = _mm256_min_ps(p1[2], pMinB[0]); //compare and store min of 8 B values into global min
}

inline void reduce_min_float24_host(__m256 *pMinR, __m256 *pMinG, __m256 *pMinB, __m256 *result)   // TO CHANGE
{
    __m128 px[2];
    px[0] = _mm_min_ps(_mm256_castps256_ps128(pMinR[0]), _mm256_extractf128_ps(pMinR[0], 1));
    px[1] = _mm_min_ps(_mm256_castps256_ps128(pMinG[0]), _mm256_extractf128_ps(pMinG[0], 1));
    px[0] = _mm_min_ps(_mm_unpacklo_ps(px[0], px[1]), _mm_unpackhi_ps(px[0], px[1]));
    px[0] = _mm_permute_ps(px[0], 0b11011000);
    result[0] = _mm256_castps128_ps256(px[0]);
    px[0] = _mm_min_ps(_mm256_castps256_ps128(pMinB[0]), _mm256_extractf128_ps(pMinB[0], 1));
    px[1] = _mm_min_ps(_mm_unpacklo_ps(px[0], xmm_p0), _mm_unpackhi_ps(px[0], xmm_p0));
    px[0] = _mm_shuffle_ps(px[1], px[1], 34);
    result[0] = _mm256_insertf128_ps(result[0], px[0], 1);
}

inline void compute_max_float8_host(__m256 *p1, __m256 *pMax)
{
    pMax[0] = _mm256_max_ps(p1[0], pMax[0]); //compare and store max of 8 values into global min
}

inline void reduce_max_float8_host(__m256 *pMax, __m128 *result)
{
    __m128 px;
    px = _mm_max_ps(_mm256_castps256_ps128(pMax[0]), _mm256_extractf128_ps(pMax[0], 1));
    px = _mm_max_ps(_mm_unpacklo_ps(xmm_p0, px), _mm_unpackhi_ps(xmm_p0, px));
    result[0] = _mm_shuffle_ps(px, px, 39);
}

inline void compute_max_float24_host(__m256 *p1, __m256 *pMaxR, __m256 *pMaxG, __m256 *pMaxB)
{
    pMaxR[0] = _mm256_max_ps(p1[0], pMaxR[0]); //compare and store max of 8 R values into global min
    pMaxG[0] = _mm256_max_ps(p1[1], pMaxG[0]); //compare and store max of 8 G values into global min
    pMaxB[0] = _mm256_max_ps(p1[2], pMaxB[0]); //compare and store max of 8 B values into global min
}

inline void reduce_max_float24_host(__m256 *pMaxR, __m256 *pMaxG, __m256 *pMaxB, __m256 *result)
{
    __m128 px[2];
    px[0] = _mm_max_ps(_mm256_castps256_ps128(pMaxR[0]), _mm256_extractf128_ps(pMaxR[0], 1));
    px[1] = _mm_max_ps(_mm256_castps256_ps128(pMaxG[0]), _mm256_extractf128_ps(pMaxG[0], 1));
    px[0] = _mm_max_ps(_mm_unpacklo_ps(px[0], px[1]), _mm_unpackhi_ps(px[0], px[1]));
    px[0] = _mm_permute_ps(px[0], 0b11011000);
    result[0] = _mm256_castps128_ps256(px[0]);
    px[0] = _mm_max_ps(_mm256_castps256_ps128(pMaxB[0]), _mm256_extractf128_ps(pMaxB[0], 1));
    px[1] = _mm_max_ps(_mm_unpacklo_ps(px[0], xmm_p0), _mm_unpackhi_ps(px[0], xmm_p0));
    px[0] = _mm_shuffle_ps(px[1], px[1], 34);
    result[0] = _mm256_insertf128_ps(result[0], px[0], 1);
}

inline void reduce_min_i32_host(__m256i *pMin, __m128i *result)
{
    __m128i px;
    __m128i zero = _mm_setzero_si128();
    __m128i mask = _mm_set_epi8(0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,7);
    px = _mm_min_epi8(_mm256_castsi256_si128(pMin[0]), _mm256_extracti128_si256(pMin[0], 1));
    px = _mm_min_epi8(_mm_unpacklo_epi8(zero, px), _mm_unpackhi_epi8(zero, px));
    px = _mm_min_epi16(_mm_unpacklo_epi16(zero, px), _mm_unpackhi_epi16(zero, px));
    px = _mm_min_epi32(_mm_unpacklo_epi32(zero, px), _mm_unpackhi_epi32(zero, px));
    result[0] = _mm_shuffle_epi8(px, mask);
}

inline void compute_min_i96_host(__m256i *p1, __m256i *pMinR, __m256i *pMinG, __m256i *pMinB)
{
    pMinR[0] = _mm256_min_epi8(p1[0], pMinR[0]); //compare and store min of 32 R values into global min
    pMinG[0] = _mm256_min_epi8(p1[1], pMinG[0]); //compare and store min of 32 G values into global min
    pMinB[0] = _mm256_min_epi8(p1[2], pMinB[0]); //compare and store min of 32 B values into global min
}

inline void reduce_min_i96_host(__m256i *pMinR, __m256i *pMinG, __m256i *pMinB, __m128i *result)
{
    __m128i px[4];
    __m128i zero = _mm_setzero_si128();
    px[0] = _mm_min_epi8(_mm256_castsi256_si128(pMinR[0]), _mm256_extracti128_si256(pMinR[0], 1));
    px[1] = _mm_min_epi8(_mm256_castsi256_si128(pMinG[0]), _mm256_extracti128_si256(pMinG[0], 1));
    px[1] = _mm_min_epi8(_mm_unpacklo_epi8(px[0], px[1]), _mm_unpackhi_epi8(px[0], px[1]));
    px[0] = _mm_min_epi8(_mm256_castsi256_si128(pMinB[0]), _mm256_extracti128_si256(pMinB[0], 1));
    px[0] = _mm_min_epi8(_mm_unpacklo_epi8(px[0], zero), _mm_unpackhi_epi8(px[0], zero));
    px[1] = _mm_min_epi8(_mm_unpacklo_epi16(px[1], px[0]), _mm_unpackhi_epi16(px[1], px[0]));
    px[0] = _mm_min_epi8(_mm_unpacklo_epi32(px[1], zero), _mm_unpackhi_epi32(px[1], zero));
    result[0] = _mm_min_epi8(_mm_unpacklo_epi64(px[0], zero), _mm_unpackhi_epi64(px[0], zero));
}

inline void compute_min_i48_host(__m128i *p1, __m128i *pMinR, __m128i *pMinG, __m128i *pMinB)
{
    pMinR[0] = _mm_min_epi8(p1[0], pMinR[0]); //compare and store min of 16 R values into global min
    pMinG[0] = _mm_min_epi8(p1[1], pMinG[0]); //compare and store min of 16 G values into global min
    pMinB[0] = _mm_min_epi8(p1[2], pMinB[0]); //compare and store min of 16 B values into global min
}

inline void reduce_min_i48_host(__m128i *pMinR, __m128i *pMinG, __m128i *pMinB, __m128i *result)
{
    __m128i px[2];
    __m128i zero = _mm_setzero_si128();
    px[1] = _mm_min_epi8(_mm_unpacklo_epi8(pMinR[0], pMinG[0]), _mm_unpackhi_epi8(pMinR[0], pMinG[0]));
    px[0] = _mm_min_epi8(_mm_unpacklo_epi8(pMinB[0], zero), _mm_unpackhi_epi8(pMinB[0], zero));
    px[1] = _mm_min_epi8(_mm_unpacklo_epi16(px[1], px[0]), _mm_unpackhi_epi16(px[1], px[0]));
    px[0] = _mm_min_epi8(_mm_unpacklo_epi32(px[1], zero), _mm_unpackhi_epi32(px[1], zero));
    result[0] = _mm_min_epi8(_mm_unpacklo_epi64(px[0], zero), _mm_unpackhi_epi64(px[0], zero));
}

inline void reduce_max_i32_host(__m256i *pMax, __m128i *result)
{
    __m128i px[2];
    __m128i zero = _mm_setzero_si128();
    __m128i mask = _mm_set_epi8(0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,7);
    px[0] = _mm_max_epi8(_mm256_castsi256_si128(pMax[0]), _mm256_extracti128_si256(pMax[0], 1));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi8(zero, px[0]), _mm_unpackhi_epi8(zero, px[0]));
    px[0] = _mm_max_epi16(_mm_unpacklo_epi16(zero, px[0]), _mm_unpackhi_epi16(zero, px[0]));
    px[0] = _mm_max_epi32(_mm_unpacklo_epi32(zero, px[0]), _mm_unpackhi_epi32(zero, px[0]));
    result[0] = _mm_shuffle_epi8(px[0], mask);
}

inline void compute_max_i96_host(__m256i *p1, __m256i *pMaxR, __m256i *pMaxG, __m256i *pMaxB)
{
    pMaxR[0] = _mm256_max_epi8(p1[0], pMaxR[0]); //compare and store max of 32 R values into global max
    pMaxG[0] = _mm256_max_epi8(p1[1], pMaxG[0]); //compare and store max of 32 G values into global max
    pMaxB[0] = _mm256_max_epi8(p1[2], pMaxB[0]); //compare and store max of 32 B values into global max
}

inline void reduce_max_i96_host(__m256i *pMaxR, __m256i *pMaxG, __m256i *pMaxB, __m128i *result)
{
    __m128i px[4];
    __m128i zero = _mm_setzero_si128();
    px[0] = _mm_max_epi8(_mm256_castsi256_si128(pMaxR[0]), _mm256_extracti128_si256(pMaxR[0], 1));
    px[1] = _mm_max_epi8(_mm256_castsi256_si128(pMaxG[0]), _mm256_extracti128_si256(pMaxG[0], 1));
    px[1] = _mm_max_epi8(_mm_unpacklo_epi8(px[0], px[1]), _mm_unpackhi_epi8(px[0], px[1]));
    px[0] = _mm_max_epi8(_mm256_castsi256_si128(pMaxB[0]), _mm256_extracti128_si256(pMaxB[0], 1));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi8(px[0], zero), _mm_unpackhi_epi8(px[0], zero));
    px[1] = _mm_max_epi8(_mm_unpacklo_epi16(px[1], px[0]), _mm_unpackhi_epi16(px[1], px[0]));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi32(px[1], zero), _mm_unpackhi_epi32(px[1], zero));
    result[0] = _mm_max_epi8(_mm_unpacklo_epi64(px[0], zero), _mm_unpackhi_epi64(px[0], zero));
}

inline void compute_max_i48_host(__m128i *p1, __m128i *pMaxR, __m128i *pMaxG, __m128i *pMaxB)
{
    pMaxR[0] = _mm_max_epi8(p1[0], pMaxR[0]); //compare and store max of 16 R values into global max
    pMaxG[0] = _mm_max_epi8(p1[1], pMaxG[0]); //compare and store max of 16 G values into global max
    pMaxB[0] = _mm_max_epi8(p1[2], pMaxB[0]); //compare and store max of 16 B values into global max
}

inline void reduce_max_i48_host(__m128i *pMaxR, __m128i *pMaxG, __m128i *pMaxB, __m128i *result)
{
    __m128i px[2];
    __m128i zero = _mm_setzero_si128();
    px[1] = _mm_max_epi8(_mm_unpacklo_epi8(pMaxR[0], pMaxG[0]), _mm_unpackhi_epi8(pMaxR[0], pMaxG[0]));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi8(pMaxB[0], zero), _mm_unpackhi_epi8(pMaxB[0], zero));
    px[1] = _mm_max_epi8(_mm_unpacklo_epi16(px[1], px[0]), _mm_unpackhi_epi16(px[1], px[0]));
    px[0] = _mm_max_epi8(_mm_unpacklo_epi32(px[1], zero), _mm_unpackhi_epi32(px[1], zero));
    result[0] = _mm_max_epi8(_mm_unpacklo_epi64(px[0], zero), _mm_unpackhi_epi64(px[0], zero));
}

inline void compute_remap_src_loc_sse(Rpp32f *rowRemapTablePtr, Rpp32f *colRemapTablePtr, Rpp32s *locArray, __m128 &pStride, __m128 &pWidthLimit, __m128 &pHeightLimit, const __m128 &pChannel = xmm_p1)
{
    __m128 pRowRemapVal = _mm_loadu_ps(rowRemapTablePtr);
    pRowRemapVal = _mm_max_ps(_mm_min_ps(pRowRemapVal, pHeightLimit), xmm_p0);
    __m128 pColRemapVal = _mm_loadu_ps(colRemapTablePtr);
    pColRemapVal = _mm_max_ps(_mm_min_ps(pColRemapVal, pWidthLimit), xmm_p0);
    __m128i pxRemappedSrcLoc = _mm_cvtps_epi32(_mm_fmadd_ps(pRowRemapVal, pStride, _mm_mul_ps(pColRemapVal, pChannel)));
    _mm_storeu_si128((__m128i*) locArray, pxRemappedSrcLoc);
}

inline void compute_remap_src_loc(Rpp32f rowLoc, Rpp32f colLoc, Rpp32s &srcLoc, Rpp32s stride, Rpp32f widthLimit, Rpp32f heightLimit, Rpp32s channels = 1)
{
    rowLoc = std::max(0.0f, std::min(rowLoc, heightLimit));
    colLoc = std::max(0.0f, std::min(colLoc, widthLimit));
    srcLoc = (rowLoc * stride) + colLoc * channels;
}

inline void compute_log_16_host(__m256 *p)
{
    p[0] = log_ps(p[0]);    // log compute
    p[1] = log_ps(p[1]);    // log compute
}

inline void compute_transpose4x8_avx(__m256 *pSrc, __m128 *pDst)
{
    __m256 tmp0, tmp1, tmp2, tmp3;
    tmp0 = _mm256_shuffle_ps(pSrc[0], pSrc[1], 0x44);   /* shuffle to get [P01|P02|P09|P10|P05|P06|P13|P14] */
    tmp2 = _mm256_shuffle_ps(pSrc[0], pSrc[1], 0xEE);   /* shuffle to get [P03|P04|P11|P12|P07|P08|P15|P16] */
    tmp1 = _mm256_shuffle_ps(pSrc[2], pSrc[3], 0x44);   /* shuffle to get [P17|P18|P25|P26|P21|P22|P29|P30] */
    tmp3 = _mm256_shuffle_ps(pSrc[2], pSrc[3], 0xEE);   /* shuffle to get [P19|P20|P27|P28|P23|P24|P31|P32] */
    pSrc[0] = _mm256_shuffle_ps(tmp0, tmp1, 0x88);  /* shuffle to get [P01|P09|P17|P25|P05|P13|P21|P29] */
    pSrc[1] = _mm256_shuffle_ps(tmp0, tmp1, 0xDD);  /* shuffle to get [P02|P10|P18|P26|P06|P14|P22|P30] */
    pSrc[2] = _mm256_shuffle_ps(tmp2, tmp3, 0x88);  /* shuffle to get [P03|P11|P19|P27|P07|P15|P23|P31] */
    pSrc[3] = _mm256_shuffle_ps(tmp2, tmp3, 0xDD);  /* shuffle to get [P04|P12|P20|P28|P08|P16|P24|P32] */

    pDst[0] = _mm256_castps256_ps128(pSrc[0]);  /* extract [P01|P09|P17|P25] */
    pDst[1] = _mm256_castps256_ps128(pSrc[1]);  /* extract [P02|P10|P18|P26] */
    pDst[2] = _mm256_castps256_ps128(pSrc[2]);  /* extract [P03|P11|P19|P27] */
    pDst[3] = _mm256_castps256_ps128(pSrc[3]);  /* extract [P04|P12|P20|P28] */
    pDst[4] = _mm256_extractf128_ps(pSrc[0], 1);    /* extract [P05|P13|P21|P29] */
    pDst[5] = _mm256_extractf128_ps(pSrc[1], 1);    /* extract [P06|P14|P22|P30] */
    pDst[6] = _mm256_extractf128_ps(pSrc[2], 1);    /* extract [P07|P15|P23|P31] */
    pDst[7] = _mm256_extractf128_ps(pSrc[3], 1);    /* extract [P08|P16|P24|P32] */
}

#endif //RPP_CPU_COMMON_H
