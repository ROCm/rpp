#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"
#include "func_specific/rng_seed_stream.hpp"
#include "func_specific/sqrt_pixel_lut.hpp"

__device__ static __constant__ double __cr_lgamma_table [] = { 0.000000000000000000e-1,
    0.000000000000000000e-1,
    6.931471805599453094e-1,
    1.791759469228055001e0,
    3.178053830347945620e0,
    4.787491742782045994e0,
    6.579251212010100995e0,
    8.525161361065414300e0,
    1.060460290274525023e1};

__device__ double __cr_lgamma_integer(int a)
{
    double s;
    double t;
    double fa = fabsf((float)a);
    double sum;

    if (a > 8) {
        /* Stirling approximation; coefficients from Hart et al, "Computer
         * Approximations", Wiley 1968. Approximation 5404.
         */
        s = 1.0 / fa;
        t = s * s;
        sum =          -0.1633436431e-2;
        sum = sum * t + 0.83645878922e-3;
        sum = sum * t - 0.5951896861197e-3;
        sum = sum * t + 0.793650576493454e-3;
        sum = sum * t - 0.277777777735865004e-2;
        sum = sum * t + 0.833333333333331018375e-1;
        sum = sum * s + 0.918938533204672;
        s = 0.5 * log (fa);
        t = fa - 0.5;
        s = s * t;
        t = s - fa;
        s = s + sum;
        t = t + s;
        return t;
    } else {
// #ifdef 1
        return __cr_lgamma_table [(int) fa-1];
// #else
//         switch(a) {
//             case 1: return 0.000000000000000000e-1;
//             case 2: return 0.000000000000000000e-1;
//             case 3: return 6.931471805599453094e-1;
//             case 4: return 1.791759469228055001e0;
//             case 5: return 3.178053830347945620e0;
//             case 6: return 4.787491742782045994e0;
//             case 7: return 6.579251212010100995e0;
//             case 8: return 8.525161361065414300e0;
//             default: return 1.060460290274525023e1;
//         }
// #endif
    }
}

// Computes regularized gamma function:  gammainc(a,x)/gamma(a)
__device__ float __cr_pgammainc (float a, float x)
{
    float t, alpha, beta;

    // First level parametrization constants
    float ma1 = 1.43248035075540910f,
          ma2 = 0.12400979329415655f,
          ma3 = 0.00025361074907033f,
          mb1 = 0.21096734870196546f,
          mb2 = 1.97381164089999420f,
          mb3 = 0.94201734077887530f;

    // Second level parametrization constants (depends only on a)
    alpha = rsqrtf(a - ma2);
    alpha = ma1 * alpha + ma3;
    beta = rsqrtf(a - mb2);
    beta = mb1 * beta + mb3;

    // Final approximation (depends on a and x)
    t = a - x;
    t = alpha * t - beta;
    t = 1.0f + expf(t);
    t = t * t;
    t = __frcp_rn(t);

    // Negative a,x or a,x=NAN requires special handling
    // t = !(x > 0 && a >= 0) ? 0.0 : t;

    return t;
}

// Computes inverse of pgammainc
__device__ float __cr_pgammaincinv(float a, float y)
{
    float t, alpha, beta;

    // First level parametrization constants
    float ma1 = 1.43248035075540910f,
          ma2 = 0.12400979329415655f,
          ma3 = 0.00025361074907033f,
          mb1 = 0.21096734870196546f,
          mb2 = 1.97381164089999420f,
          mb3 = 0.94201734077887530f;

    // Second level parametrization constants (depends only on a)
    alpha = rsqrtf(a - ma2);
    alpha = ma1 * alpha + ma3;
    beta = rsqrtf(a - mb2);
    beta = mb1 * beta + mb3;

    // Final approximation (depends on a and y)
    t = rsqrtf(y) - 1.0f;
    t = __logf(t);
    t = beta + t;
    t = - t * __frcp_rn(alpha) + a;

    // Negative a,x or a,x=NAN requires special handling
    // t = !(y > 0 && a >= 0) ? 0.0 : t;
    return t;
}

__device__ void shot_noise_1_hip_compute(float *pix, RpptXorwowStateBoxMuller *xorwowState)
{
    // Knuth method
    // float shotNoiseValue = 0;                               // initialize shotNoiseValue to 0
    // float factValue = expf(*pix);                           // initialize factValue to e^lambda
    // do
    // {
    //     shotNoiseValue++;                                   // additively cumulate shotNoiseValue by 1 until exit condition
    //     factValue *= rpp_hip_rng_xorwow_f32(xorwowState);   // multiplicatively cumulate factValue by the next uniform random number until exit condition
    // } while (factValue > 1.0f);                             // loop while factValue >= 1.0f
    // *pix = shotNoiseValue - 1;

    // Inverse transform sampling
    // float shotNoiseValue = 0;
    // float factValue = expf(-*pix);
    // // float factValue = rpp_hip_math_exp_lim256approx(0 - *pix); //expf(-*pix);
    // // float factValue = rpp_hip_math_exp_approx(-*pix); //expf(-*pix);
    // float sumValue = factValue;
    // float randomNumber = rpp_hip_rng_xorwow_f32(xorwowState);
    // while (randomNumber > sumValue)
    // {
    //     shotNoiseValue += 1;
    //     factValue *= (*pix / shotNoiseValue);
    //     sumValue += factValue;
    // }
    // *pix = shotNoiseValue;

    // Gamma inverse function method
    // float y, x, t, z, v;
    // float lambda = *pix;
    // float logl = __logf(lambda);
    // while (true)
    // {
    //     y = rpp_hip_rng_xorwow_u32(xorwowState) * RPP_2POW32_INV + RPP_2POW32_INV_DIV_2;
    //     x = __cr_pgammaincinv(lambda, y);
    //     x = floorf(x);
    //     z = rpp_hip_rng_xorwow_u32(xorwowState) * RPP_2POW32_INV + RPP_2POW32_INV_DIV_2;
    //     v = (__cr_pgammainc(lambda, x + 1.0f) - __cr_pgammainc(lambda, x)) * 1.3f;
    //     z = z * v;
    //     t = (float)expf(-lambda + x * logl - (float)__cr_lgamma_integer ((int)(1.0f + x)));
    //     // if ((z < t) && (v>=1e-20))
    //         break;
    // }
    // // // return (unsigned int)x;
    // *pix = x;

    // Inverse Transform Sampling as separate function alone
    // *pix = rpp_hip_rng_1_inverse_transform_sampling_f32(*pix, xorwowState);

    // Normal Approximation as separate function alone
    // *pix = (1 / rpp_hip_math_inverse_sqrt1_const(*pix)) * rpp_hip_rng_1_gaussian_f32(xorwowState) + *pix;

    // Mixed calls to Inverse Transform Sampling or Normal Approximation
    // *pix = (*pix < 0.05) ? rpp_hip_rng_1_inverse_transform_sampling_f32(*pix, xorwowState) : (1 / rpp_hip_math_inverse_sqrt1_const(*pix)) * rpp_hip_rng_1_gaussian_f32(xorwowState) + *pix;
    if (*pix < 5)
        *pix = rpp_hip_rng_1_inverse_transform_sampling_f32(*pix, xorwowState);
    else
        *pix = (1 / rpp_hip_math_inverse_sqrt1_const(*pix)) * rpp_hip_rng_1_gaussian_f32(xorwowState) + *pix;
}

__device__ void shot_noise_2_hip_compute(float2 *pix_f2, RpptXorwowStateBoxMuller *xorwowState)
{
    float2 sqrtPix_f2;
    sqrtPix_f2.x = 1 / rpp_hip_math_inverse_sqrt1_const(pix_f2->x);
    sqrtPix_f2.y = 1 / rpp_hip_math_inverse_sqrt1_const(pix_f2->y);
    *pix_f2 = sqrtPix_f2 * rpp_hip_rng_2_gaussian_f32(xorwowState) + *pix_f2;
}

__device__ void shot_noise_8_hip_compute(d_float8 *pix_f8, RpptXorwowStateBoxMuller *xorwowState)
{
    shot_noise_1_hip_compute(&pix_f8->f1[0], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[1], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[2], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[3], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[4], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[5], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[6], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[7], xorwowState);

    // shot_noise_2_hip_compute(&pix_f8->f2[0], xorwowState);
    // shot_noise_2_hip_compute(&pix_f8->f2[1], xorwowState);
    // shot_noise_2_hip_compute(&pix_f8->f2[2], xorwowState);
    // shot_noise_2_hip_compute(&pix_f8->f2[3], xorwowState);
}

__device__ void shot_noise_24_hip_compute(d_float24 *pix_f24, RpptXorwowStateBoxMuller *xorwowState, float shotNoiseFactorInv, float shotNoiseFactor)
{
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)shotNoiseFactorInv);
    shot_noise_8_hip_compute(&pix_f24->f8[0], xorwowState);
    shot_noise_8_hip_compute(&pix_f24->f8[1], xorwowState);
    shot_noise_8_hip_compute(&pix_f24->f8[2], xorwowState);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)shotNoiseFactor);
    rpp_hip_pixel_check_0to255(pix_f24);
}

__device__ void shot_noise_adjusted_input_hip_compute(uchar *srcPtr, d_float24 *pix_f24) {}
__device__ void shot_noise_adjusted_input_hip_compute(float *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)255.0f); }
__device__ void shot_noise_adjusted_input_hip_compute(schar *srcPtr, d_float24 *pix_f24) { rpp_hip_math_add24_const(pix_f24, pix_f24, (float4)128.0f); }
__device__ void shot_noise_adjusted_input_hip_compute(half *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)255.0f); }

__device__ void shot_noise_adjusted_output_hip_compute(uchar *srcPtr, d_float24 *pix_f24) {}
__device__ void shot_noise_adjusted_output_hip_compute(float *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)ONE_OVER_255); }
__device__ void shot_noise_adjusted_output_hip_compute(schar *srcPtr, d_float24 *pix_f24) { rpp_hip_math_subtract24_const(pix_f24, pix_f24, (float4)128.0f); }
__device__ void shot_noise_adjusted_output_hip_compute(half *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)ONE_OVER_255); }

template <typename T>
__global__ void shot_noise_pkd_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      T *dstPtr,
                                      uint2 dstStridesNH,
                                      float *shotNoiseFactorTensor,
                                      RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                      uint *xorwowSeedStream,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float shotNoiseFactor = shotNoiseFactorTensor[id_z];

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    if (shotNoiseFactor != 0.0f)
    {
        float shotNoiseFactorInv = 1 / shotNoiseFactor;

        RpptXorwowStateBoxMuller xorwowState;
        uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
        xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
        xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
        xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
        xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
        xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
        xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

        shot_noise_adjusted_input_hip_compute(srcPtr, &pix_f24);
        shot_noise_24_hip_compute(&pix_f24, &xorwowState, shotNoiseFactorInv, shotNoiseFactor);
        shot_noise_adjusted_output_hip_compute(srcPtr, &pix_f24);
    }
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void shot_noise_pln_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint3 dstStridesNCH,
                                      int channelsDst,
                                      float *shotNoiseFactorTensor,
                                      RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                      uint *xorwowSeedStream,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint seedStreamIdx = (id_y * dstStridesNCH.z) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float shotNoiseFactor = shotNoiseFactorTensor[id_z];

    d_float8 pix_f8;

    // float shotNoiseFactorInv = 1 / shotNoiseFactor;

    // RpptXorwowStateBoxMuller xorwowState;
    // uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    // xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    // xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    // xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    // xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    // xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    // xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    // d_float8 pix_f8, randomNumbers_f8;
    // rpp_hip_rng_8_xorwow_f32(&xorwowState, &randomNumbers_f8);
    // // shot_noise_adjusted_input_hip_compute(srcPtr, &saltValue, &pepperValue);

    // rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
    // shot_noise_8_hip_compute(&pix_f8, shotNoiseFactor, &randomNumbers_f8);
    // rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

    // if (channelsDst == 3)
    // {
    //     srcIdx += srcStridesNCH.y;
    //     dstIdx += dstStridesNCH.y;

    //     rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
    //     shot_noise_8_hip_compute(&pix_f8, shotNoiseFactor, &randomNumbers_f8);
    //     rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

    //     srcIdx += srcStridesNCH.y;
    //     dstIdx += dstStridesNCH.y;

    //     rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
    //     shot_noise_8_hip_compute(&pix_f8, shotNoiseFactor, &randomNumbers_f8);
    //     rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
    // }
}

template <typename T>
__global__ void shot_noise_pkd3_pln3_tensor(T *srcPtr,
                                            uint2 srcStridesNH,
                                            T *dstPtr,
                                            uint3 dstStridesNCH,
                                            float *shotNoiseFactorTensor,
                                            RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                            uint *xorwowSeedStream,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint seedStreamIdx = (id_y * dstStridesNCH.z) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float shotNoiseFactor = shotNoiseFactorTensor[id_z];

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    if (shotNoiseFactor != 0.0f)
    {
        float shotNoiseFactorInv = 1 / shotNoiseFactor;

        RpptXorwowStateBoxMuller xorwowState;
        uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
        xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
        xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
        xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
        xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
        xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
        xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

        shot_noise_adjusted_input_hip_compute(srcPtr, &pix_f24);
        // shot_noise_24_hip_compute(&pix_f24, &xorwowState, shotNoiseFactorInv, shotNoiseFactor);
        shot_noise_adjusted_output_hip_compute(srcPtr, &pix_f24);
    }
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void shot_noise_pln3_pkd3_tensor(T *srcPtr,
                                            uint3 srcStridesNCH,
                                            T *dstPtr,
                                            uint2 dstStridesNH,
                                            float *shotNoiseFactorTensor,
                                            RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                            uint *xorwowSeedStream,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float shotNoiseFactor = shotNoiseFactorTensor[id_z];

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    if (shotNoiseFactor != 0.0f)
    {
        float shotNoiseFactorInv = 1 / shotNoiseFactor;

        RpptXorwowStateBoxMuller xorwowState;
        uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
        xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
        xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
        xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
        xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
        xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
        xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

        shot_noise_adjusted_input_hip_compute(srcPtr, &pix_f24);
        // shot_noise_24_hip_compute(&pix_f24, &xorwowState, shotNoiseFactorInv, shotNoiseFactor);
        shot_noise_adjusted_output_hip_compute(srcPtr, &pix_f24);
    }
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_shot_noise_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32u *xorwowSeedStream;
    xorwowSeedStream = (Rpp32u *)&xorwowInitialStatePtr[1];
    hipMemcpy(xorwowSeedStream, rngSeedStream4050, SEED_STREAM_MAX_SIZE * sizeof(Rpp32u), hipMemcpyHostToDevice);

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(shot_noise_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(shot_noise_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(shot_noise_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(shot_noise_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
