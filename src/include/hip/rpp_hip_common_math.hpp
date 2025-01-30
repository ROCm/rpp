// /******************** DEVICE MATH HELPER FUNCTIONS ********************/

// float8 min

__device__ __forceinline__ void rpp_hip_math_min8(d_float8 *srcPtr_f8, float *dstPtr)
{
    *dstPtr = fminf(fminf(fminf(fminf(fminf(fminf(fminf(srcPtr_f8->f1[0], srcPtr_f8->f1[1]), srcPtr_f8->f1[2]), srcPtr_f8->f1[3]), srcPtr_f8->f1[4]), srcPtr_f8->f1[5]), srcPtr_f8->f1[6]), srcPtr_f8->f1[7]);
}

// float8 max

__device__ __forceinline__ void rpp_hip_math_max8(d_float8 *srcPtr_f8, float *dstPtr)
{
    *dstPtr = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(srcPtr_f8->f1[0], srcPtr_f8->f1[1]), srcPtr_f8->f1[2]), srcPtr_f8->f1[3]), srcPtr_f8->f1[4]), srcPtr_f8->f1[5]), srcPtr_f8->f1[6]), srcPtr_f8->f1[7]);
}

// d_float16 floor

__device__ __forceinline__ void rpp_hip_math_floor16(d_float16 *srcPtr_f16, d_float16 *dstPtr_f16)
{
    dstPtr_f16->f1[ 0] = floorf(srcPtr_f16->f1[ 0]);
    dstPtr_f16->f1[ 1] = floorf(srcPtr_f16->f1[ 1]);
    dstPtr_f16->f1[ 2] = floorf(srcPtr_f16->f1[ 2]);
    dstPtr_f16->f1[ 3] = floorf(srcPtr_f16->f1[ 3]);
    dstPtr_f16->f1[ 4] = floorf(srcPtr_f16->f1[ 4]);
    dstPtr_f16->f1[ 5] = floorf(srcPtr_f16->f1[ 5]);
    dstPtr_f16->f1[ 6] = floorf(srcPtr_f16->f1[ 6]);
    dstPtr_f16->f1[ 7] = floorf(srcPtr_f16->f1[ 7]);
    dstPtr_f16->f1[ 8] = floorf(srcPtr_f16->f1[ 8]);
    dstPtr_f16->f1[ 9] = floorf(srcPtr_f16->f1[ 9]);
    dstPtr_f16->f1[10] = floorf(srcPtr_f16->f1[10]);
    dstPtr_f16->f1[11] = floorf(srcPtr_f16->f1[11]);
    dstPtr_f16->f1[12] = floorf(srcPtr_f16->f1[12]);
    dstPtr_f16->f1[13] = floorf(srcPtr_f16->f1[13]);
    dstPtr_f16->f1[14] = floorf(srcPtr_f16->f1[14]);
    dstPtr_f16->f1[15] = floorf(srcPtr_f16->f1[15]);
}

// d_float8 nearbyintf

__device__ __forceinline__ void rpp_hip_math_nearbyintf8(d_float8 *srcPtr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f1[0] = nearbyintf(srcPtr_f8->f1[0]);
    dstPtr_f8->f1[1] = nearbyintf(srcPtr_f8->f1[1]);
    dstPtr_f8->f1[2] = nearbyintf(srcPtr_f8->f1[2]);
    dstPtr_f8->f1[3] = nearbyintf(srcPtr_f8->f1[3]);
    dstPtr_f8->f1[4] = nearbyintf(srcPtr_f8->f1[4]);
    dstPtr_f8->f1[5] = nearbyintf(srcPtr_f8->f1[5]);
    dstPtr_f8->f1[6] = nearbyintf(srcPtr_f8->f1[6]);
    dstPtr_f8->f1[7] = nearbyintf(srcPtr_f8->f1[7]);
}

// d_float8 add

__device__ __forceinline__ void rpp_hip_math_add8(d_float8 *src1Ptr_f8, d_float8 *src2Ptr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f4[0] = src1Ptr_f8->f4[0] + src2Ptr_f8->f4[0];
    dstPtr_f8->f4[1] = src1Ptr_f8->f4[1] + src2Ptr_f8->f4[1];
}

// d_float24 add

__device__ __forceinline__ void rpp_hip_math_add24(d_float24 *src1Ptr_f24, d_float24 *src2Ptr_f24, d_float24 *dstPtr_f24)
{
    dstPtr_f24->f4[0] = src1Ptr_f24->f4[0] + src2Ptr_f24->f4[0];
    dstPtr_f24->f4[1] = src1Ptr_f24->f4[1] + src2Ptr_f24->f4[1];
    dstPtr_f24->f4[2] = src1Ptr_f24->f4[2] + src2Ptr_f24->f4[2];
    dstPtr_f24->f4[3] = src1Ptr_f24->f4[3] + src2Ptr_f24->f4[3];
    dstPtr_f24->f4[4] = src1Ptr_f24->f4[4] + src2Ptr_f24->f4[4];
    dstPtr_f24->f4[5] = src1Ptr_f24->f4[5] + src2Ptr_f24->f4[5];
}

// d_float8 add with constant

__device__ __forceinline__ void rpp_hip_math_add8_const(d_float8 *src_f8, d_float8 *dst_f8, float4 addend_f4)
{
    dst_f8->f4[0] = src_f8->f4[0] + addend_f4;
    dst_f8->f4[1] = src_f8->f4[1] + addend_f4;
}

// d_float24 add with constant

__device__ __forceinline__ void rpp_hip_math_add24_const(d_float24 *src_f24, d_float24 *dst_f24, float4 addend_f4)
{
    dst_f24->f4[0] = src_f24->f4[0] + addend_f4;
    dst_f24->f4[1] = src_f24->f4[1] + addend_f4;
    dst_f24->f4[2] = src_f24->f4[2] + addend_f4;
    dst_f24->f4[3] = src_f24->f4[3] + addend_f4;
    dst_f24->f4[4] = src_f24->f4[4] + addend_f4;
    dst_f24->f4[5] = src_f24->f4[5] + addend_f4;
}

// d_float16 subtract

__device__ __forceinline__ void rpp_hip_math_subtract16(d_float16 *src1Ptr_f16, d_float16 *src2Ptr_f16, d_float16 *dstPtr_f16)
{
    dstPtr_f16->f4[0] = src1Ptr_f16->f4[0] - src2Ptr_f16->f4[0];
    dstPtr_f16->f4[1] = src1Ptr_f16->f4[1] - src2Ptr_f16->f4[1];
    dstPtr_f16->f4[2] = src1Ptr_f16->f4[2] - src2Ptr_f16->f4[2];
    dstPtr_f16->f4[3] = src1Ptr_f16->f4[3] - src2Ptr_f16->f4[3];
}

// d_float8 subtract with constant

__device__ __forceinline__ void rpp_hip_math_subtract8_const(d_float8 *src_f8, d_float8 *dst_f8, float4 subtrahend_f4)
{
    dst_f8->f4[0] = src_f8->f4[0] - subtrahend_f4;
    dst_f8->f4[1] = src_f8->f4[1] - subtrahend_f4;
}

// d_float24 subtract with constant

__device__ __forceinline__ void rpp_hip_math_subtract24_const(d_float24 *src_f24, d_float24 *dst_f24, float4 subtrahend_f4)
{
    dst_f24->f4[0] = src_f24->f4[0] - subtrahend_f4;
    dst_f24->f4[1] = src_f24->f4[1] - subtrahend_f4;
    dst_f24->f4[2] = src_f24->f4[2] - subtrahend_f4;
    dst_f24->f4[3] = src_f24->f4[3] - subtrahend_f4;
    dst_f24->f4[4] = src_f24->f4[4] - subtrahend_f4;
    dst_f24->f4[5] = src_f24->f4[5] - subtrahend_f4;
}

// d_float8 multiply

__device__ __forceinline__ void rpp_hip_math_multiply8(d_float8 *src1Ptr_f8, d_float8 *src2Ptr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f4[0] = src1Ptr_f8->f4[0] * src2Ptr_f8->f4[0];
    dstPtr_f8->f4[1] = src1Ptr_f8->f4[1] * src2Ptr_f8->f4[1];
}

// d_float24 multiply

__device__ __forceinline__ void rpp_hip_math_multiply24(d_float24 *src1Ptr_f24, d_float24 *src2Ptr_f24, d_float24 *dstPtr_f24)
{
    dstPtr_f24->f4[0] = src1Ptr_f24->f4[0] * src2Ptr_f24->f4[0];
    dstPtr_f24->f4[1] = src1Ptr_f24->f4[1] * src2Ptr_f24->f4[1];
    dstPtr_f24->f4[2] = src1Ptr_f24->f4[2] * src2Ptr_f24->f4[2];
    dstPtr_f24->f4[3] = src1Ptr_f24->f4[3] * src2Ptr_f24->f4[3];
    dstPtr_f24->f4[4] = src1Ptr_f24->f4[4] * src2Ptr_f24->f4[4];
    dstPtr_f24->f4[5] = src1Ptr_f24->f4[5] * src2Ptr_f24->f4[5];
}

// d_float8 multiply with constant

__device__ __forceinline__ void rpp_hip_math_multiply8_const(d_float8 *src_f8, d_float8 *dst_f8, float4 multiplier_f4)
{
    dst_f8->f4[0] = src_f8->f4[0] * multiplier_f4;
    dst_f8->f4[1] = src_f8->f4[1] * multiplier_f4;
}

// d_float24 multiply with constant

__device__ __forceinline__ void rpp_hip_math_multiply24_const(d_float24 *src_f24, d_float24 *dst_f24, float4 multiplier_f4)
{
    dst_f24->f4[0] = src_f24->f4[0] * multiplier_f4;
    dst_f24->f4[1] = src_f24->f4[1] * multiplier_f4;
    dst_f24->f4[2] = src_f24->f4[2] * multiplier_f4;
    dst_f24->f4[3] = src_f24->f4[3] * multiplier_f4;
    dst_f24->f4[4] = src_f24->f4[4] * multiplier_f4;
    dst_f24->f4[5] = src_f24->f4[5] * multiplier_f4;
}

// d_float8 divide

__device__ __forceinline__ void rpp_hip_math_divide8(d_float8 *src1Ptr_f8, d_float8 *src2Ptr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f4[0] = src1Ptr_f8->f4[0] / src2Ptr_f8->f4[0];
    dstPtr_f8->f4[1] = src1Ptr_f8->f4[1] / src2Ptr_f8->f4[1];
}

// d_float8 divide with constant

__device__ __forceinline__ void rpp_hip_math_divide8_const(d_float8 *src_f8, d_float8 *dst_f8, float4 divisor_f4)
{
    dst_f8->f4[0] = divisor_f4 / src_f8->f4[0];
    dst_f8->f4[1] = divisor_f4 / src_f8->f4[1];
}

// d_float8 bitwiseAND

__device__ __forceinline__ void rpp_hip_math_bitwiseAnd8(d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8)
{
        dst_f8->f1[0] = (float)((uchar)(src1_f8->f1[0]) & (uchar)(src2_f8->f1[0]));
        dst_f8->f1[1] = (float)((uchar)(src1_f8->f1[1]) & (uchar)(src2_f8->f1[1]));
        dst_f8->f1[2] = (float)((uchar)(src1_f8->f1[2]) & (uchar)(src2_f8->f1[2]));
        dst_f8->f1[3] = (float)((uchar)(src1_f8->f1[3]) & (uchar)(src2_f8->f1[3]));
        dst_f8->f1[4] = (float)((uchar)(src1_f8->f1[4]) & (uchar)(src2_f8->f1[4]));
        dst_f8->f1[5] = (float)((uchar)(src1_f8->f1[5]) & (uchar)(src2_f8->f1[5]));
        dst_f8->f1[6] = (float)((uchar)(src1_f8->f1[6]) & (uchar)(src2_f8->f1[6]));
        dst_f8->f1[7] = (float)((uchar)(src1_f8->f1[7]) & (uchar)(src2_f8->f1[7]));
}

// d_float8 bitwiseOR

__device__ __forceinline__ void rpp_hip_math_bitwiseOr8(d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8)
{
        dst_f8->f1[0] = (float)((uchar)(src1_f8->f1[0]) | (uchar)(src2_f8->f1[0]));
        dst_f8->f1[1] = (float)((uchar)(src1_f8->f1[1]) | (uchar)(src2_f8->f1[1]));
        dst_f8->f1[2] = (float)((uchar)(src1_f8->f1[2]) | (uchar)(src2_f8->f1[2]));
        dst_f8->f1[3] = (float)((uchar)(src1_f8->f1[3]) | (uchar)(src2_f8->f1[3]));
        dst_f8->f1[4] = (float)((uchar)(src1_f8->f1[4]) | (uchar)(src2_f8->f1[4]));
        dst_f8->f1[5] = (float)((uchar)(src1_f8->f1[5]) | (uchar)(src2_f8->f1[5]));
        dst_f8->f1[6] = (float)((uchar)(src1_f8->f1[6]) | (uchar)(src2_f8->f1[6]));
        dst_f8->f1[7] = (float)((uchar)(src1_f8->f1[7]) | (uchar)(src2_f8->f1[7]));
}

__device__ __forceinline__ void rpp_hip_math_bitwiseXor8(d_uchar8 *src1_uc8, d_uchar8 *src2_uc8, d_uchar8 *dst_uc8)
{
        dst_uc8->uc1[0] = src1_uc8->uc1[0] ^ src2_uc8->uc1[0];
        dst_uc8->uc1[1] = src1_uc8->uc1[1] ^ src2_uc8->uc1[1];
        dst_uc8->uc1[2] = src1_uc8->uc1[2] ^ src2_uc8->uc1[2];
        dst_uc8->uc1[3] = src1_uc8->uc1[3] ^ src2_uc8->uc1[3];
        dst_uc8->uc1[4] = src1_uc8->uc1[4] ^ src2_uc8->uc1[4];
        dst_uc8->uc1[5] = src1_uc8->uc1[5] ^ src2_uc8->uc1[5];
        dst_uc8->uc1[6] = src1_uc8->uc1[6] ^ src2_uc8->uc1[6];
        dst_uc8->uc1[7] = src1_uc8->uc1[7] ^ src2_uc8->uc1[7];
}

__device__ __forceinline__ float4 rpp_hip_math_inverse_sqrt4(float4 x_f4)
{
    float4 xHalf_f4 = (float4)0.5f * x_f4;
    int4 i_i4 = *(int4 *)&x_f4;                                     // float bits in int
    i_i4 = (int4) NEWTON_METHOD_INITIAL_GUESS - (i_i4 >> (int4)1);  // initial guess for Newton's method
    x_f4 = *(float4 *)&i_i4;                                        // new bits to float
    x_f4 = x_f4 * ((float4)1.5f - xHalf_f4 * x_f4 * x_f4);          // One round of Newton's method

    return x_f4;
}

__device__ __forceinline__ void rpp_hip_math_sqrt8(d_float8 *pix_f8, d_float8 *pixSqrt_f8)
{
    pixSqrt_f8->f4[0] = rpp_hip_math_inverse_sqrt4(pix_f8->f4[0]);
    pixSqrt_f8->f4[1] = rpp_hip_math_inverse_sqrt4(pix_f8->f4[1]);

    float4 one_f4 = (float4)1.0f;
    pixSqrt_f8->f4[0] = one_f4 / pixSqrt_f8->f4[0];
    pixSqrt_f8->f4[1] = one_f4 / pixSqrt_f8->f4[1];
}

__device__ __forceinline__ void rpp_hip_math_sqrt24(d_float24 *pix_f24, d_float24 *pixSqrt_f24)
{
    pixSqrt_f24->f4[0] = rpp_hip_math_inverse_sqrt4(pix_f24->f4[0]);
    pixSqrt_f24->f4[1] = rpp_hip_math_inverse_sqrt4(pix_f24->f4[1]);
    pixSqrt_f24->f4[2] = rpp_hip_math_inverse_sqrt4(pix_f24->f4[2]);
    pixSqrt_f24->f4[3] = rpp_hip_math_inverse_sqrt4(pix_f24->f4[3]);
    pixSqrt_f24->f4[4] = rpp_hip_math_inverse_sqrt4(pix_f24->f4[4]);
    pixSqrt_f24->f4[5] = rpp_hip_math_inverse_sqrt4(pix_f24->f4[5]);

    float4 one_f4 = (float4)1.0f;
    pixSqrt_f24->f4[0] = one_f4 / pixSqrt_f24->f4[0];
    pixSqrt_f24->f4[1] = one_f4 / pixSqrt_f24->f4[1];
    pixSqrt_f24->f4[2] = one_f4 / pixSqrt_f24->f4[2];
    pixSqrt_f24->f4[3] = one_f4 / pixSqrt_f24->f4[3];
    pixSqrt_f24->f4[4] = one_f4 / pixSqrt_f24->f4[4];
    pixSqrt_f24->f4[5] = one_f4 / pixSqrt_f24->f4[5];
}

__device__ __forceinline__ void rpp_hip_math_log(d_float8 *src_f8, d_float8 *dst_f8)
{
    for(int i = 0; i < 8; i++)
        src_f8->f1[i] = (!src_f8->f1[i]) ? std::nextafter(0.0f, 1.0f) : fabsf(src_f8->f1[i]);

    dst_f8->f1[0] = __logf(src_f8->f1[0]);
    dst_f8->f1[1] = __logf(src_f8->f1[1]);
    dst_f8->f1[2] = __logf(src_f8->f1[2]);
    dst_f8->f1[3] = __logf(src_f8->f1[3]);
    dst_f8->f1[4] = __logf(src_f8->f1[4]);
    dst_f8->f1[5] = __logf(src_f8->f1[5]);
    dst_f8->f1[6] = __logf(src_f8->f1[6]);
    dst_f8->f1[7] = __logf(src_f8->f1[7]);
}
