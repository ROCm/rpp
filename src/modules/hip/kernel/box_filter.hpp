#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

__device__ void box_filter_3x3_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8)
{
    uint src;
    float src_f;
    uint *src_uchar4 = (uint *)srcPtr;
    src = src_uchar4[0];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.1111111f, dst_f8->x.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.x = fmaf(src_f, 0.1111111f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.1111111f, dst_f8->x.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.x = fmaf(src_f, 0.1111111f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.1111111f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.1111111f, dst_f8->x.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.y = fmaf(src_f, 0.1111111f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.1111111f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.1111111f, dst_f8->x.w);
    src = src_uchar4[1];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.z = fmaf(src_f, 0.1111111f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.1111111f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.1111111f, dst_f8->y.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.w = fmaf(src_f, 0.1111111f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.1111111f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.1111111f, dst_f8->y.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->y.x = fmaf(src_f, 0.1111111f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.1111111f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.1111111f, dst_f8->y.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->y.y = fmaf(src_f, 0.1111111f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.1111111f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.1111111f, dst_f8->y.w);
    src = src_uchar4[2];
    src_f = rpp_hip_unpack0(src);
    dst_f8->y.z = fmaf(src_f, 0.1111111f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.1111111f, dst_f8->y.w);
    src_f = rpp_hip_unpack1(src);
    dst_f8->y.w = fmaf(src_f, 0.1111111f, dst_f8->y.w);
}

__device__ void box_filter_5x5_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8)
{
    uint src;
    float src_f;
    uint *src_uchar4 = (uint *)srcPtr;
    src = src_uchar4[0];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.04f, dst_f8->x.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.x = fmaf(src_f, 0.04f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.04f, dst_f8->x.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.x = fmaf(src_f, 0.04f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.04f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.04f, dst_f8->x.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.x = fmaf(src_f, 0.04f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.04f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.04f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.04f, dst_f8->x.w);
    src = src_uchar4[1];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.04f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.04f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.04f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.04f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.04f, dst_f8->y.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.y = fmaf(src_f, 0.04f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.04f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.04f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.04f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.04f, dst_f8->y.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.z = fmaf(src_f, 0.04f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.04f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.04f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.04f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.04f, dst_f8->y.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.w = fmaf(src_f, 0.04f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.04f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.04f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.04f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.04f, dst_f8->y.w);
    src = src_uchar4[2];
    src_f = rpp_hip_unpack0(src);
    dst_f8->y.x = fmaf(src_f, 0.04f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.04f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.04f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.04f, dst_f8->y.w);
    src_f = rpp_hip_unpack1(src);
    dst_f8->y.y = fmaf(src_f, 0.04f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.04f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.04f, dst_f8->y.w);
    src_f = rpp_hip_unpack2(src);
    dst_f8->y.z = fmaf(src_f, 0.04f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.04f, dst_f8->y.w);
    src_f = rpp_hip_unpack3(src);
    dst_f8->y.w = fmaf(src_f, 0.04f, dst_f8->y.w);
}

__device__ void box_filter_7x7_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8)
{
    uint src;
    float src_f;
    uint *src_uchar4 = (uint *)srcPtr;
    src = src_uchar4[0];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    src = src_uchar4[1];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
    src = src_uchar4[2];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
    src_f = rpp_hip_unpack2(src);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
    src_f = rpp_hip_unpack3(src);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
    src = src_uchar4[3];
    src_f = rpp_hip_unpack0(src);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
    src_f = rpp_hip_unpack1(src);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
}

__device__ void box_filter_9x9_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8)
{
    uint src;
    float src_f;
    uint *src_uchar4 = (uint *)srcPtr;
    src = src_uchar4[0];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    src = src_uchar4[1];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src = src_uchar4[2];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src = src_uchar4[3];
    src_f = rpp_hip_unpack0(src);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src_f = rpp_hip_unpack1(src);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src_f = rpp_hip_unpack2(src);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src_f = rpp_hip_unpack3(src);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
}

// Handles PKD3->PKD3 for any combination of kernelSize = 3/5/7/9 and T = U8/F32/F16/I8
template <typename T>
__global__ void box_filter_pkd_tensor(T *srcPtr,
                                      int nStrideSrc,
                                      int hStrideSrc,
                                      T *dstPtr,
                                      int nStrideDst,
                                      int hStrideDst,
                                      uint kernelSize,
                                      uint padLength,
                                      uint2 tileSize,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 sum_f24;
    __shared__ uchar src_lds[48][128];

    int srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o * 3;
    sum_f24.x.x = (float4) 0;
    sum_f24.x.y = (float4) 0;
    sum_f24.y.x = (float4) 0;
    sum_f24.y.y = (float4) 0;
    sum_f24.z.x = (float4) 0;
    sum_f24.z.y = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_lds_channel[3];
    src_lds_channel[0] = &src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_lds_channel[1] = &src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_lds_channel[2] = &src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_lds_load24_pkd3_to_pln3(srcPtr, srcIdx, src_lds_channel);
    }
    else
    {
        *(uint2 *)src_lds_channel[0] = make_uint2(0, 0);
        *(uint2 *)src_lds_channel[1] = make_uint2(0, 0);
        *(uint2 *)src_lds_channel[2] = make_uint2(0, 0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        if (kernelSize == 3)
            for(int row = 0; row < 3; row++)
            {
                box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + row][hipThreadIdx_x8], &sum_f24.x);
                box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + row][hipThreadIdx_x8], &sum_f24.y);
                box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + row][hipThreadIdx_x8], &sum_f24.z);
            }
        else if (kernelSize == 5)
            for(int row = 0; row < 5; row++)
            {
                box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + row][hipThreadIdx_x8], &sum_f24.x);
                box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + row][hipThreadIdx_x8], &sum_f24.y);
                box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + row][hipThreadIdx_x8], &sum_f24.z);
            }
        else if (kernelSize == 7)
            for(int row = 0; row < 7; row++)
            {
                box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + row][hipThreadIdx_x8], &sum_f24.x);
                box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + row][hipThreadIdx_x8], &sum_f24.y);
                box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + row][hipThreadIdx_x8], &sum_f24.z);
            }
        else if (kernelSize == 9)
            for(int row = 0; row < 9; row++)
            {
                box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + row][hipThreadIdx_x8], &sum_f24.x);
                box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + row][hipThreadIdx_x8], &sum_f24.y);
                box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + row][hipThreadIdx_x8], &sum_f24.z);
            }
        rpp_hip_adjust_range(dstPtr, &sum_f24.x);
        rpp_hip_adjust_range(dstPtr, &sum_f24.y);
        rpp_hip_adjust_range(dstPtr, &sum_f24.z);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr, dstIdx, &sum_f24);
    }
}

// Handles PLN1->PLN1, PLN3->PLN3 for any combination of kernelSize = 3/5/7/9 and T = U8/F32/F16/I8
template <typename T>
__global__ void box_filter_pln_tensor(T *srcPtr,
                                      int nStrideSrc,
                                      int cStrideSrc,
                                      int hStrideSrc,
                                      T *dstPtr,
                                      int nStrideDst,
                                      int cStrideDst,
                                      int hStrideDst,
                                      int channelsDst,
                                      uint kernelSize,
                                      uint padLength,
                                      uint2 tileSize,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ uchar src_lds[16][128];

    int srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;
    sum_f8.x = (float4) 0;
    sum_f8.y = (float4) 0;
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_lds_load8(srcPtr, srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = make_uint2(0, 0);
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        if (kernelSize == 3)
            for(int row = 0; row < 3; row++)
                box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
        else if (kernelSize == 5)
            for(int row = 0; row < 5; row++)
                box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
        else if (kernelSize == 7)
            for(int row = 0; row < 7; row++)
                box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
        else if (kernelSize == 9)
            for(int row = 0; row < 9; row++)
                box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
        rpp_hip_adjust_range(dstPtr, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;
        sum_f8.x = (float4) 0;
        sum_f8.y = (float4) 0;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_lds_load8(srcPtr, srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = make_uint2(0, 0);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            if (kernelSize == 3)
                for(int row = 0; row < 3; row++)
                    box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            else if (kernelSize == 5)
                for(int row = 0; row < 5; row++)
                    box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            else if (kernelSize == 7)
                for(int row = 0; row < 7; row++)
                    box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            else if (kernelSize == 9)
                for(int row = 0; row < 9; row++)
                    box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
        }

        __syncthreads();
        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;
        sum_f8.x = (float4) 0;
        sum_f8.y = (float4) 0;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_lds_load8(srcPtr, srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = make_uint2(0, 0);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            if (kernelSize == 3)
                for(int row = 0; row < 3; row++)
                    box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            else if (kernelSize == 5)
                for(int row = 0; row < 5; row++)
                    box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            else if (kernelSize == 7)
                for(int row = 0; row < 7; row++)
                    box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            else if (kernelSize == 9)
                for(int row = 0; row < 9; row++)
                    box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
        }
    }
}

// Handles PKD3->PLN3 for any combination of kernelSize = 3/5/7/9 and T = U8/F32/F16/I8
template <typename T>
__global__ void box_filter_pkd3_pln3_tensor(T *srcPtr,
                                            int nStrideSrc,
                                            int hStrideSrc,
                                            T *dstPtr,
                                            int nStrideDst,
                                            int cStrideDst,
                                            int hStrideDst,
                                            uint kernelSize,
                                            uint padLength,
                                            uint2 tileSize,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 sum_f24;
    __shared__ uchar src_lds[48][128];

    int srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;
    sum_f24.x.x = (float4) 0;
    sum_f24.x.y = (float4) 0;
    sum_f24.y.x = (float4) 0;
    sum_f24.y.y = (float4) 0;
    sum_f24.z.x = (float4) 0;
    sum_f24.z.y = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_lds_channel[3];
    src_lds_channel[0] = &src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_lds_channel[1] = &src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_lds_channel[2] = &src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_lds_load24_pkd3_to_pln3(srcPtr, srcIdx, src_lds_channel);
    }
    else
    {
        *(uint2 *)src_lds_channel[0] = make_uint2(0, 0);
        *(uint2 *)src_lds_channel[1] = make_uint2(0, 0);
        *(uint2 *)src_lds_channel[2] = make_uint2(0, 0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        if (kernelSize == 3)
            for(int row = 0; row < 3; row++)
            {
                box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + row][hipThreadIdx_x8], &sum_f24.x);
                box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + row][hipThreadIdx_x8], &sum_f24.y);
                box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + row][hipThreadIdx_x8], &sum_f24.z);
            }
        else if (kernelSize == 5)
            for(int row = 0; row < 5; row++)
            {
                box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + row][hipThreadIdx_x8], &sum_f24.x);
                box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + row][hipThreadIdx_x8], &sum_f24.y);
                box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + row][hipThreadIdx_x8], &sum_f24.z);
            }
        else if (kernelSize == 7)
            for(int row = 0; row < 7; row++)
            {
                box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + row][hipThreadIdx_x8], &sum_f24.x);
                box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + row][hipThreadIdx_x8], &sum_f24.y);
                box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + row][hipThreadIdx_x8], &sum_f24.z);
            }
        else if (kernelSize == 9)
            for(int row = 0; row < 9; row++)
            {
                box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + row][hipThreadIdx_x8], &sum_f24.x);
                box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + row][hipThreadIdx_x8], &sum_f24.y);
                box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + row][hipThreadIdx_x8], &sum_f24.z);
            }
        rpp_hip_adjust_range(dstPtr, &sum_f24.x);
        rpp_hip_adjust_range(dstPtr, &sum_f24.y);
        rpp_hip_adjust_range(dstPtr, &sum_f24.z);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr, dstIdx, cStrideDst, &sum_f24);
    }
}

// Handles PLN3->PKD3 for any combination of kernelSize = 3/5/7/9 and T = U8/F32/F16/I8
template <typename T>
__global__ void box_filter_pln3_pkd3_tensor(T *srcPtr,
                                            int nStrideSrc,
                                            int cStrideSrc,
                                            int hStrideSrc,
                                            T *dstPtr,
                                            int nStrideDst,
                                            int hStrideDst,
                                            uint kernelSize,
                                            uint padLength,
                                            uint2 tileSize,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 sum_f24;
    __shared__ uchar src_lds[48][128];

    int3 srcIdx;
    srcIdx.x = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + cStrideSrc;
    srcIdx.z = srcIdx.y + cStrideSrc;
    int dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o * 3;
    sum_f24.x.x = (float4) 0;
    sum_f24.x.y = (float4) 0;
    sum_f24.y.x = (float4) 0;
    sum_f24.y.y = (float4) 0;
    sum_f24.z.x = (float4) 0;
    sum_f24.z.y = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_lds_load8(srcPtr, srcIdx.x, &src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_lds_load8(srcPtr, srcIdx.y, &src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_lds_load8(srcPtr, srcIdx.z, &src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = make_uint2(0, 0);
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = make_uint2(0, 0);
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = make_uint2(0, 0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        if (kernelSize == 3)
            for(int row = 0; row < 3; row++)
            {
                box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + row][hipThreadIdx_x8], &sum_f24.x);
                box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + row][hipThreadIdx_x8], &sum_f24.y);
                box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + row][hipThreadIdx_x8], &sum_f24.z);
            }
        else if (kernelSize == 5)
            for(int row = 0; row < 5; row++)
            {
                box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + row][hipThreadIdx_x8], &sum_f24.x);
                box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + row][hipThreadIdx_x8], &sum_f24.y);
                box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + row][hipThreadIdx_x8], &sum_f24.z);
            }
        else if (kernelSize == 7)
            for(int row = 0; row < 7; row++)
            {
                box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + row][hipThreadIdx_x8], &sum_f24.x);
                box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + row][hipThreadIdx_x8], &sum_f24.y);
                box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + row][hipThreadIdx_x8], &sum_f24.z);
            }
        else if (kernelSize == 9)
            for(int row = 0; row < 9; row++)
            {
                box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + row][hipThreadIdx_x8], &sum_f24.x);
                box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + row][hipThreadIdx_x8], &sum_f24.y);
                box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + row][hipThreadIdx_x8], &sum_f24.z);
            }
        rpp_hip_adjust_range(dstPtr, &sum_f24.x);
        rpp_hip_adjust_range(dstPtr, &sum_f24.y);
        rpp_hip_adjust_range(dstPtr, &sum_f24.z);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr, dstIdx, &sum_f24);
    }
}

template <typename T>
RppStatus hip_exec_box_filter_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u kernelSize,
                                     RpptROIPtr roiTensorPtrSrc,
                                     rpp::Handle& handle)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (128 - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(box_filter_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcDescPtr->strides.nStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           dstDescPtr->strides.hStride,
                           kernelSize,
                           padLength,
                           tileSize,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(box_filter_pln_tensor,
                           dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcDescPtr->strides.nStride,
                           srcDescPtr->strides.cStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           dstDescPtr->strides.cStride,
                           dstDescPtr->strides.hStride,
                           dstDescPtr->c,
                           kernelSize,
                           padLength,
                           tileSize,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(box_filter_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               srcDescPtr->strides.nStride,
                               srcDescPtr->strides.hStride,
                               dstPtr,
                               dstDescPtr->strides.nStride,
                               dstDescPtr->strides.cStride,
                               dstDescPtr->strides.hStride,
                               kernelSize,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(box_filter_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               srcDescPtr->strides.nStride,
                               srcDescPtr->strides.cStride,
                               srcDescPtr->strides.hStride,
                               dstPtr,
                               dstDescPtr->strides.nStride,
                               dstDescPtr->strides.hStride,
                               kernelSize,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
