#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - box_filter device helpers --------------------

__device__ void box_filter_3x3_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8)
{
    float src_f1;
    uint3 src_ui3;
    src_ui3 = *(uint3 *)srcPtr;
    src_f1 = rpp_hip_unpack0(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.1111111f, dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.1111111f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.1111111f, dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.1111111f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.1111111f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.1111111f, dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui3.x);
    dst_f8->f1[1] = fmaf(src_f1, 0.1111111f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.1111111f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.1111111f, dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui3.y);
    dst_f8->f1[2] = fmaf(src_f1, 0.1111111f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.1111111f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.1111111f, dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui3.y);
    dst_f8->f1[3] = fmaf(src_f1, 0.1111111f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.1111111f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.1111111f, dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui3.y);
    dst_f8->f1[4] = fmaf(src_f1, 0.1111111f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.1111111f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.1111111f, dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui3.y);
    dst_f8->f1[5] = fmaf(src_f1, 0.1111111f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.1111111f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.1111111f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui3.z);
    dst_f8->f1[6] = fmaf(src_f1, 0.1111111f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.1111111f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui3.z);
    dst_f8->f1[7] = fmaf(src_f1, 0.1111111f, dst_f8->f1[7]);
}

__device__ void box_filter_5x5_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8)
{
    float src_f1;
    uint3 src_ui3;
    src_ui3 = *(uint3 *)srcPtr;
    src_f1 = rpp_hip_unpack0(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.04f, dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.04f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.04f, dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.04f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.04f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.04f, dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.04f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.04f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.04f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.04f, dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui3.y);
    dst_f8->f1[0] = fmaf(src_f1, 0.04f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.04f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.04f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.04f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.04f, dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui3.y);
    dst_f8->f1[1] = fmaf(src_f1, 0.04f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.04f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.04f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.04f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.04f, dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui3.y);
    dst_f8->f1[2] = fmaf(src_f1, 0.04f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.04f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.04f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.04f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.04f, dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui3.y);
    dst_f8->f1[3] = fmaf(src_f1, 0.04f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.04f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.04f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.04f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.04f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui3.z);
    dst_f8->f1[4] = fmaf(src_f1, 0.04f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.04f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.04f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.04f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui3.z);
    dst_f8->f1[5] = fmaf(src_f1, 0.04f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.04f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.04f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui3.z);
    dst_f8->f1[6] = fmaf(src_f1, 0.04f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.04f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui3.z);
    dst_f8->f1[7] = fmaf(src_f1, 0.04f, dst_f8->f1[7]);
}

__device__ void box_filter_7x7_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8)
{
    float src_f1;
    uint4 src_ui4 = *(uint4 *)srcPtr;
    src_f1 = rpp_hip_unpack0(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.02040816f, dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.02040816f, dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.02040816f, dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.02040816f, dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.02040816f, dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.02040816f, dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.02040816f, dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui4.y);
    dst_f8->f1[1] = fmaf(src_f1, 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.02040816f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui4.z);
    dst_f8->f1[2] = fmaf(src_f1, 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.02040816f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui4.z);
    dst_f8->f1[3] = fmaf(src_f1, 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.02040816f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui4.z);
    dst_f8->f1[4] = fmaf(src_f1, 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.02040816f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui4.z);
    dst_f8->f1[5] = fmaf(src_f1, 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.02040816f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui4.w);
    dst_f8->f1[6] = fmaf(src_f1, 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.02040816f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui4.w);
    dst_f8->f1[7] = fmaf(src_f1, 0.02040816f, dst_f8->f1[7]);
}

__device__ void box_filter_9x9_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8)
{
    float src_f1;
    uint4 src_ui4 = *(uint4 *)srcPtr;
    src_f1 = rpp_hip_unpack0(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.01234568f, dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.01234568f, dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.01234568f, dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.01234568f, dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.01234568f, dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.01234568f, dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.01234568f, dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.01234568f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui4.z);
    dst_f8->f1[0] = fmaf(src_f1, 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.01234568f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui4.z);
    dst_f8->f1[1] = fmaf(src_f1, 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.01234568f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui4.z);
    dst_f8->f1[2] = fmaf(src_f1, 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.01234568f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui4.z);
    dst_f8->f1[3] = fmaf(src_f1, 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.01234568f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui4.w);
    dst_f8->f1[4] = fmaf(src_f1, 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.01234568f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui4.w);
    dst_f8->f1[5] = fmaf(src_f1, 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.01234568f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui4.w);
    dst_f8->f1[6] = fmaf(src_f1, 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, 0.01234568f, dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui4.w);
    dst_f8->f1[7] = fmaf(src_f1, 0.01234568f, dst_f8->f1[7]);
}

// -------------------- Set 1 - PKD3->PKD3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void box_filter_3x3_pkd_hip_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    sum_f24.f4[0] = (float4) 0;
    sum_f24.f4[1] = (float4) 0;
    sum_f24.f4[2] = (float4) 0;
    sum_f24.f4[3] = (float4) 0;
    sum_f24.f4[4] = (float4) 0;
    sum_f24.f4[5] = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        *(uint2 *)src_smem_channel[0] = (uint2)0;
        *(uint2 *)src_smem_channel[1] = (uint2)0;
        *(uint2 *)src_smem_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void box_filter_5x5_pkd_hip_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    sum_f24.f4[0] = (float4) 0;
    sum_f24.f4[1] = (float4) 0;
    sum_f24.f4[2] = (float4) 0;
    sum_f24.f4[3] = (float4) 0;
    sum_f24.f4[4] = (float4) 0;
    sum_f24.f4[5] = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        *(uint2 *)src_smem_channel[0] = (uint2)0;
        *(uint2 *)src_smem_channel[1] = (uint2)0;
        *(uint2 *)src_smem_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void box_filter_7x7_pkd_hip_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    sum_f24.f4[0] = (float4) 0;
    sum_f24.f4[1] = (float4) 0;
    sum_f24.f4[2] = (float4) 0;
    sum_f24.f4[3] = (float4) 0;
    sum_f24.f4[4] = (float4) 0;
    sum_f24.f4[5] = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        *(uint2 *)src_smem_channel[0] = (uint2)0;
        *(uint2 *)src_smem_channel[1] = (uint2)0;
        *(uint2 *)src_smem_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void box_filter_9x9_pkd_hip_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    sum_f24.f4[0] = (float4) 0;
    sum_f24.f4[1] = (float4) 0;
    sum_f24.f4[2] = (float4) 0;
    sum_f24.f4[3] = (float4) 0;
    sum_f24.f4[4] = (float4) 0;
    sum_f24.f4[5] = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        *(uint2 *)src_smem_channel[0] = (uint2)0;
        *(uint2 *)src_smem_channel[1] = (uint2)0;
        *(uint2 *)src_smem_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8], &sum_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// -------------------- Set 2 - PLN1->PLN1, PLN3->PLN3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void box_filter_3x3_pln_hip_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          int channelsDst,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    sum_f8.f4[0] = (float4) 0;
    sum_f8.f4[1] = (float4) 0;
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(uint2 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
        rpp_hip_adjust_range(dstPtr, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = (float4) 0;
        sum_f8.f4[1] = (float4) 0;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
            box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = (float4) 0;
        sum_f8.f4[1] = (float4) 0;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
            box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// kernelSize = 5
template <typename T>
__global__ void box_filter_5x5_pln_hip_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          int channelsDst,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    sum_f8.f4[0] = (float4) 0;
    sum_f8.f4[1] = (float4) 0;
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(uint2 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
        rpp_hip_adjust_range(dstPtr, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = (float4) 0;
        sum_f8.f4[1] = (float4) 0;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
            box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
            box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = (float4) 0;
        sum_f8.f4[1] = (float4) 0;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
            box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
            box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// kernelSize = 7
template <typename T>
__global__ void box_filter_7x7_pln_hip_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          int channelsDst,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    sum_f8.f4[0] = (float4) 0;
    sum_f8.f4[1] = (float4) 0;
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(uint2 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
        rpp_hip_adjust_range(dstPtr, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = (float4) 0;
        sum_f8.f4[1] = (float4) 0;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = (float4) 0;
        sum_f8.f4[1] = (float4) 0;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
            box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// kernelSize = 9
template <typename T>
__global__ void box_filter_9x9_pln_hip_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          int channelsDst,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    sum_f8.f4[0] = (float4) 0;
    sum_f8.f4[1] = (float4) 0;
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(uint2 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8);
        rpp_hip_adjust_range(dstPtr, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = (float4) 0;
        sum_f8.f4[1] = (float4) 0;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = (float4) 0;
        sum_f8.f4[1] = (float4) 0;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8);
            box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// -------------------- Set 3 - PKD3->PLN3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void box_filter_3x3_pkd3_pln3_hip_tensor(T *srcPtr,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    sum_f24.f4[0] = (float4) 0;
    sum_f24.f4[1] = (float4) 0;
    sum_f24.f4[2] = (float4) 0;
    sum_f24.f4[3] = (float4) 0;
    sum_f24.f4[4] = (float4) 0;
    sum_f24.f4[5] = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        *(uint2 *)src_smem_channel[0] = (uint2)0;
        *(uint2 *)src_smem_channel[1] = (uint2)0;
        *(uint2 *)src_smem_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void box_filter_5x5_pkd3_pln3_hip_tensor(T *srcPtr,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    sum_f24.f4[0] = (float4) 0;
    sum_f24.f4[1] = (float4) 0;
    sum_f24.f4[2] = (float4) 0;
    sum_f24.f4[3] = (float4) 0;
    sum_f24.f4[4] = (float4) 0;
    sum_f24.f4[5] = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        *(uint2 *)src_smem_channel[0] = (uint2)0;
        *(uint2 *)src_smem_channel[1] = (uint2)0;
        *(uint2 *)src_smem_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void box_filter_7x7_pkd3_pln3_hip_tensor(T *srcPtr,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    sum_f24.f4[0] = (float4) 0;
    sum_f24.f4[1] = (float4) 0;
    sum_f24.f4[2] = (float4) 0;
    sum_f24.f4[3] = (float4) 0;
    sum_f24.f4[4] = (float4) 0;
    sum_f24.f4[5] = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        *(uint2 *)src_smem_channel[0] = (uint2)0;
        *(uint2 *)src_smem_channel[1] = (uint2)0;
        *(uint2 *)src_smem_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void box_filter_9x9_pkd3_pln3_hip_tensor(T *srcPtr,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    sum_f24.f4[0] = (float4) 0;
    sum_f24.f4[1] = (float4) 0;
    sum_f24.f4[2] = (float4) 0;
    sum_f24.f4[3] = (float4) 0;
    sum_f24.f4[4] = (float4) 0;
    sum_f24.f4[5] = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        *(uint2 *)src_smem_channel[0] = (uint2)0;
        *(uint2 *)src_smem_channel[1] = (uint2)0;
        *(uint2 *)src_smem_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8], &sum_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// -------------------- Set 4 - PLN3->PKD3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void box_filter_3x3_pln3_pkd3_hip_tensor(T *srcPtr,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    sum_f24.f4[0] = (float4) 0;
    sum_f24.f4[1] = (float4) 0;
    sum_f24.f4[2] = (float4) 0;
    sum_f24.f4[3] = (float4) 0;
    sum_f24.f4[4] = (float4) 0;
    sum_f24.f4[5] = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(uint2 *)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void box_filter_5x5_pln3_pkd3_hip_tensor(T *srcPtr,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    sum_f24.f4[0] = (float4) 0;
    sum_f24.f4[1] = (float4) 0;
    sum_f24.f4[2] = (float4) 0;
    sum_f24.f4[3] = (float4) 0;
    sum_f24.f4[4] = (float4) 0;
    sum_f24.f4[5] = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(uint2 *)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void box_filter_7x7_pln3_pkd3_hip_tensor(T *srcPtr,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    sum_f24.f4[0] = (float4) 0;
    sum_f24.f4[1] = (float4) 0;
    sum_f24.f4[2] = (float4) 0;
    sum_f24.f4[3] = (float4) 0;
    sum_f24.f4[4] = (float4) 0;
    sum_f24.f4[5] = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(uint2 *)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void box_filter_9x9_pln3_pkd3_hip_tensor(T *srcPtr,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
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
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    sum_f24.f4[0] = (float4) 0;
    sum_f24.f4[1] = (float4) 0;
    sum_f24.f4[2] = (float4) 0;
    sum_f24.f4[3] = (float4) 0;
    sum_f24.f4[4] = (float4) 0;
    sum_f24.f4[5] = (float4) 0;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(uint2 *)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8], &sum_f24.f8[2]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8], &sum_f24.f8[0]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8], &sum_f24.f8[1]);
        box_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8], &sum_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// -------------------- Set 5 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_box_filter_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u kernelSize,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

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

        if (kernelSize == 3)
        {
            hipLaunchKernelGGL(box_filter_3x3_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 5)
        {
            hipLaunchKernelGGL(box_filter_5x5_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 7)
        {
            hipLaunchKernelGGL(box_filter_7x7_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 9)
        {
            hipLaunchKernelGGL(box_filter_9x9_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        if (kernelSize == 3)
        {
            hipLaunchKernelGGL(box_filter_3x3_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 5)
        {
            hipLaunchKernelGGL(box_filter_5x5_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 7)
        {
            hipLaunchKernelGGL(box_filter_7x7_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 9)
        {
            hipLaunchKernelGGL(box_filter_9x9_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if (kernelSize == 3)
            {
                hipLaunchKernelGGL(box_filter_3x3_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 5)
            {
                hipLaunchKernelGGL(box_filter_5x5_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 7)
            {
                hipLaunchKernelGGL(box_filter_7x7_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 9)
            {
                hipLaunchKernelGGL(box_filter_9x9_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;

            if (kernelSize == 3)
            {
                hipLaunchKernelGGL(box_filter_3x3_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 5)
            {
                hipLaunchKernelGGL(box_filter_5x5_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 7)
            {
                hipLaunchKernelGGL(box_filter_7x7_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 9)
            {
                hipLaunchKernelGGL(box_filter_9x9_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
        }
    }

    return RPP_SUCCESS;
}
