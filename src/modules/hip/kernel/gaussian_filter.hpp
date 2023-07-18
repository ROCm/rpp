#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - gaussian_filter device helpers --------------------

__device__ void gaussian_filter_3x3_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8, float *filter)
{
    float src_f1;
    uint3 src_ui3;
    src_ui3 = *(reinterpret_cast<uint3 *>(srcPtr));
    src_f1 = rpp_hip_unpack0(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[0], dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[0], dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[0], dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui3.x);
    dst_f8->f1[1] = fmaf(src_f1, filter[2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[0], dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui3.y);
    dst_f8->f1[2] = fmaf(src_f1, filter[2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[0], dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui3.y);
    dst_f8->f1[3] = fmaf(src_f1, filter[2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[0], dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui3.y);
    dst_f8->f1[4] = fmaf(src_f1, filter[2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[0], dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui3.y);
    dst_f8->f1[5] = fmaf(src_f1, filter[2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[0], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui3.z);
    dst_f8->f1[6] = fmaf(src_f1, filter[2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[1], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui3.z);
    dst_f8->f1[7] = fmaf(src_f1, filter[2], dst_f8->f1[7]);
}

__device__ void gaussian_filter_5x5_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8, float *filter)
{
    float src_f1;
    uint3 src_ui3;
    src_ui3 = *(reinterpret_cast<uint3 *>(srcPtr));
    src_f1 = rpp_hip_unpack0(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[0], dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[0], dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[0], dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[3], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[0], dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui3.y);
    dst_f8->f1[0] = fmaf(src_f1, filter[4], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[3], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[0], dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui3.y);
    dst_f8->f1[1] = fmaf(src_f1, filter[4], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[3], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[0], dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui3.y);
    dst_f8->f1[2] = fmaf(src_f1, filter[4], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[3], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[0], dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui3.y);
    dst_f8->f1[3] = fmaf(src_f1, filter[4], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[3], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[0], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui3.z);
    dst_f8->f1[4] = fmaf(src_f1, filter[4], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[3], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[1], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui3.z);
    dst_f8->f1[5] = fmaf(src_f1, filter[4], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[3], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[2], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui3.z);
    dst_f8->f1[6] = fmaf(src_f1, filter[4], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[3], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui3.z);
    dst_f8->f1[7] = fmaf(src_f1, filter[4], dst_f8->f1[7]);
}

__device__ void gaussian_filter_7x7_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8, float *filter)
{
    float src_f1;
    uint4 src_ui4 = *(reinterpret_cast<uint4 *>(srcPtr));
    src_f1 = rpp_hip_unpack0(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[0], dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[0], dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[0], dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[3], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[0], dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter[4], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[3], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[0], dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter[5], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[4], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[3], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[0], dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter[6], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[5], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[4], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[3], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[0], dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui4.y);
    dst_f8->f1[1] = fmaf(src_f1, filter[6], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[5], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[4], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[3], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[0], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui4.z);
    dst_f8->f1[2] = fmaf(src_f1, filter[6], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[5], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[4], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[3], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[1], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui4.z);
    dst_f8->f1[3] = fmaf(src_f1, filter[6], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[5], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[4], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[3], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[2], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui4.z);
    dst_f8->f1[4] = fmaf(src_f1, filter[6], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[5], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[4], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[3], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui4.z);
    dst_f8->f1[5] = fmaf(src_f1, filter[6], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[5], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[4], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui4.w);
    dst_f8->f1[6] = fmaf(src_f1, filter[6], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[5], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui4.w);
    dst_f8->f1[7] = fmaf(src_f1, filter[6], dst_f8->f1[7]);
}

__device__ void gaussian_filter_9x9_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8, float *filter)
{
    float src_f1;
    uint4 src_ui4 = *(reinterpret_cast<uint4 *>(srcPtr));
    src_f1 = rpp_hip_unpack0(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[0], dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[0], dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[0], dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[3], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[0], dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter[4], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[3], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[0], dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter[5], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[4], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[3], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[0], dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter[6], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[5], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[4], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[3], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[0], dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter[7], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[6], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[5], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[4], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[3], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[0], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui4.z);
    dst_f8->f1[0] = fmaf(src_f1, filter[8], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[7], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[6], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[5], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[4], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[3], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[1], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui4.z);
    dst_f8->f1[1] = fmaf(src_f1, filter[8], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[7], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[6], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[5], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[4], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[3], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[2], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui4.z);
    dst_f8->f1[2] = fmaf(src_f1, filter[8], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[7], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[6], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[5], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[4], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[3], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui4.z);
    dst_f8->f1[3] = fmaf(src_f1, filter[8], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[7], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[6], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[5], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[4], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui4.w);
    dst_f8->f1[4] = fmaf(src_f1, filter[8], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[7], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[6], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[5], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui4.w);
    dst_f8->f1[5] = fmaf(src_f1, filter[8], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[7], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[6], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui4.w);
    dst_f8->f1[6] = fmaf(src_f1, filter[8], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[7], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui4.w);
    dst_f8->f1[7] = fmaf(src_f1, filter[8], dst_f8->f1[7]);
}

// -------------------- Set 1 - PKD3->PKD3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void gaussian_filter_3x3_pkd_tensor(T *srcPtr,
                                               uint2 srcStridesNH,
                                               T *dstPtr,
                                               uint2 dstStridesNH,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               d_float9 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 sum_f24;
    __shared__ uchar src_lds[48][128];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    d_float9 filter_f9 = filterTensor[id_z];
    float *filter_row1 = &filter_f9.f1[0];
    float *filter_row2 = &filter_f9.f1[3];
    float *filter_row3 = &filter_f9.f1[6];
    sum_f24.f4[0] = static_cast<float4>(0);
    sum_f24.f4[1] = static_cast<float4>(0);
    sum_f24.f4[2] = static_cast<float4>(0);
    sum_f24.f4[3] = static_cast<float4>(0);
    sum_f24.f4[4] = static_cast<float4>(0);
    sum_f24.f4[5] = static_cast<float4>(0);

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
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_lds_channel);
    }
    else
    {
        *(reinterpret_cast<uint2 *>(src_lds_channel[0])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[1])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[2])) = static_cast<uint2>(0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], filter_row1);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], filter_row1);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], filter_row1);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], filter_row2);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], filter_row2);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], filter_row2);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], filter_row3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], filter_row3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], filter_row3);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void gaussian_filter_5x5_pkd_tensor(T *srcPtr,
                                               uint2 srcStridesNH,
                                               T *dstPtr,
                                               uint2 dstStridesNH,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               d_float25 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 sum_f24;
    __shared__ uchar src_lds[48][128];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    d_float25 filter_f25 = filterTensor[id_z];
    float *filter_row1 = &filter_f25.f1[0];
    float *filter_row2 = &filter_f25.f1[5];
    float *filter_row3 = &filter_f25.f1[10];
    float *filter_row4 = &filter_f25.f1[15];
    float *filter_row5 = &filter_f25.f1[20];
    sum_f24.f4[0] = static_cast<float4>(0);
    sum_f24.f4[1] = static_cast<float4>(0);
    sum_f24.f4[2] = static_cast<float4>(0);
    sum_f24.f4[3] = static_cast<float4>(0);
    sum_f24.f4[4] = static_cast<float4>(0);
    sum_f24.f4[5] = static_cast<float4>(0);

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
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_lds_channel);
    }
    else
    {
        *(reinterpret_cast<uint2 *>(src_lds_channel[0])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[1])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[2])) = static_cast<uint2>(0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], filter_row1);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], filter_row1);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], filter_row1);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], filter_row2);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], filter_row2);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], filter_row2);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], filter_row3);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], filter_row3);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], filter_row3);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], filter_row4);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], filter_row4);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], filter_row4);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], filter_row5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], filter_row5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], filter_row5);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void gaussian_filter_7x7_pkd_tensor(T *srcPtr,
                                               uint2 srcStridesNH,
                                               T *dstPtr,
                                               uint2 dstStridesNH,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               d_float49 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 sum_f24;
    __shared__ uchar src_lds[48][128];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    d_float49 filter_f49 = filterTensor[id_z];
    float *filter_row1 = &filter_f49.f1[0];
    float *filter_row2 = &filter_f49.f1[7];
    float *filter_row3 = &filter_f49.f1[14];
    float *filter_row4 = &filter_f49.f1[21];
    float *filter_row5 = &filter_f49.f1[28];
    float *filter_row6 = &filter_f49.f1[35];
    float *filter_row7 = &filter_f49.f1[42];
    sum_f24.f4[0] = static_cast<float4>(0);
    sum_f24.f4[1] = static_cast<float4>(0);
    sum_f24.f4[2] = static_cast<float4>(0);
    sum_f24.f4[3] = static_cast<float4>(0);
    sum_f24.f4[4] = static_cast<float4>(0);
    sum_f24.f4[5] = static_cast<float4>(0);

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
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_lds_channel);
    }
    else
    {
        *(reinterpret_cast<uint2 *>(src_lds_channel[0])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[1])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[2])) = static_cast<uint2>(0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], filter_row1);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], filter_row1);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], filter_row1);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], filter_row2);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], filter_row2);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], filter_row2);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], filter_row3);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], filter_row3);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], filter_row3);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], filter_row4);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], filter_row4);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], filter_row4);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], filter_row5);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], filter_row5);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], filter_row5);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0], filter_row6);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1], filter_row6);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2], filter_row6);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0], filter_row7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1], filter_row7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2], filter_row7);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void gaussian_filter_9x9_pkd_tensor(T *srcPtr,
                                               uint2 srcStridesNH,
                                               T *dstPtr,
                                               uint2 dstStridesNH,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               d_float81 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 sum_f24;
    __shared__ uchar src_lds[48][128];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    d_float81 filter_f81 = filterTensor[id_z];
    float *filter_row1 = &filter_f81.f1[0];
    float *filter_row2 = &filter_f81.f1[9];
    float *filter_row3 = &filter_f81.f1[18];
    float *filter_row4 = &filter_f81.f1[27];
    float *filter_row5 = &filter_f81.f1[36];
    float *filter_row6 = &filter_f81.f1[45];
    float *filter_row7 = &filter_f81.f1[54];
    float *filter_row8 = &filter_f81.f1[63];
    float *filter_row9 = &filter_f81.f1[72];
    sum_f24.f4[0] = static_cast<float4>(0);
    sum_f24.f4[1] = static_cast<float4>(0);
    sum_f24.f4[2] = static_cast<float4>(0);
    sum_f24.f4[3] = static_cast<float4>(0);
    sum_f24.f4[4] = static_cast<float4>(0);
    sum_f24.f4[5] = static_cast<float4>(0);

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
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_lds_channel);
    }
    else
    {
        *(reinterpret_cast<uint2 *>(src_lds_channel[0])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[1])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[2])) = static_cast<uint2>(0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], filter_row1);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], filter_row1);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], filter_row1);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], filter_row2);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], filter_row2);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], filter_row2);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], filter_row3);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], filter_row3);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], filter_row3);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], filter_row4);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], filter_row4);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], filter_row4);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], filter_row5);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], filter_row5);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], filter_row5);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0], filter_row6);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1], filter_row6);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2], filter_row6);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0], filter_row7);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1], filter_row7);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2], filter_row7);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8], &sum_f24.f8[0], filter_row8);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8], &sum_f24.f8[1], filter_row8);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8], &sum_f24.f8[2], filter_row8);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8], &sum_f24.f8[0], filter_row9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8], &sum_f24.f8[1], filter_row9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8], &sum_f24.f8[2], filter_row9);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// -------------------- Set 2 - PLN1->PLN1, PLN3->PLN3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void gaussian_filter_3x3_pln_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               int channelsDst,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               d_float9 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    d_float8 sum_f8;
    __shared__ uchar src_lds[16][128];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    d_float9 filter_f9 = filterTensor[id_z];
    float *filter_row1 = &filter_f9.f1[0];
    float *filter_row2 = &filter_f9.f1[3];
    float *filter_row3 = &filter_f9.f1[6];
    sum_f8.f4[0] = static_cast<float4>(0);
    sum_f8.f4[1] = static_cast<float4>(0);
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y][hipThreadIdx_x8])) = static_cast<uint2>(0);
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        rpp_hip_adjust_range(dstPtr, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = static_cast<float4>(0);
        sum_f8.f4[1] = static_cast<float4>(0);
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y][hipThreadIdx_x8])) = static_cast<uint2>(0);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
            gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
            gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = static_cast<float4>(0);
        sum_f8.f4[1] = static_cast<float4>(0);
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y][hipThreadIdx_x8])) = static_cast<uint2>(0);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
            gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
            gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// kernelSize = 5
template <typename T>
__global__ void gaussian_filter_5x5_pln_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               int channelsDst,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               d_float25 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ uchar src_lds[16][128];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    d_float25 filter_f25 = filterTensor[id_z];
    float *filter_row1 = &filter_f25.f1[0];
    float *filter_row2 = &filter_f25.f1[5];
    float *filter_row3 = &filter_f25.f1[10];
    float *filter_row4 = &filter_f25.f1[15];
    float *filter_row5 = &filter_f25.f1[20];
    sum_f8.f4[0] = static_cast<float4>(0);
    sum_f8.f4[1] = static_cast<float4>(0);
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y][hipThreadIdx_x8])) = static_cast<uint2>(0);
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
        rpp_hip_adjust_range(dstPtr, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = static_cast<float4>(0);
        sum_f8.f4[1] = static_cast<float4>(0);
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y][hipThreadIdx_x8])) = static_cast<uint2>(0);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = static_cast<float4>(0);
        sum_f8.f4[1] = static_cast<float4>(0);
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y][hipThreadIdx_x8])) = static_cast<uint2>(0);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// kernelSize = 7
template <typename T>
__global__ void gaussian_filter_7x7_pln_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               int channelsDst,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               d_float49 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ uchar src_lds[16][128];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    d_float49 filter_f49 = filterTensor[id_z];
    float *filter_row1 = &filter_f49.f1[0];
    float *filter_row2 = &filter_f49.f1[7];
    float *filter_row3 = &filter_f49.f1[14];
    float *filter_row4 = &filter_f49.f1[21];
    float *filter_row5 = &filter_f49.f1[28];
    float *filter_row6 = &filter_f49.f1[35];
    float *filter_row7 = &filter_f49.f1[42];
    sum_f8.f4[0] = static_cast<float4>(0);
    sum_f8.f4[1] = static_cast<float4>(0);
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y][hipThreadIdx_x8])) = static_cast<uint2>(0);
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, filter_row6);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, filter_row7);
        rpp_hip_adjust_range(dstPtr, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = static_cast<float4>(0);
        sum_f8.f4[1] = static_cast<float4>(0);
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y][hipThreadIdx_x8])) = static_cast<uint2>(0);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, filter_row6);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, filter_row7);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = static_cast<float4>(0);
        sum_f8.f4[1] = static_cast<float4>(0);
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y][hipThreadIdx_x8])) = static_cast<uint2>(0);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, filter_row6);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, filter_row7);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// kernelSize = 9
template <typename T>
__global__ void gaussian_filter_9x9_pln_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               int channelsDst,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               d_float81 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ uchar src_lds[16][128];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    d_float81 filter_f81 = filterTensor[id_z];
    float *filter_row1 = &filter_f81.f1[0];
    float *filter_row2 = &filter_f81.f1[9];
    float *filter_row3 = &filter_f81.f1[18];
    float *filter_row4 = &filter_f81.f1[27];
    float *filter_row5 = &filter_f81.f1[36];
    float *filter_row6 = &filter_f81.f1[45];
    float *filter_row7 = &filter_f81.f1[54];
    float *filter_row8 = &filter_f81.f1[63];
    float *filter_row9 = &filter_f81.f1[72];
    sum_f8.f4[0] = static_cast<float4>(0);
    sum_f8.f4[1] = static_cast<float4>(0);
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y][hipThreadIdx_x8])) = static_cast<uint2>(0);
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, filter_row6);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, filter_row7);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8, filter_row8);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8, filter_row9);
        rpp_hip_adjust_range(dstPtr, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = static_cast<float4>(0);
        sum_f8.f4[1] = static_cast<float4>(0);
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y][hipThreadIdx_x8])) = static_cast<uint2>(0);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, filter_row6);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, filter_row7);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8, filter_row8);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8, filter_row9);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = static_cast<float4>(0);
        sum_f8.f4[1] = static_cast<float4>(0);
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y][hipThreadIdx_x8])) = static_cast<uint2>(0);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, filter_row6);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, filter_row7);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8, filter_row8);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8, filter_row9);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// -------------------- Set 3 - PKD3->PLN3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void gaussian_filter_3x3_pkd3_pln3_tensor(T *srcPtr,
                                                     uint2 srcStridesNH,
                                                     T *dstPtr,
                                                     uint3 dstStridesNCH,
                                                     uint padLength,
                                                     uint2 tileSize,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     d_float9 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 sum_f24;
    __shared__ uchar src_lds[48][128];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    d_float9 filter_f9 = filterTensor[id_z];
    float *filter_row1 = &filter_f9.f1[0];
    float *filter_row2 = &filter_f9.f1[3];
    float *filter_row3 = &filter_f9.f1[6];
    sum_f24.f4[0] = static_cast<float4>(0);
    sum_f24.f4[1] = static_cast<float4>(0);
    sum_f24.f4[2] = static_cast<float4>(0);
    sum_f24.f4[3] = static_cast<float4>(0);
    sum_f24.f4[4] = static_cast<float4>(0);
    sum_f24.f4[5] = static_cast<float4>(0);

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
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_lds_channel);
    }
    else
    {
        *(reinterpret_cast<uint2 *>(src_lds_channel[0])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[1])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[2])) = static_cast<uint2>(0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], filter_row1);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], filter_row1);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], filter_row1);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], filter_row2);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], filter_row2);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], filter_row2);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], filter_row3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], filter_row3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], filter_row3);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void gaussian_filter_5x5_pkd3_pln3_tensor(T *srcPtr,
                                                     uint2 srcStridesNH,
                                                     T *dstPtr,
                                                     uint3 dstStridesNCH,
                                                     uint padLength,
                                                     uint2 tileSize,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     d_float25 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 sum_f24;
    __shared__ uchar src_lds[48][128];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    d_float25 filter_f25 = filterTensor[id_z];
    float *filter_row1 = &filter_f25.f1[0];
    float *filter_row2 = &filter_f25.f1[5];
    float *filter_row3 = &filter_f25.f1[10];
    float *filter_row4 = &filter_f25.f1[15];
    float *filter_row5 = &filter_f25.f1[20];
    sum_f24.f4[0] = static_cast<float4>(0);
    sum_f24.f4[1] = static_cast<float4>(0);
    sum_f24.f4[2] = static_cast<float4>(0);
    sum_f24.f4[3] = static_cast<float4>(0);
    sum_f24.f4[4] = static_cast<float4>(0);
    sum_f24.f4[5] = static_cast<float4>(0);

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
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_lds_channel);
    }
    else
    {
        *(reinterpret_cast<uint2 *>(src_lds_channel[0])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[1])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[2])) = static_cast<uint2>(0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], filter_row1);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], filter_row1);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], filter_row1);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], filter_row2);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], filter_row2);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], filter_row2);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], filter_row3);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], filter_row3);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], filter_row3);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], filter_row4);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], filter_row4);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], filter_row4);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], filter_row5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], filter_row5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], filter_row5);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void gaussian_filter_7x7_pkd3_pln3_tensor(T *srcPtr,
                                                     uint2 srcStridesNH,
                                                     T *dstPtr,
                                                     uint3 dstStridesNCH,
                                                     uint padLength,
                                                     uint2 tileSize,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     d_float49 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 sum_f24;
    __shared__ uchar src_lds[48][128];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    d_float49 filter_f49 = filterTensor[id_z];
    float *filter_row1 = &filter_f49.f1[0];
    float *filter_row2 = &filter_f49.f1[7];
    float *filter_row3 = &filter_f49.f1[14];
    float *filter_row4 = &filter_f49.f1[21];
    float *filter_row5 = &filter_f49.f1[28];
    float *filter_row6 = &filter_f49.f1[35];
    float *filter_row7 = &filter_f49.f1[42];
    sum_f24.f4[0] = static_cast<float4>(0);
    sum_f24.f4[1] = static_cast<float4>(0);
    sum_f24.f4[2] = static_cast<float4>(0);
    sum_f24.f4[3] = static_cast<float4>(0);
    sum_f24.f4[4] = static_cast<float4>(0);
    sum_f24.f4[5] = static_cast<float4>(0);

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
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_lds_channel);
    }
    else
    {
        *(reinterpret_cast<uint2 *>(src_lds_channel[0])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[1])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[2])) = static_cast<uint2>(0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], filter_row1);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], filter_row1);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], filter_row1);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], filter_row2);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], filter_row2);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], filter_row2);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], filter_row3);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], filter_row3);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], filter_row3);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], filter_row4);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], filter_row4);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], filter_row4);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], filter_row5);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], filter_row5);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], filter_row5);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0], filter_row6);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1], filter_row6);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2], filter_row6);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0], filter_row7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1], filter_row7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2], filter_row7);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void gaussian_filter_9x9_pkd3_pln3_tensor(T *srcPtr,
                                                     uint2 srcStridesNH,
                                                     T *dstPtr,
                                                     uint3 dstStridesNCH,
                                                     uint padLength,
                                                     uint2 tileSize,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     d_float81 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 sum_f24;
    __shared__ uchar src_lds[48][128];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    d_float81 filter_f81 = filterTensor[id_z];
    float *filter_row1 = &filter_f81.f1[0];
    float *filter_row2 = &filter_f81.f1[9];
    float *filter_row3 = &filter_f81.f1[18];
    float *filter_row4 = &filter_f81.f1[27];
    float *filter_row5 = &filter_f81.f1[36];
    float *filter_row6 = &filter_f81.f1[45];
    float *filter_row7 = &filter_f81.f1[54];
    float *filter_row8 = &filter_f81.f1[63];
    float *filter_row9 = &filter_f81.f1[72];
    sum_f24.f4[0] = static_cast<float4>(0);
    sum_f24.f4[1] = static_cast<float4>(0);
    sum_f24.f4[2] = static_cast<float4>(0);
    sum_f24.f4[3] = static_cast<float4>(0);
    sum_f24.f4[4] = static_cast<float4>(0);
    sum_f24.f4[5] = static_cast<float4>(0);

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
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_lds_channel);
    }
    else
    {
        *(reinterpret_cast<uint2 *>(src_lds_channel[0])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[1])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(src_lds_channel[2])) = static_cast<uint2>(0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], filter_row1);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], filter_row1);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], filter_row1);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], filter_row2);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], filter_row2);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], filter_row2);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], filter_row3);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], filter_row3);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], filter_row3);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], filter_row4);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], filter_row4);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], filter_row4);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], filter_row5);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], filter_row5);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], filter_row5);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0], filter_row6);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1], filter_row6);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2], filter_row6);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0], filter_row7);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1], filter_row7);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2], filter_row7);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8], &sum_f24.f8[0], filter_row8);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8], &sum_f24.f8[1], filter_row8);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8], &sum_f24.f8[2], filter_row8);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8], &sum_f24.f8[0], filter_row9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8], &sum_f24.f8[1], filter_row9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8], &sum_f24.f8[2], filter_row9);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// -------------------- Set 4 - PLN3->PKD3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void gaussian_filter_3x3_pln3_pkd3_tensor(T *srcPtr,
                                                     uint3 srcStridesNCH,
                                                     T *dstPtr,
                                                     uint2 dstStridesNH,
                                                     uint padLength,
                                                     uint2 tileSize,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     d_float9 *filterTensor)
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
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    d_float9 filter_f9 = filterTensor[id_z];
    float *filter_row1 = &filter_f9.f1[0];
    float *filter_row2 = &filter_f9.f1[3];
    float *filter_row3 = &filter_f9.f1[6];
    sum_f24.f4[0] = static_cast<float4>(0);
    sum_f24.f4[1] = static_cast<float4>(0);
    sum_f24.f4[2] = static_cast<float4>(0);
    sum_f24.f4[3] = static_cast<float4>(0);
    sum_f24.f4[4] = static_cast<float4>(0);
    sum_f24.f4[5] = static_cast<float4>(0);

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8])) = static_cast<uint2>(0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], filter_row1);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], filter_row1);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], filter_row1);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], filter_row2);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], filter_row2);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], filter_row2);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], filter_row3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], filter_row3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], filter_row3);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void gaussian_filter_5x5_pln3_pkd3_tensor(T *srcPtr,
                                                     uint3 srcStridesNCH,
                                                     T *dstPtr,
                                                     uint2 dstStridesNH,
                                                     uint padLength,
                                                     uint2 tileSize,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     d_float25 *filterTensor)
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
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    d_float25 filter_f25 = filterTensor[id_z];
    float *filter_row1 = &filter_f25.f1[0];
    float *filter_row2 = &filter_f25.f1[5];
    float *filter_row3 = &filter_f25.f1[10];
    float *filter_row4 = &filter_f25.f1[15];
    float *filter_row5 = &filter_f25.f1[20];
    sum_f24.f4[0] = static_cast<float4>(0);
    sum_f24.f4[1] = static_cast<float4>(0);
    sum_f24.f4[2] = static_cast<float4>(0);
    sum_f24.f4[3] = static_cast<float4>(0);
    sum_f24.f4[4] = static_cast<float4>(0);
    sum_f24.f4[5] = static_cast<float4>(0);

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8])) = static_cast<uint2>(0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], filter_row1);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], filter_row1);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], filter_row1);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], filter_row2);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], filter_row2);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], filter_row2);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], filter_row3);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], filter_row3);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], filter_row3);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], filter_row4);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], filter_row4);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], filter_row4);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], filter_row5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], filter_row5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], filter_row5);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void gaussian_filter_7x7_pln3_pkd3_tensor(T *srcPtr,
                                                     uint3 srcStridesNCH,
                                                     T *dstPtr,
                                                     uint2 dstStridesNH,
                                                     uint padLength,
                                                     uint2 tileSize,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     d_float49 *filterTensor)
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
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    d_float49 filter_f49 = filterTensor[id_z];
    float *filter_row1 = &filter_f49.f1[0];
    float *filter_row2 = &filter_f49.f1[7];
    float *filter_row3 = &filter_f49.f1[14];
    float *filter_row4 = &filter_f49.f1[21];
    float *filter_row5 = &filter_f49.f1[28];
    float *filter_row6 = &filter_f49.f1[35];
    float *filter_row7 = &filter_f49.f1[42];
    sum_f24.f4[0] = static_cast<float4>(0);
    sum_f24.f4[1] = static_cast<float4>(0);
    sum_f24.f4[2] = static_cast<float4>(0);
    sum_f24.f4[3] = static_cast<float4>(0);
    sum_f24.f4[4] = static_cast<float4>(0);
    sum_f24.f4[5] = static_cast<float4>(0);

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8])) = static_cast<uint2>(0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], filter_row1);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], filter_row1);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], filter_row1);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], filter_row2);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], filter_row2);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], filter_row2);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], filter_row3);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], filter_row3);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], filter_row3);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], filter_row4);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], filter_row4);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], filter_row4);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], filter_row5);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], filter_row5);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], filter_row5);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0], filter_row6);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1], filter_row6);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2], filter_row6);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0], filter_row7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1], filter_row7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2], filter_row7);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void gaussian_filter_9x9_pln3_pkd3_tensor(T *srcPtr,
                                                     uint3 srcStridesNCH,
                                                     T *dstPtr,
                                                     uint2 dstStridesNH,
                                                     uint padLength,
                                                     uint2 tileSize,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     d_float81 *filterTensor)
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
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    d_float81 filter_f81 = filterTensor[id_z];
    float *filter_row1 = &filter_f81.f1[0];
    float *filter_row2 = &filter_f81.f1[9];
    float *filter_row3 = &filter_f81.f1[18];
    float *filter_row4 = &filter_f81.f1[27];
    float *filter_row5 = &filter_f81.f1[36];
    float *filter_row6 = &filter_f81.f1[45];
    float *filter_row7 = &filter_f81.f1[54];
    float *filter_row8 = &filter_f81.f1[63];
    float *filter_row9 = &filter_f81.f1[72];
    sum_f24.f4[0] = static_cast<float4>(0);
    sum_f24.f4[1] = static_cast<float4>(0);
    sum_f24.f4[2] = static_cast<float4>(0);
    sum_f24.f4[3] = static_cast<float4>(0);
    sum_f24.f4[4] = static_cast<float4>(0);
    sum_f24.f4[5] = static_cast<float4>(0);

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8])) = static_cast<uint2>(0);
        *(reinterpret_cast<uint2 *>(&src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8])) = static_cast<uint2>(0);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], filter_row1);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], filter_row1);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], filter_row1);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], filter_row2);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], filter_row2);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], filter_row2);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], filter_row3);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], filter_row3);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], filter_row3);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], filter_row4);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], filter_row4);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], filter_row4);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], filter_row5);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], filter_row5);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], filter_row5);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0], filter_row6);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1], filter_row6);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2], filter_row6);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0], filter_row7);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1], filter_row7);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2], filter_row7);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8], &sum_f24.f8[0], filter_row8);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8], &sum_f24.f8[1], filter_row8);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8], &sum_f24.f8[2], filter_row8);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8], &sum_f24.f8[0], filter_row9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8], &sum_f24.f8[1], filter_row9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8], &sum_f24.f8[2], filter_row9);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

__device__ float gaussian(int iSquare, int j, float mulFactor)
{
    float expFactor = - (iSquare + (j * j)) * mulFactor;
    expFactor = expf(expFactor);
    return expFactor;
}

template <typename T, typename U>
__global__ void create_gaussian_kernel(T *filterTensor,
                                       U *rowType,
                                       float *stdDevTensor,
                                       int kernelSize,
                                       int filterStride,
                                       int batchSize)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    if(id_x >= batchSize)
        return;

    T filter_temp;
    T *filter = &filterTensor[id_x];
    float stdDev = stdDevTensor[id_x];
    float mulFactor = 1 / (2 * stdDev * stdDev);
    int kernelSizeByTwo = kernelSize / 2;
    int startIdx = - (kernelSizeByTwo);
    int lastRowIdx = (kernelSize - 1) * kernelSize;
    int rowIdx = 0;

    // compute values for only top left quarter and replicate the values
    for(int i = startIdx; i <= 0; i++, rowIdx += kernelSize)
    {
        int iSquare = i * i;
        int colIdx = 0;
        for(int j = startIdx; j <= 0; j++, colIdx++)
        {
            filter_temp.f1[rowIdx + colIdx] = gaussian(iSquare, j, mulFactor);
            filter_temp.f1[rowIdx + kernelSize - 1 - colIdx] = filter_temp.f1[rowIdx + colIdx];
        }

        // Copy symmetric rows
        if((lastRowIdx - rowIdx) != rowIdx)
            *(U *)(&filter_temp.f1[lastRowIdx - rowIdx]) = *(U *)(&filter_temp.f1[rowIdx]); // To check, if branching inside if for loop cause degration in performance
    }


    // Find the sum of kernel values
    int cnt = 0;
    float kernelSum = 0.0f;
    for(int i = startIdx; i <= kernelSizeByTwo; i++)
    {
        for(int j = startIdx; j <= kernelSizeByTwo; j++)
        {
            kernelSum += filter_temp.f1[cnt];
            cnt++;
        }
    }
    kernelSum = (1.0f / kernelSum);

    // Multiply kernel values by (1 / kernelSum)
    cnt = 0;
    for(int i = startIdx; i <= kernelSizeByTwo; i++)
    {
        for(int j = startIdx; j <= kernelSizeByTwo; j++)
        {
            filter_temp.f1[cnt] *= kernelSum;
            cnt++;
        }
    }

    *(reinterpret_cast<T *>(filter)) = filter_temp;
}

static RppStatus hip_exec_create_gaussian_kernel(Rpp32f *filterTensor,
                                                 Rpp32s kernelSize,
                                                 Rpp32f *stdDevTensor,
                                                 rpp::Handle &handle)
{
    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;
    int globalThreads_x = handle.GetBatchSize();
    int globalThreads_y = 1;
    int globalThreads_z = 1;
    int numValues = kernelSize * kernelSize;

    if (kernelSize == 3)
    {
        float3 *rowType = NULL;
        hipLaunchKernelGGL(create_gaussian_kernel,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           reinterpret_cast<d_float9 *>(filterTensor),
                           rowType,
                           stdDevTensor,
                           kernelSize,
                           numValues,
                           handle.GetBatchSize());
    }
    else if (kernelSize == 5)
    {
        d_float5 *rowType = NULL;
        hipLaunchKernelGGL(create_gaussian_kernel,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           reinterpret_cast<d_float25 *>(filterTensor),
                           rowType,
                           stdDevTensor,
                           kernelSize,
                           numValues,
                           handle.GetBatchSize());
    }
    else if (kernelSize == 7)
    {
        d_float7 *rowType = NULL;
        hipLaunchKernelGGL(create_gaussian_kernel,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           reinterpret_cast<d_float49 *>(filterTensor),
                           rowType,
                           stdDevTensor,
                           kernelSize,
                           numValues,
                           handle.GetBatchSize());
    }
    else if (kernelSize == 9)
    {
        d_float9 *rowType = NULL;
        hipLaunchKernelGGL(create_gaussian_kernel,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           reinterpret_cast<d_float81 *>(filterTensor),
                           rowType,
                           stdDevTensor,
                           kernelSize,
                           numValues,
                           handle.GetBatchSize());
    }

    return RPP_SUCCESS;
}

// -------------------- Set 5 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_gaussian_filter_tensor(T *srcPtr,
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

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (128 - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;

    // Create a filter of size (kernel size x kernel size)
    float *filterTensor = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
    hip_exec_create_gaussian_kernel(filterTensor,
                                    kernelSize,
                                    handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                    handle);


    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;

        if (kernelSize == 3)
        {
            hipLaunchKernelGGL(gaussian_filter_3x3_pkd_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc,
                               reinterpret_cast<d_float9 *>(filterTensor));
        }
        else if (kernelSize == 5)
        {
            hipLaunchKernelGGL(gaussian_filter_5x5_pkd_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc,
                               reinterpret_cast<d_float25 *>(filterTensor));
        }
        else if (kernelSize == 7)
        {
            hipLaunchKernelGGL(gaussian_filter_7x7_pkd_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc,
                               reinterpret_cast<d_float49 *>(filterTensor));
        }
        else if (kernelSize == 9)
        {
            hipLaunchKernelGGL(gaussian_filter_9x9_pkd_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc,
                               reinterpret_cast<d_float81 *>(filterTensor));
        }
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        if (kernelSize == 3)
        {
            hipLaunchKernelGGL(gaussian_filter_3x3_pln_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc,
                               reinterpret_cast<d_float9 *>(filterTensor));
        }
        else if (kernelSize == 5)
        {
            hipLaunchKernelGGL(gaussian_filter_5x5_pln_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc,
                               reinterpret_cast<d_float25 *>(filterTensor));
        }
        else if (kernelSize == 7)
        {
            hipLaunchKernelGGL(gaussian_filter_7x7_pln_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc,
                               reinterpret_cast<d_float49 *>(filterTensor));
        }
        else if (kernelSize == 9)
        {
            hipLaunchKernelGGL(gaussian_filter_9x9_pln_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc,
                               reinterpret_cast<d_float81 *>(filterTensor));
        }
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if (kernelSize == 3)
            {
                hipLaunchKernelGGL(gaussian_filter_3x3_pkd3_pln3_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc,
                                   reinterpret_cast<d_float9 *>(filterTensor));
            }
            else if (kernelSize == 5)
            {
                hipLaunchKernelGGL(gaussian_filter_5x5_pkd3_pln3_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc,
                                   reinterpret_cast<d_float25 *>(filterTensor));
            }
            else if (kernelSize == 7)
            {
                hipLaunchKernelGGL(gaussian_filter_7x7_pkd3_pln3_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc,
                                   reinterpret_cast<d_float49 *>(filterTensor));
            }
            else if (kernelSize == 9)
            {
                hipLaunchKernelGGL(gaussian_filter_9x9_pkd3_pln3_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc,
                                   reinterpret_cast<d_float81 *>(filterTensor));
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;

            if (kernelSize == 3)
            {
                hipLaunchKernelGGL(gaussian_filter_3x3_pln3_pkd3_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc,
                                   reinterpret_cast<d_float9 *>(filterTensor));
            }
            else if (kernelSize == 5)
            {
                hipLaunchKernelGGL(gaussian_filter_5x5_pln3_pkd3_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc,
                                   reinterpret_cast<d_float25 *>(filterTensor));
            }
            else if (kernelSize == 7)
            {
                hipLaunchKernelGGL(gaussian_filter_7x7_pln3_pkd3_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc,
                                   reinterpret_cast<d_float49 *>(filterTensor));
            }
            else if (kernelSize == 9)
            {
                hipLaunchKernelGGL(gaussian_filter_9x9_pln3_pkd3_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc,
                                   reinterpret_cast<d_float81 *>(filterTensor));
            }
        }
    }

    return RPP_SUCCESS;
}
