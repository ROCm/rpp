#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - gaussian_filter device helpers --------------------

__device__ void gaussian_filter_3x3_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8, d_float9 *filter_f9, int rowIndex)
{
    float src_f1;
    uint3 src_ui3;
    src_ui3 = *(uint3 *)srcPtr;
    src_f1 = rpp_hip_unpack0(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f9->f1[rowIndex], dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f9->f1[rowIndex + 1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f9->f1[rowIndex], dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f9->f1[rowIndex + 2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f9->f1[rowIndex + 1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f9->f1[rowIndex], dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui3.x);
    dst_f8->f1[1] = fmaf(src_f1, filter_f9->f1[rowIndex + 2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f9->f1[rowIndex + 1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f9->f1[rowIndex], dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui3.y);
    dst_f8->f1[2] = fmaf(src_f1, filter_f9->f1[rowIndex + 2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f9->f1[rowIndex + 1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f9->f1[rowIndex], dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui3.y);
    dst_f8->f1[3] = fmaf(src_f1, filter_f9->f1[rowIndex + 2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f9->f1[rowIndex + 1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f9->f1[rowIndex], dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui3.y);
    dst_f8->f1[4] = fmaf(src_f1, filter_f9->f1[rowIndex + 2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f9->f1[rowIndex + 1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f9->f1[rowIndex], dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui3.y);
    dst_f8->f1[5] = fmaf(src_f1, filter_f9->f1[rowIndex + 2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f9->f1[rowIndex + 1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f9->f1[rowIndex], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui3.z);
    dst_f8->f1[6] = fmaf(src_f1, filter_f9->f1[rowIndex + 2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f9->f1[rowIndex + 1], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui3.z);
    dst_f8->f1[7] = fmaf(src_f1, filter_f9->f1[rowIndex + 2], dst_f8->f1[7]);
}

__device__ void gaussian_filter_5x5_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8, d_float25 *filter_f25, int rowIndex)
{
    float src_f1;
    uint3 src_ui3;
    src_ui3 = *(uint3 *)srcPtr;
    src_f1 = rpp_hip_unpack0(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f25->f1[rowIndex], dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f25->f1[rowIndex + 1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f25->f1[rowIndex], dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f25->f1[rowIndex + 2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f25->f1[rowIndex + 1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f25->f1[rowIndex], dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f25->f1[rowIndex + 3], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f25->f1[rowIndex + 2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f25->f1[rowIndex + 1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f25->f1[rowIndex], dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui3.y);
    dst_f8->f1[0] = fmaf(src_f1, filter_f25->f1[rowIndex + 4], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f25->f1[rowIndex + 3], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f25->f1[rowIndex + 2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f25->f1[rowIndex + 1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f25->f1[rowIndex], dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui3.y);
    dst_f8->f1[1] = fmaf(src_f1, filter_f25->f1[rowIndex + 4], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f25->f1[rowIndex + 3], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f25->f1[rowIndex + 2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f25->f1[rowIndex + 1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f25->f1[rowIndex], dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui3.y);
    dst_f8->f1[2] = fmaf(src_f1, filter_f25->f1[rowIndex + 4], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f25->f1[rowIndex + 3], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f25->f1[rowIndex + 2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f25->f1[rowIndex + 1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f25->f1[rowIndex], dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui3.y);
    dst_f8->f1[3] = fmaf(src_f1, filter_f25->f1[rowIndex + 4], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f25->f1[rowIndex + 3], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f25->f1[rowIndex + 2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f25->f1[rowIndex + 1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f25->f1[rowIndex], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui3.z);
    dst_f8->f1[4] = fmaf(src_f1, filter_f25->f1[rowIndex + 4], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f25->f1[rowIndex + 3], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f25->f1[rowIndex + 2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f25->f1[rowIndex + 1], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui3.z);
    dst_f8->f1[5] = fmaf(src_f1, filter_f25->f1[rowIndex + 4], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f25->f1[rowIndex + 3], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f25->f1[rowIndex + 2], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui3.z);
    dst_f8->f1[6] = fmaf(src_f1, filter_f25->f1[rowIndex + 4], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f25->f1[rowIndex + 3], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui3.z);
    dst_f8->f1[7] = fmaf(src_f1, filter_f25->f1[rowIndex + 4], dst_f8->f1[7]);
}

__device__ void gaussian_filter_7x7_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8, d_float49 *filter_f49, int rowIndex)
{
    float src_f1;
    uint4 src_ui4 = *(uint4 *)srcPtr;
    src_f1 = rpp_hip_unpack0(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f49->f1[rowIndex], dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f49->f1[rowIndex + 1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f49->f1[rowIndex], dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f49->f1[rowIndex + 2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f49->f1[rowIndex + 1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f49->f1[rowIndex], dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f49->f1[rowIndex + 3], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f49->f1[rowIndex + 2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f49->f1[rowIndex + 1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f49->f1[rowIndex], dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter_f49->f1[rowIndex + 4], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f49->f1[rowIndex + 3], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f49->f1[rowIndex + 2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f49->f1[rowIndex + 1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f49->f1[rowIndex], dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter_f49->f1[rowIndex + 5], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f49->f1[rowIndex + 4], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f49->f1[rowIndex + 3], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f49->f1[rowIndex + 2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f49->f1[rowIndex + 1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f49->f1[rowIndex], dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter_f49->f1[rowIndex + 6], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f49->f1[rowIndex + 5], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f49->f1[rowIndex + 4], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f49->f1[rowIndex + 3], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f49->f1[rowIndex + 2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f49->f1[rowIndex + 1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f49->f1[rowIndex], dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui4.y);
    dst_f8->f1[1] = fmaf(src_f1, filter_f49->f1[rowIndex + 6], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f49->f1[rowIndex + 5], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f49->f1[rowIndex + 4], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f49->f1[rowIndex + 3], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f49->f1[rowIndex + 2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f49->f1[rowIndex + 1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f49->f1[rowIndex], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui4.z);
    dst_f8->f1[2] = fmaf(src_f1, filter_f49->f1[rowIndex + 6], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f49->f1[rowIndex + 5], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f49->f1[rowIndex + 4], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f49->f1[rowIndex + 3], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f49->f1[rowIndex + 2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f49->f1[rowIndex + 1], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui4.z);
    dst_f8->f1[3] = fmaf(src_f1, filter_f49->f1[rowIndex + 6], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f49->f1[rowIndex + 5], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f49->f1[rowIndex + 4], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f49->f1[rowIndex + 3], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f49->f1[rowIndex + 2], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui4.z);
    dst_f8->f1[4] = fmaf(src_f1, filter_f49->f1[rowIndex + 6], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f49->f1[rowIndex + 5], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f49->f1[rowIndex + 4], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f49->f1[rowIndex + 3], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui4.z);
    dst_f8->f1[5] = fmaf(src_f1, filter_f49->f1[rowIndex + 6], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f49->f1[rowIndex + 5], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f49->f1[rowIndex + 4], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui4.w);
    dst_f8->f1[6] = fmaf(src_f1, filter_f49->f1[rowIndex + 6], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f49->f1[rowIndex + 5], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui4.w);
    dst_f8->f1[7] = fmaf(src_f1, filter_f49->f1[rowIndex + 6], dst_f8->f1[7]);
}

__device__ void gaussian_filter_9x9_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8, d_float81 *filter_f81, int rowIndex)
{
    float src_f1;
    uint4 src_ui4 = *(uint4 *)srcPtr;
    src_f1 = rpp_hip_unpack0(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f81->f1[rowIndex], dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f81->f1[rowIndex + 1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f81->f1[rowIndex], dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f81->f1[rowIndex + 2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f81->f1[rowIndex + 1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f81->f1[rowIndex], dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f1, filter_f81->f1[rowIndex + 3], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f81->f1[rowIndex + 2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f81->f1[rowIndex + 1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f81->f1[rowIndex], dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter_f81->f1[rowIndex + 4], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f81->f1[rowIndex + 3], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f81->f1[rowIndex + 2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f81->f1[rowIndex + 1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f81->f1[rowIndex], dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter_f81->f1[rowIndex + 5], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f81->f1[rowIndex + 4], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f81->f1[rowIndex + 3], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f81->f1[rowIndex + 2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f81->f1[rowIndex + 1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f81->f1[rowIndex], dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter_f81->f1[rowIndex + 6], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f81->f1[rowIndex + 5], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f81->f1[rowIndex + 4], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f81->f1[rowIndex + 3], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f81->f1[rowIndex + 2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f81->f1[rowIndex + 1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f81->f1[rowIndex], dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f1, filter_f81->f1[rowIndex + 7], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f81->f1[rowIndex + 6], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f81->f1[rowIndex + 5], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f81->f1[rowIndex + 4], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f81->f1[rowIndex + 3], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f81->f1[rowIndex + 2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f81->f1[rowIndex + 1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f81->f1[rowIndex], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui4.z);
    dst_f8->f1[0] = fmaf(src_f1, filter_f81->f1[rowIndex + 8], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter_f81->f1[rowIndex + 7], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f81->f1[rowIndex + 6], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f81->f1[rowIndex + 5], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f81->f1[rowIndex + 4], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f81->f1[rowIndex + 3], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f81->f1[rowIndex + 2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f81->f1[rowIndex + 1], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui4.z);
    dst_f8->f1[1] = fmaf(src_f1, filter_f81->f1[rowIndex + 8], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter_f81->f1[rowIndex + 7], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f81->f1[rowIndex + 6], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f81->f1[rowIndex + 5], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f81->f1[rowIndex + 4], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f81->f1[rowIndex + 3], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f81->f1[rowIndex + 2], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui4.z);
    dst_f8->f1[2] = fmaf(src_f1, filter_f81->f1[rowIndex + 8], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter_f81->f1[rowIndex + 7], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f81->f1[rowIndex + 6], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f81->f1[rowIndex + 5], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f81->f1[rowIndex + 4], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f81->f1[rowIndex + 3], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui4.z);
    dst_f8->f1[3] = fmaf(src_f1, filter_f81->f1[rowIndex + 8], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter_f81->f1[rowIndex + 7], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f81->f1[rowIndex + 6], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f81->f1[rowIndex + 5], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f81->f1[rowIndex + 4], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui4.w);
    dst_f8->f1[4] = fmaf(src_f1, filter_f81->f1[rowIndex + 8], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter_f81->f1[rowIndex + 7], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f81->f1[rowIndex + 6], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f81->f1[rowIndex + 5], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui4.w);
    dst_f8->f1[5] = fmaf(src_f1, filter_f81->f1[rowIndex + 8], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter_f81->f1[rowIndex + 7], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f81->f1[rowIndex + 6], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui4.w);
    dst_f8->f1[6] = fmaf(src_f1, filter_f81->f1[rowIndex + 8], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter_f81->f1[rowIndex + 7], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui4.w);
    dst_f8->f1[7] = fmaf(src_f1, filter_f81->f1[rowIndex + 8], dst_f8->f1[7]);
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
        *(uint2 *)src_lds_channel[0] = (uint2)0;
        *(uint2 *)src_lds_channel[1] = (uint2)0;
        *(uint2 *)src_lds_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f9, 0);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f9, 0);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f9, 0);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f9, 3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f9, 3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f9, 3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f9, 6);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f9, 6);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f9, 6);
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
        *(uint2 *)src_lds_channel[0] = (uint2)0;
        *(uint2 *)src_lds_channel[1] = (uint2)0;
        *(uint2 *)src_lds_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 0);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 0);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 0);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 10);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 10);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 10);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 15);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 15);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 15);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 20);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 20);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 20);
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
        *(uint2 *)src_lds_channel[0] = (uint2)0;
        *(uint2 *)src_lds_channel[1] = (uint2)0;
        *(uint2 *)src_lds_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 0);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 0);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 0);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 14);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 14);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 14);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 21);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 21);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 21);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 28);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 28);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 28);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 35);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 35);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 35);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 42);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 42);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 42);
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
        *(uint2 *)src_lds_channel[0] = (uint2)0;
        *(uint2 *)src_lds_channel[1] = (uint2)0;
        *(uint2 *)src_lds_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 0);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 0);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 0);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 18);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 18);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 18);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 27);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 27);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 27);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 36);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 36);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 36);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 45);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 45);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 45);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 54);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 54);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 54);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 63);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 63);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 63);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 72);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 72);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 72);
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
    sum_f8.f4[0] = (float4) 0;
    sum_f8.f4[1] = (float4) 0;
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, &filter_f9, 0);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, &filter_f9, 3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, &filter_f9, 6);
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
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, &filter_f9, 0);
            gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, &filter_f9, 3);
            gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, &filter_f9, 6);
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
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, &filter_f9, 0);
            gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, &filter_f9, 3);
            gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, &filter_f9, 6);
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
    sum_f8.f4[0] = (float4) 0;
    sum_f8.f4[1] = (float4) 0;
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, &filter_f25, 0);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, &filter_f25, 5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, &filter_f25, 10);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, &filter_f25, 15);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, &filter_f25, 20);
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
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, &filter_f25, 0);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, &filter_f25, 5);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, &filter_f25, 10);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, &filter_f25, 15);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, &filter_f25, 20);
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
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, &filter_f25, 0);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, &filter_f25, 5);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, &filter_f25, 10);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, &filter_f25, 15);
            gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, &filter_f25, 20);
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
    sum_f8.f4[0] = (float4) 0;
    sum_f8.f4[1] = (float4) 0;
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, &filter_f49, 0);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, &filter_f49, 7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, &filter_f49, 14);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, &filter_f49, 21);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, &filter_f49, 28);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, &filter_f49, 35);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, &filter_f49, 42);
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
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, &filter_f49, 0);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, &filter_f49, 7);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, &filter_f49, 14);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, &filter_f49, 21);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, &filter_f49, 28);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, &filter_f49, 35);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, &filter_f49, 42);
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
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, &filter_f49, 0);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, &filter_f49, 7);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, &filter_f49, 14);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, &filter_f49, 21);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, &filter_f49, 28);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, &filter_f49, 35);
            gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, &filter_f49, 42);
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
    sum_f8.f4[0] = (float4) 0;
    sum_f8.f4[1] = (float4) 0;
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, &filter_f81, 0);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, &filter_f81, 9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, &filter_f81, 18);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, &filter_f81, 27);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, &filter_f81, 36);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, &filter_f81, 45);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, &filter_f81, 54);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8, &filter_f81, 63);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8, &filter_f81, 72);
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
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, &filter_f81, 0);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, &filter_f81, 9);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, &filter_f81, 18);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, &filter_f81, 27);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, &filter_f81, 36);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, &filter_f81, 45);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, &filter_f81, 54);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8, &filter_f81, 63);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8, &filter_f81, 72);
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
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, &filter_f81, 0);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, &filter_f81, 9);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, &filter_f81, 18);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, &filter_f81, 27);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, &filter_f81, 36);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, &filter_f81, 45);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, &filter_f81, 54);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8, &filter_f81, 63);
            gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8, &filter_f81, 72);
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
        *(uint2 *)src_lds_channel[0] = (uint2)0;
        *(uint2 *)src_lds_channel[1] = (uint2)0;
        *(uint2 *)src_lds_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f9, 0);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f9, 0);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f9, 0);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f9, 3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f9, 3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f9, 3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f9, 6);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f9, 6);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f9, 6);
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
        *(uint2 *)src_lds_channel[0] = (uint2)0;
        *(uint2 *)src_lds_channel[1] = (uint2)0;
        *(uint2 *)src_lds_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 0);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 0);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 0);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 10);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 10);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 10);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 15);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 15);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 15);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 20);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 20);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 20);
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
        *(uint2 *)src_lds_channel[0] = (uint2)0;
        *(uint2 *)src_lds_channel[1] = (uint2)0;
        *(uint2 *)src_lds_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 0);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 0);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 0);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 14);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 14);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 14);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 21);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 21);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 21);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 28);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 28);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 28);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 35);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 35);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 35);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 42);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 42);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 42);
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
        *(uint2 *)src_lds_channel[0] = (uint2)0;
        *(uint2 *)src_lds_channel[1] = (uint2)0;
        *(uint2 *)src_lds_channel[2] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 0);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 0);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 0);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 18);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 18);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 18);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 27);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 27);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 27);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 36);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 36);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 36);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 45);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 45);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 45);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 54);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 54);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 54);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 63);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 63);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 63);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 72);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 72);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 72);
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
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f9, 0);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f9, 0);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f9, 0);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f9, 3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f9, 3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f9, 3);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f9, 6);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f9, 6);
        gaussian_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f9, 6);
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
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 0);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 0);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 0);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 5);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 10);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 10);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 10);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 15);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 15);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 15);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f25, 20);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f25, 20);
        gaussian_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f25, 20);
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
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 0);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 0);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 0);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 7);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 14);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 14);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 14);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 21);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 21);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 21);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 28);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 28);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 28);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 35);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 35);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 35);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f49, 42);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f49, 42);
        gaussian_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f49, 42);
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
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = (uint2)0;
        *(uint2 *)&src_lds[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = (uint2)0;
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 0);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 0);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 0);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 9);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 18);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 18);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 18);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 27);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 27);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 27);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 36);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 36);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 36);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 45);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 45);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 45);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 54);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 54);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 54);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 63);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 63);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 63);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8], &sum_f24.f8[0], &filter_f81, 72);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8], &sum_f24.f8[1], &filter_f81, 72);
        gaussian_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8], &sum_f24.f8[2], &filter_f81, 72);
        rpp_hip_adjust_range(dstPtr, &sum_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

__device__ float gaussian(int x, int y, float mulFactor)
{
    float expFactor = - ((x * x) + (y * y)) * mulFactor;
    expFactor = expf(expFactor);
    float res  = (expFactor * mulFactor) / PI;
    return res;
}

__global__ void set_filter_values(float *filterTensor, float *stdDevTensor, int kernelSize, int N, int batchSize)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    if(id_x >= batchSize)
        return;

    float *filter = &filterTensor[id_x * N];
    float stdDev = stdDevTensor[id_x];
    int cnt = 0;
    float mulFactor = 1 / (2 * stdDev * stdDev);
    float kernelSum = 0.0f;
    for(int i = 0; i < kernelSize; i++)
    {
        for(int j = 0; j < kernelSize; j++)
        {
            filter[cnt] = gaussian(i, j, mulFactor);
            kernelSum += filter[cnt];
            cnt++;
        }
    }
    kernelSum = (kernelSum == 0.0f) ? 1.0f : (1.0f / kernelSum);

    // Multiply kernel values by (1 / kernelSum)
    cnt = 0;
    for(int i = 0; i < kernelSize; i++)
    {
        for(int j = 0; j < kernelSize; j++)
        {
            filter[cnt] *= kernelSum;
            cnt++;
        }
    }
}

static RppStatus hip_exec_fill_kernel_values(float *filterTensor, int kernelSize, Rpp32f *stdDevTensor, rpp::Handle &handle)
{
    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;
    int globalThreads_x = handle.GetBatchSize();
    int globalThreads_y = 1;
    int globalThreads_z = 1;
    int numValues = kernelSize * kernelSize;

    hipLaunchKernelGGL(set_filter_values,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       filterTensor,
                       stdDevTensor,
                       kernelSize,
                       numValues,
                       handle.GetBatchSize());

    hipDeviceSynchronize();
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
    void *filterTensor;
    int numValues = kernelSize * kernelSize;
    hipMalloc(&filterTensor,  numValues * dstDescPtr->n * sizeof(float));
    hip_exec_fill_kernel_values((float *)filterTensor,
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
                               (d_float9 *)filterTensor);

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
                               (d_float25 *)filterTensor);

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
                               (d_float49 *)filterTensor);

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
                               (d_float81 *)filterTensor);

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
                               (d_float9 *)filterTensor);
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
                               (d_float25 *)filterTensor);
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
                               (d_float49 *)filterTensor);
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
                               (d_float81 *)filterTensor);
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
                                   (d_float9 *)filterTensor);

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
                                   (d_float25 *)filterTensor);
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
                                   (d_float49 *)filterTensor);
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
                                   (d_float81 *)filterTensor);
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
                                   (d_float9 *)filterTensor);
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
                                   (d_float25 *)filterTensor);
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
                                   (d_float49 *)filterTensor);
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
                                   (d_float81 *)filterTensor);
            }
        }
    }

    hipFree(filterTensor);

    return RPP_SUCCESS;
}
