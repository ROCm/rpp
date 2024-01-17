#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ float4 rpp_hip_load4(float *table, uint4 &tableValLoc)
{
    return make_float4(table[tableValLoc.x], table[tableValLoc.y], table[tableValLoc.z], table[tableValLoc.w]);
}

__device__ void remap_srclocs_hip_compute(int4 *srcRoiPtr_i4, float *rowRemapTable, float *colRemapTable, int id_x, int id_y, int id_z, d_float16 *locSrc_f16)
{
    d_uint8 increment_ui8, locSrc_ui8;

    increment_ui8.ui4[0] = make_uint4(0, 1, 2, 3);
    increment_ui8.ui4[1] = make_uint4(4, 5, 6, 7);
    uint4 locSrc_ui4 = (uint4)(id_x);
    locSrc_ui8.ui4[0] = locSrc_ui4 + increment_ui8.ui4[0];
    locSrc_ui8.ui4[1] = locSrc_ui4 + increment_ui8.ui4[1];

    locSrc_f16->f8[0].f4[0] = rpp_hip_load4(colRemapTable, locSrc_ui8.ui4[0]);
    locSrc_f16->f8[0].f4[1] = rpp_hip_load4(colRemapTable, locSrc_ui8.ui4[1]);
    locSrc_f16->f8[1].f4[0] = rpp_hip_load4(rowRemapTable, locSrc_ui8.ui4[0]);
    locSrc_f16->f8[1].f4[1] = rpp_hip_load4(rowRemapTable, locSrc_ui8.ui4[1]);
}

// -------------------- Set 2 - Nearest Neighbor Interpolation --------------------

template <typename T>
__global__ void remap_nearest_neighbor_pkd_tensor(T *srcPtr,
                                                  uint2 srcStridesNH,
                                                  T *dstPtr,
                                                  uint2 dstStridesNH,
                                                  uint2 dstDimsWH,
                                                  float *rowRemapTable,
                                                  float *colRemapTable,
                                                  uint2 remapTableStridesNH,
                                                  RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void remap_nearest_neighbor_pln_tensor(T *srcPtr,
                                                  uint3 srcStridesNCH,
                                                  T *dstPtr,
                                                  uint3 dstStridesNCH,
                                                  uint2 dstDimsWH,
                                                  int channelsDst,
                                                  float *rowRemapTable,
                                                  float *colRemapTable,
                                                  uint2 remapTableStridesNH,
                                                  RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void remap_nearest_neighbor_pkd3_pln3_tensor(T *srcPtr,
                                                        uint2 srcStridesNH,
                                                        T *dstPtr,
                                                        uint3 dstStridesNCH,
                                                        uint2 dstDimsWH,
                                                        int channelsDst,
                                                        float *rowRemapTable,
                                                        float *colRemapTable,
                                                        uint2 remapTableStridesNH,
                                                        RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void remap_nearest_neighbor_pln3_pkd3_tensor(T *srcPtr,
                                                        uint3 srcStridesNCH,
                                                        T *dstPtr,
                                                        uint2 dstStridesNH,
                                                        uint2 dstDimsWH,
                                                        float *rowRemapTable,
                                                        float *colRemapTable,
                                                        uint2 remapTableStridesNH,
                                                        RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

// -------------------- Set 2 - Bilinear Interpolation --------------------

template <typename T>
__global__ void remap_bilinear_pkd_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
                                          uint2 dstDimsWH,
                                          float *rowRemapTable,
                                          float *colRemapTable,
                                          uint2 remapTableStridesNH,
                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void remap_bilinear_pln_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          uint2 dstDimsWH,
                                          int channelsDst,
                                          float *rowRemapTable,
                                          float *colRemapTable,
                                          uint2 remapTableStridesNH,
                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void remap_bilinear_pkd3_pln3_tensor(T *srcPtr,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
                                                uint2 dstDimsWH,
                                                int channelsDst,
                                                float *rowRemapTable,
                                                float *colRemapTable,
                                                uint2 remapTableStridesNH,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void remap_bilinear_pln3_pkd3_tensor(T *srcPtr,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
                                                uint2 dstDimsWH,
                                                float *rowRemapTable,
                                                float *colRemapTable,
                                                uint2 remapTableStridesNH,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

// -------------------- Set 3 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_remap_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *rowRemapTable,
                                Rpp32f *colRemapTable,
                                RpptDescPtr remapTableDescPtr,
                                RpptInterpolationType interpolationType,
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

    if (interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(remap_nearest_neighbor_pkd_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(remap_nearest_neighbor_pln_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               dstDescPtr->c,
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(remap_nearest_neighbor_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               dstDescPtr->c,
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(remap_nearest_neighbor_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
    }
    else if (interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(remap_bilinear_pkd_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(remap_bilinear_pln_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               dstDescPtr->c,
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(remap_bilinear_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               dstDescPtr->c,
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(remap_bilinear_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
    }
    return RPP_SUCCESS;
}