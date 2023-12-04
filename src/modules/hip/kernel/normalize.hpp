#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ __forceinline__ uint rpp_hip_mod(uint a, uint b)
{
    return (a % b);
}

__device__ uint compute_2d_paramindex(uint y, uint x, uint *paramShape, uint *paramStrides)
{
    uint yFactor =  ((paramShape[0] > 1)) ? (rpp_hip_mod(y, paramShape[0])) * paramStrides[0] : 0;
    uint xFactor =  ((paramShape[1] > 1)) ? (rpp_hip_mod(x, paramShape[1])) * paramStrides[1] : 0;
    uint paramIndex = yFactor + xFactor;
    return paramIndex;
}

__global__ void normalize_2d_hip_tensor(float *srcPtr,
                                        uint2 srcStridesNH,
                                        float *dstPtr,
                                        uint2 dstStridesNH,
                                        float *meanTensor,
                                        float *stdDevTensor,
                                        float scale,
                                        float shift,
                                        uint *roiTensor,
                                        uint *paramShapeTensor,
                                        uint *paramStridesTensor,
                                        uint maxParamVolume)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // width
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // height
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    uint *roi = &roiTensor[id_z * 4 + 2];
    uint height = roi[0];
    uint width = roi[1];

    if (id_x >= width || id_y >= height)
        return;

    uint *paramShape = &paramShapeTensor[id_z * 2];
    uint *paramStrides = &paramStridesTensor[id_z * 2];
    uint paramIndex = compute_2d_paramindex(id_y, id_x, paramShape, paramStrides);

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
    float mean = meanTensor[id_z * maxParamVolume + paramIndex];
    float stdDev = stdDevTensor[id_z * maxParamVolume + paramIndex];
    float stdDevSquare = stdDev * stdDev;
    float invStdDev = stdDevSquare ? rsqrtf(stdDevSquare) * scale : 0;
    dstPtr[dstIdx] = fmaf((srcPtr[srcIdx] - mean), invStdDev, shift);
}

// __device__ void normalize_hip_compute(d_float8 *data_f8, d_float8 *mean_f8, d_float8 *invStdDev_f8, d_float8 *shift_f8)
// {
//     data_f8->f4[0] = ((data_f8->f4[0] - mean_f8->f4[0]) * invStdDev_f8->f4[1]) + shift_f8->f4[0];
//     data_f8->f4[1] = ((data_f8->f4[1] - mean_f8->f4[1]) * invStdDev_f8->f4[1]) + shift_f8->f4[1];
// }

// __device__ void load_normalize_params(d_uint8 *locParam_ui8, float *meanPtr, float *stdDevPtr, float scale, d_float8 *mean_f8,  d_float8 *invStdDev_f8)
// {
//     for(int i = 0; i < 8; i++)
//     {
//         mean_f8->f1[i] = meanPtr[locParam_ui8->ui1[i]];
//         float stdDev = stdDevPtr[locParam_ui8->ui1[i]];
//         float stdDevSquare = stdDev * stdDev;
//         float invStdDev = stdDevSquare ? rsqrtf(stdDevSquare) * scale : 0;
//         invStdDev_f8->f1[i] = invStdDev;
//     }
// }

// __device__ void normalize_2d_paramlocs_hip_compute(uint id_y, uint id_x, d_uint8 *locParam_ui8, uint *paramShape, uint *paramStrides)
// {
//     d_uint8 increment_ui8, locDstx_ui8;
//     increment_ui8.ui4[0] = make_uint4(0, 1, 2, 3);
//     increment_ui8.ui4[1] = make_uint4(4, 5, 6, 7);
//     locDstx_ui8.ui4[0] = static_cast<uint4>(id_x) + increment_ui8.ui4[0];
//     locDstx_ui8.ui4[1] = static_cast<uint4>(id_x) + increment_ui8.ui4[1];

//     locParam_ui8->ui1[0] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[0], paramShape, paramStrides);
//     locParam_ui8->ui1[1] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[1], paramShape, paramStrides);
//     locParam_ui8->ui1[2] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[2], paramShape, paramStrides);
//     locParam_ui8->ui1[3] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[3], paramShape, paramStrides);
//     locParam_ui8->ui1[4] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[4], paramShape, paramStrides);
//     locParam_ui8->ui1[5] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[5], paramShape, paramStrides);
//     locParam_ui8->ui1[6] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[6], paramShape, paramStrides);
//     locParam_ui8->ui1[7] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[7], paramShape, paramStrides);
// }

// __global__ void normalize_2d_hip_tensor(float *srcPtr,
//                                         uint2 srcStridesNH,
//                                         float *dstPtr,
//                                         uint2 dstStridesNH,
//                                         float *meanTensor,
//                                         float *stdDevTensor,
//                                         float scale,
//                                         float shift,
//                                         uint *roiTensor,
//                                         uint *paramShapeTensor,
//                                         uint *paramStridesTensor,
//                                         uint maxParamVolume)
// {
//     uint id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8; // width
//     uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // height
//     uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

//     uint *roi = &roiTensor[id_z * 4 + 2];
//     uint height = roi[0];
//     uint width = roi[1];

//     if (id_x >= width || id_y >= height)
//         return;

//     uint *paramShape = &paramShapeTensor[id_z * 2];
//     uint *paramStrides = &paramStridesTensor[id_z * 2];

//     d_uint8 locParam_ui8;
//     normalize_2d_paramlocs_hip_compute(id_y, id_x, &locParam_ui8, paramShape, paramStrides);

//     d_float8 mean_f8, invStdDev_f8, shift_f8;
//     float *meanPtr = &meanTensor[id_z * maxParamVolume];
//     float *stdDevPtr = &stdDevTensor[id_z * maxParamVolume];
//     load_normalize_params(&locParam_ui8, meanPtr, stdDevPtr, scale, &mean_f8, &invStdDev_f8);
//     shift_f8.f4[0] = static_cast<float4>(shift);
//     shift_f8.f4[1] = shift_f8.f4[0];

//     d_float8 data_f8;
//     uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + id_x;
//     uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
//     rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &data_f8);
//     normalize_hip_compute(&data_f8, &mean_f8, &invStdDev_f8, &shift_f8);
//     rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &data_f8);
// }

__device__ uint compute_3d_paramindex(uint z, uint y, uint x, uint *paramShape, uint *paramStrides)
{
    uint zFactor =  ((paramShape[0] > 1)) ? rpp_hip_mod(z, paramShape[0]) * paramStrides[0] : 0;
    uint yFactor =  ((paramShape[1] > 1)) ? rpp_hip_mod(y, paramShape[1]) * paramStrides[1] : 0;
    uint xFactor =  ((paramShape[2] > 1)) ? rpp_hip_mod(x, paramShape[2]) * paramStrides[2] : 0;
    uint paramIndex = zFactor + yFactor + xFactor;
    return paramIndex;
}

__global__ void normalize_3d_hip_tensor(float *srcPtr,
                                        uint2 srcStridesDH,
                                        float *dstPtr,
                                        uint2 dstStridesDH,
                                        float *meanTensor,
                                        float *stdDevTensor,
                                        float scale,
                                        float shift,
                                        uint *roiTensor,
                                        uint *paramShapeTensor,
                                        uint *paramStridesTensor,
                                        uint maxParamVolume)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // width
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // height
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // depth

    uint *roi = roiTensor;
    uint width = roi[2];
    uint height = roi[1];
    uint depth = roi[0];

    if (id_x >= width || id_y >= height || id_z >= depth)
        return;

    uint *paramShape = paramShapeTensor;
    uint *paramStrides = paramStridesTensor;
    uint paramIndex = compute_3d_paramindex(id_z, id_y, id_x, paramShape, paramStrides);

    uint srcIdx = (id_z * srcStridesDH.x) + (id_y * srcStridesDH.y) + id_x;
    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x;
    float mean = meanTensor[paramIndex];
    float stdDev = stdDevTensor[paramIndex];
    float stdDevSquare = stdDev * stdDev;
    float invStdDev = stdDevSquare ? rsqrtf(stdDevSquare) * scale : 0;
    dstPtr[dstIdx] = fmaf((srcPtr[srcIdx] - mean), invStdDev, shift);
}

__global__ void normalize_nd_hip_tensor(float *srcPtr,
                                        uint *srcMaxDims,
                                        uint *srcStrides,
                                        float *dstPtr,
                                        float *meanTensor,
                                        float *stdDevTensor,
                                        float scale,
                                        float shift,
                                        uint *roiTensor,
                                        uint *paramShapeTensor,
                                        uint *paramStridesTensor,
                                        uint maxParamVolume,
                                        uint numDims)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *roi = &roiTensor[id_z * numDims * 2 + numDims];
    uint *paramShape = &paramShapeTensor[id_z * numDims];
    uint *paramStrides = &paramStridesTensor[id_z * numDims];

    uint paramIndex = 0;
    for (int i = 0; i < numDims; i++)
    {
        uint coord = id_x / srcStrides[i] % srcMaxDims[i];
        if(coord > roi[i])
            return;
        paramIndex += rpp_hip_mod(coord, paramShape[i]) * paramStrides[i];
    }

    float mean = meanTensor[id_z * maxParamVolume + paramIndex];
    float stdDev = stdDevTensor[id_z * maxParamVolume + paramIndex];
    float stdDevSquare = stdDev * stdDev;
    float invStdDev = stdDevSquare ? rsqrtf(stdDevSquare) * scale : 0;
    dstPtr[id_x] = fmaf((srcPtr[id_x] - mean), invStdDev, shift);
}

void normalize_setup(Rpp32u *roiTensor, Rpp32u batchSize, Rpp32u numDims, Rpp32u axisMask,
                     Rpp32u *paramShapeTensor, Rpp32u *paramStridesTensor, Rpp32u &maxParamVolume)
{
    maxParamVolume = 1;
    uint axisSet[RPPT_MAX_DIMS];
    for(int i = 0; i < numDims; i++)
        axisSet[i] = ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : 0;

    for(uint i = 0; i < batchSize; i++)
    {
        // calculate the param shape and param volume based on the axis mask
        Rpp32u paramVolume = 1;
        Rpp32u *roi = &roiTensor[numDims * 2 * i + numDims];
        Rpp32u *paramShape = &paramShapeTensor[i * numDims];
        for(uint j = 0; j < numDims; j++)
        {
            paramShape[j] = (axisSet[j]) ? 1 : roi[j];
            paramVolume *= paramShape[j];
        }
        maxParamVolume = std::max(maxParamVolume, paramVolume);

        // calculate the param strides from the param shape
        Rpp32u *paramStrides = &paramStridesTensor[i * numDims];
        Rpp32u val = 1;
        for(uint j = numDims - 1; j > 0; j--)
        {
            paramStrides[j] = val;
            val *= paramShape[j];
        }
        paramStrides[0] = val;
    }
}

RppStatus hip_exec_normalize_tensor(Rpp32f *srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32u axisMask,
                                    Rpp32f *meanTensor,
                                    Rpp32f *stdDevTensor,
                                    Rpp32u computeMean,
                                    Rpp32u computeStdDev,
                                    Rpp32f scale,
                                    Rpp32f shift,
                                    Rpp32u *roiTensor,
                                    rpp::Handle& handle)
{
    Rpp32u batchSize = srcGenericDescPtr->dims[0];
    Rpp32u numDims = srcGenericDescPtr->numDims - 1; // exclude batchsize from input dims

    // create buffer for paramShape and paramStride
    Rpp32u *paramShape, *paramStrides;
    paramShape =  handle.GetInitHandle()->mem.mgpu.scratchBuf.uintmem;
    paramStrides = handle.GetInitHandle()->mem.mgpu.scratchBuf.uintmem + (batchSize * numDims);

    // do initial preprocessing and fill the values for paramShape and paramStrides
    Rpp32u maxParamVolume;
    normalize_setup(roiTensor, batchSize, numDims, axisMask,
                    paramShape, paramStrides, maxParamVolume);

    if((computeMean == 0) && (computeStdDev == 0))
        maxParamVolume = 0;

    // based on number of dimensions call the corresponding kernel
    if (numDims == 2)
    {
        // NHW
        int globalThreads_x = dstGenericDescPtr->dims[2];
        int globalThreads_y = dstGenericDescPtr->dims[1];
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(normalize_2d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcGenericDescPtr->strides[0], srcGenericDescPtr->strides[1]),
                           dstPtr,
                           make_uint2(dstGenericDescPtr->strides[0], dstGenericDescPtr->strides[1]),
                           meanTensor,
                           stdDevTensor,
                           scale,
                           shift,
                           roiTensor,
                           paramShape,
                           paramStrides,
                           maxParamVolume);
    }
    else if (numDims == 3)
    {
        // NDHW
        int globalThreads_x = dstGenericDescPtr->dims[3];
        int globalThreads_y = dstGenericDescPtr->dims[2];
        int globalThreads_z = dstGenericDescPtr->dims[1];

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            hipLaunchKernelGGL(normalize_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint2(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                               &meanTensor[batchCount * maxParamVolume],
                               &stdDevTensor[batchCount * maxParamVolume],
                               scale,
                               shift,
                               &roiTensor[batchCount * 6 + 3],
                               &paramShape[batchCount * 3],
                               &paramStrides[batchCount * 3],
                               maxParamVolume);
        }
    }
    else
    {
        // interpret the input as 1D tensor
        int globalThreads_x = dstGenericDescPtr->strides[0];
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];

        // allocate tensor for src strides
        Rpp32u *srcMaxDims = handle.GetInitHandle()->mem.mgpu.scratchBuf.uintmem + (2 * batchSize * numDims);
        Rpp32u *srcStrides = handle.GetInitHandle()->mem.mgpu.scratchBuf.uintmem + (3 * batchSize * numDims);
        memcpy(srcMaxDims, &srcGenericDescPtr->dims[1], numDims * sizeof(Rpp32u));
        memcpy(srcStrides, &srcGenericDescPtr->strides[1], numDims * sizeof(Rpp32u));

        hipLaunchKernelGGL(normalize_nd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y), ceil((float)globalThreads_z)),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcMaxDims,
                           srcStrides,
                           dstPtr,
                           meanTensor,
                           stdDevTensor,
                           scale,
                           shift,
                           roiTensor,
                           paramShape,
                           paramStrides,
                           maxParamVolume,
                           numDims);
    }

    return RPP_SUCCESS;
}