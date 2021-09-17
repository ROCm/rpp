#include <hip/hip_runtime.h>

extern "C" __global__ void roi_converison_ltrb_to_xywh(int *roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 4;

    int4 *roiTensorPtrSrc_i4;
    roiTensorPtrSrc_i4 = (int4 *)&roiTensorPtrSrc[id_x];

    roiTensorPtrSrc_i4->z -= (roiTensorPtrSrc_i4->x - 1);
    roiTensorPtrSrc_i4->w -= (roiTensorPtrSrc_i4->y - 1);
}

RppStatus hip_exec_roi_converison_ltrb_to_xywh(RpptROIPtr roiTensorPtrSrc,
                                               rpp::Handle& handle)
{
    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;
    int globalThreads_x = handle.GetBatchSize();
    int globalThreads_y = 1;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(roi_converison_ltrb_to_xywh,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       (int *) roiTensorPtrSrc);

    return RPP_SUCCESS;
}