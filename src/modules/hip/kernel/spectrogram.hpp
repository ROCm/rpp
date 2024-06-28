#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// compute window output
__global__ void window_output_hip_tensor(float *srcPtr,
                                         uint srcStride,
                                         float *dstPtr,
                                         uint dstStride,
                                         float *windowFn,
                                         int *srcLengthTensor,
                                         int nfft,
                                         int windowLength,
                                         int windowStep,
                                         int windowCenterOffset,
                                         bool centerWindows,
                                         bool reflectPadding)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int srcLength = srcLengthTensor[id_z];
    int numWindows = get_num_windows(srcLength, windowLength, windowStep, centerWindows);

    if (id_x >= windowLength || id_y >= numWindows)
        return;

    int dstIdx = id_z * dstStride + id_y * nfft + id_x;
    int srcIdx = id_z * srcStride;
    int windowStart = id_y * windowStep - windowCenterOffset;
    int inIdx = windowStart + id_x;
    if (windowStart < 0 || (windowStart + windowLength) > srcLength)
    {
        if (reflectPadding)
        {
            inIdx = get_idx_reflect(inIdx, 0, srcLength);
            dstPtr[dstIdx] = windowFn[id_x] * srcPtr[srcIdx + inIdx];
        }
        else
        {
            if (inIdx >= 0 && inIdx < srcLength)
                dstPtr[dstIdx] = windowFn[id_x] * srcPtr[srcIdx + inIdx];
            else
                dstPtr[dstIdx] = 0;
        }
    }
    else
    {
        dstPtr[dstIdx] = windowFn[id_x] * srcPtr[srcIdx + inIdx];
    }
}

RppStatus hip_exec_spectrogram_tensor(Rpp32f* srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f* dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcLengthTensor,
                                      bool centerWindows,
                                      bool reflectPadding,
                                      Rpp32f *windowFunction,
                                      Rpp32s nfft,
                                      Rpp32s power,
                                      Rpp32s windowLength,
                                      Rpp32s windowStep,
                                      rpp::Handle& handle)
{
    Rpp32f *windowFn = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;

    // generate hanning window
    if (windowFunction == NULL)
        hann_window(windowFn, windowLength);
    Rpp32f *d_windowFn = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    hipMemcpyAsync(d_windowFn, windowFn, windowLength * sizeof(Rpp32f), hipMemcpyHostToDevice, handle.GetStream());
    hipStreamSynchronize(handle.GetStream());

    Rpp32s maxSrcLength = *std::max_element(srcLengthTensor, srcLengthTensor + dstDescPtr->n);
    Rpp32s maxNumWindows = get_num_windows(maxSrcLength, windowLength, windowStep, centerWindows);
    Rpp32s windowCenterOffset = 0;
    if (centerWindows) windowCenterOffset = windowLength / 2;

    Rpp32f *windowOutput = d_windowFn + windowLength;

    Rpp32s globalThreads_x = windowLength;
    Rpp32s globalThreads_y = maxNumWindows;
    Rpp32s globalThreads_z = dstDescPtr->n;
    hipLaunchKernelGGL(window_output_hip_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       srcDescPtr->strides.nStride,
                       windowOutput,
                       maxNumWindows * nfft,
                       d_windowFn,
                       srcLengthTensor,
                       nfft,
                       windowLength,
                       windowStep,
                       windowCenterOffset,
                       centerWindows,
                       reflectPadding);

    return RPP_SUCCESS;
}