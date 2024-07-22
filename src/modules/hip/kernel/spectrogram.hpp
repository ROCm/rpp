#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "hipfft/hipfft.h"
#include <iostream>
#include <fstream>

#define CHECK_HIPFFT_STATUS(x) do { \
  int retval = (x); \
  if (retval != HIPFFT_SUCCESS) { \
    fprintf(stderr, "Runtime error: %s returned %d at %s:%d", #x, retval, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)

// compute window output
__global__ void window_output_hip_tensor(float *srcPtr,
                                         uint srcStride,
                                         float *dstPtr,
                                         uint dstStride,
                                         float *windowFn,
                                         int *srcLengthTensor,
                                         int *numWindowsTensor,
                                         int4 params_i4,
                                         bool reflectPadding)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int srcLength = srcLengthTensor[id_z];
    int numWindows = numWindowsTensor[id_z];
    int nfft = params_i4.x;
    int windowLength = params_i4.y;
    int windowStep = params_i4.z;
    int windowCenterOffset = params_i4.w;

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
        else if (inIdx >= 0 && inIdx < srcLength)
            dstPtr[dstIdx] = windowFn[id_x] * srcPtr[srcIdx + inIdx];
    }
    else
    {
        dstPtr[dstIdx] = windowFn[id_x] * srcPtr[srcIdx + inIdx];
    }
}

__global__ void compute_fft_coefficients_hip_tensor(float *cosFactor,
                                                    float *sinFactor,
                                                    int numBins,
                                                    int nfft)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (id_y >= numBins || id_x >= nfft)
        return;

    float factor = (2.0f * id_y * M_PI) / nfft;
    cosFactor[id_x * numBins + id_y] = cosf(factor * id_x);
    sinFactor[id_x * numBins + id_y] = sinf(factor * id_x);
}

__global__ void compute_fft_tf_hip_tensor(float *srcPtr,
                                          uint2 srcStrideNH,
                                          float *dstPtr,
                                          uint2 dstStrideNH,
                                          int *numWindowsTensor,
                                          int nfft,
                                          int numBins,
                                          float *cosFactor,
                                          float *sinFactor)

{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int numWindows = numWindowsTensor[id_z];

    if (id_y >= numWindows || id_x >= numBins)
        return;

    int srcIdx = id_z * srcStrideNH.x + id_y * srcStrideNH.y;
    int dstIdx = id_z * dstStrideNH.x + id_y * dstStrideNH.y + id_x;
    float real = 0.0f, imag = 0.0f;
    int paramIndex = id_x;
    for(int i = 0 ; i < nfft; i++)
    {
        float x = srcPtr[srcIdx + i];
        real += x * cosFactor[paramIndex + i * numBins];
        imag += -x * sinFactor[paramIndex + i * numBins];
    }
    dstPtr[dstIdx] = (real * real) + (imag * imag);
}

__global__ void compute_fft_ft_hip_tensor(float *srcPtr,
                                          uint2 srcStrideNH,
                                          float *dstPtr,
                                          uint2 dstStrideNH,
                                          int *numWindowsTensor,
                                          int nfft,
                                          int numBins,
                                          float *cosFactor,
                                          float *sinFactor)

{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int numWindows = numWindowsTensor[id_z];

    if (id_y >= numWindows || id_x >= numBins)
        return;

    int srcIdx = id_z * srcStrideNH.x + id_y * srcStrideNH.y;
    int dstIdx = id_z * dstStrideNH.x + id_x * dstStrideNH.y + id_y;
    float real = 0.0f, imag = 0.0f;
    int paramIndex = id_x;
    for(int i = 0 ; i < nfft; i++)
    {
        float x = srcPtr[srcIdx + i];
        real += x * cosFactor[paramIndex + i * numBins];
        imag += -x * sinFactor[paramIndex + i * numBins];
    }
    dstPtr[dstIdx] = (real * real) + (imag * imag);
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
    Rpp32f *windowFn;
    // generate hanning window
    if (windowFunction == NULL)
    {
        windowFn = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;
        hann_window(windowFn, windowLength);
    }
    else
    {
        windowFn = windowFunction;
    }
    Rpp32f *d_windowFn = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    CHECK_RETURN_STATUS(hipMemcpyAsync(d_windowFn, windowFn, windowLength * sizeof(Rpp32f), hipMemcpyHostToDevice, handle.GetStream()));
    CHECK_RETURN_STATUS(hipStreamSynchronize(handle.GetStream()));

    Rpp32s *numWindowsTensor = reinterpret_cast<Rpp32s*>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
    for (Rpp32u i = 0; i < dstDescPtr->n; i++)
        numWindowsTensor[i] = get_num_windows(srcLengthTensor[i], windowLength, windowStep, centerWindows);

    // find the maximum windows required across all inputs in batch
    Rpp32s maxNumWindows = *std::max_element(numWindowsTensor, numWindowsTensor + dstDescPtr->n);
    Rpp32s windowCenterOffset = (centerWindows) ? (windowLength / 2) : 0;

    Rpp32f *windowOutput = d_windowFn + windowLength;
    CHECK_RETURN_STATUS(hipMemsetAsync(windowOutput, 0, maxNumWindows * nfft * dstDescPtr->n * sizeof(Rpp32f), handle.GetStream()));
    CHECK_RETURN_STATUS(hipStreamSynchronize(handle.GetStream()));
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
                       numWindowsTensor,
                       make_int4(nfft, windowLength, windowStep, windowCenterOffset),
                       reflectPadding);

    // compute the sin and cos factors required for FFT
    Rpp32s numBins = (nfft / 2 + 1);
    Rpp32f *cosfTensor, *sinfTensor;
    CHECK_RETURN_STATUS(hipMalloc(&cosfTensor, nfft * numBins * sizeof(Rpp32f)));
    CHECK_RETURN_STATUS(hipMalloc(&sinfTensor, nfft * numBins * sizeof(Rpp32f)));
    hipLaunchKernelGGL(compute_fft_coefficients_hip_tensor,
                       dim3(ceil((float)nfft/LOCAL_THREADS_X), ceil((float)numBins/LOCAL_THREADS_Y), ceil((float)1/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       cosfTensor,
                       sinfTensor,
                       numBins,
                       nfft);

    // compute the final output
    globalThreads_x = numBins;
    if (dstDescPtr->layout == RpptLayout::NTF)
    {
        hipLaunchKernelGGL(compute_fft_tf_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           windowOutput,
                           make_uint2(maxNumWindows * nfft, nfft),
                           dstPtr,
                           make_uint2(maxNumWindows * numBins, numBins),
                           numWindowsTensor,
                           nfft,
                           numBins,
                           cosfTensor,
                           sinfTensor);
    }
    else if (dstDescPtr->layout == RpptLayout::NFT)
    {
        hipLaunchKernelGGL(compute_fft_ft_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           windowOutput,
                           make_uint2(maxNumWindows * nfft, nfft),
                           dstPtr,
                           make_uint2(maxNumWindows * numBins, maxNumWindows),
                           numWindowsTensor,
                           nfft,
                           numBins,
                           cosfTensor,
                           sinfTensor);
    }
    CHECK_RETURN_STATUS(hipStreamSynchronize(handle.GetStream()));
    CHECK_RETURN_STATUS(hipFree(cosfTensor));
    CHECK_RETURN_STATUS(hipFree(sinfTensor));

    return RPP_SUCCESS;
}