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

// do post processing
__global__ void post_process_hip_tensor(std::complex<float> *srcPtr,
                                        uint2 srcStrideNH,
                                        float *dstPtr,
                                        uint2 dstStrideNH,
                                        int *numWindowsTensor,
                                        int numBins)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int numWindows = numWindowsTensor[id_z];

    if (id_y >= numWindows || id_x >= numBins)
        return;

    int srcIdx = id_z * srcStrideNH.x + id_y * srcStrideNH.y + id_x;
    int dstIdx = id_z * dstStrideNH.x + id_y * dstStrideNH.y + id_x;
    dstPtr[dstIdx] = norm(srcPtr[srcIdx]);
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
    hipMemcpyAsync(d_windowFn, windowFn, windowLength * sizeof(Rpp32f), hipMemcpyHostToDevice, handle.GetStream());
    hipStreamSynchronize(handle.GetStream());

    Rpp32s *numWindowsTensor = reinterpret_cast<Rpp32s*>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
    for (Rpp32u i = 0; i < dstDescPtr->n; i++)
        numWindowsTensor[i] = get_num_windows(srcLengthTensor[i], windowLength, windowStep, centerWindows);

    // find the maximum windows required across all inputs in batch
    Rpp32s maxNumWindows = *std::max_element(numWindowsTensor, numWindowsTensor + dstDescPtr->n);
    Rpp32s windowCenterOffset = (centerWindows) ? (windowLength / 2) : 0;

    Rpp32f *windowOutput = d_windowFn + windowLength;
    hipMemsetAsync(windowOutput, 0, maxNumWindows * nfft * dstDescPtr->n * sizeof(Rpp32f), handle.GetStream());
    hipStreamSynchronize(handle.GetStream());
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
    hipStreamSynchronize(handle.GetStream());

    // Declare the fftIn and fftOut buffers
    Rpp32s numBins = (nfft / 2 + 1);
    hipfftReal *fftIn = reinterpret_cast<hipfftReal *>(windowOutput);
    hipfftComplex *fftOut;
    hipMalloc((void**)&fftOut, sizeof(hipfftComplex) * maxNumWindows * nfft * globalThreads_z);

    // create the fft plan required
    RppSize_t workSize = 0;
    hipfftHandle fftHandle;
    CHECK_HIPFFT_STATUS(hipfftCreate(&fftHandle));
    CHECK_HIPFFT_STATUS(hipfftSetStream(fftHandle, handle.GetStream()));

    Rpp32s n[1] = {nfft};
    // execute fft for all inputs in batch
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        hipfftReal *fftInTemp = fftIn + batchCount * maxNumWindows * nfft;
        hipfftComplex *fftOutTemp = fftOut + batchCount * maxNumWindows * numBins;
        Rpp32s nWindows = get_num_windows(srcLengthTensor[0], windowLength, windowStep, centerWindows);
        CHECK_HIPFFT_STATUS(hipfftPlanMany(&fftHandle, 1, n, NULL, 0, 0, NULL, 0, 0, HIPFFT_R2C, nWindows));
        CHECK_HIPFFT_STATUS(hipfftExecR2C(fftHandle, fftInTemp, fftOutTemp));
        hipStreamSynchronize(handle.GetStream());
    }

    globalThreads_x = numBins;
    hipLaunchKernelGGL(post_process_hip_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       reinterpret_cast<std::complex<Rpp32f> *>(fftOut),
                       make_uint2(maxNumWindows * numBins, numBins),
                       dstPtr,
                       make_uint2(maxNumWindows * numBins, numBins),
                       numWindowsTensor,
                       numBins);
    hipStreamSynchronize(handle.GetStream());
    hipFree(fftOut);
    hipfftDestroy(fftHandle);

    return RPP_SUCCESS;
}