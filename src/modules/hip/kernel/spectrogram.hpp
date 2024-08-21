#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 -  spectrogram hip kernels --------------------

// compute window output by applying hanning window
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

    // check if windowStart is beyond the bounds of input
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

// compute factors required for fourier transform
__global__ void compute_coefficients_hip_tensor(float *cosFactor,
                                                float *sinFactor,
                                                int numBins,
                                                int nfft)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (id_y >= numBins || id_x >= nfft)
        return;

    float factor = id_x * ((2.0f * id_y * M_PI) / nfft);
    int dstIdx = id_x * numBins + id_y; 
    cosFactor[dstIdx] = cosf(factor);
    sinFactor[dstIdx] = sinf(factor);
}

/* compute fourier transform on windowed output
   it internally computes a matrix multiplication of 
   - windowOutput of size (numWindows, nfft) with cosFactor of size (nfft, nfft/2 + 1) for real part of output
   - windowOutput of size (numWindows, nfft) with sinFactor of size (nfft, nfft/2 + 1) for imaginary part of output */
__global__ void fourier_transform_hip_tensor(float *srcPtr,
                                             uint2 srcStrideNH,
                                             float *dstPtr,
                                             uint2 dstStrideNH,
                                             int *numWindowsTensor,
                                             float *cosFactor,
                                             float *sinFactor,
                                             int4 params_i4,
                                             bool vertical)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int numWindows = numWindowsTensor[id_z];
    int nfft = params_i4.x;
    int numBins = params_i4.y;
    int power = params_i4.z;
    int numTiles = params_i4.w;

    __shared__ float input_smem[16][16];        // 16 rows of src, 16 cols of src in a 16 x 16 thread block
    __shared__ float cosFactor_smem[16][16];    // 16 rows of cosFactor, 16 cols of cosFactor in a 16 x 16 thread block
    __shared__ float sinFactor_smem[16][16];    // 16 rows of sinFactor, 16 cols of sinFactor in a 16 x 16 thread block
    input_smem[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;      // initialization of shared memory to 0 using all 16 x 16 threads
    cosFactor_smem[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;  // initialization of shared memory to 0 using all 16 x 16 threads
    sinFactor_smem[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;  // initialization of shared memory to 0 using all 16 x 16 threads
    __syncthreads();

    int srcIdx = id_z * srcStrideNH.x + id_y * srcStrideNH.y;
    float realVal = 0.0f, imaginaryVal = 0.0f;
    for(int t = 0, offset = 0; t < numTiles; t++, offset += hipBlockDim_x)
    {
        // load input values to shared memory if (id_y, srcCol) < (numWindows, nfft) - range of input
        int srcCol = (offset + hipThreadIdx_x);
        int factorRow = (offset  + hipThreadIdx_y);
        if ((id_y < numWindows) && (srcCol < nfft))
            input_smem[hipThreadIdx_y][hipThreadIdx_x] = srcPtr[srcIdx + srcCol];

        // load cosfactor and sinfactor values to shared memory if (factorRow, id_x) < (nfft, numBins) - range of sinFactor and cosFactor
        if ((factorRow < nfft) && (id_x < numBins))
        {
            factorRow *= numBins;
            cosFactor_smem[hipThreadIdx_y][hipThreadIdx_x] = cosFactor[factorRow + id_x];
            sinFactor_smem[hipThreadIdx_y][hipThreadIdx_x] = sinFactor[factorRow + id_x];
        }

        // wait for all threads to load input, cosFactor and sinFactor values to shared memory
        __syncthreads();

        // do matrix multiplication on the small matrix
        for (int j = 0; j < hipBlockDim_x; j++)
        {
            realVal += (input_smem[hipThreadIdx_y][j] * cosFactor_smem[j][hipThreadIdx_x]);
            imaginaryVal += (-input_smem[hipThreadIdx_y][j] * sinFactor_smem[j][hipThreadIdx_x]);
        }
        __syncthreads();
    }

    // final store to dst
    if (id_y < numWindows && id_x < numBins)
    {
        float magnitudeSquare = ((realVal * realVal) + (imaginaryVal * imaginaryVal));
        
        /* if vertical is set to true, then get the transposed output index (id_x * dstStrideNH.y + id_y)
           else get the normal output index (id_y * dstStrideNH.y + id_x) */
        int dstIdx = (vertical) ? (id_z * dstStrideNH.x + id_x * dstStrideNH.y + id_y) :
                                  (id_z * dstStrideNH.x + id_y * dstStrideNH.y + id_x);
        dstPtr[dstIdx] = (power == 2) ? magnitudeSquare : sqrtf(magnitudeSquare);
    }
}

// -------------------- Set 1 - kernel executor --------------------

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
    // generate hanning window
    Rpp32f *windowFn;
    if (windowFunction == NULL)
    {
        windowFn = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;
        hann_window(windowFn, windowLength);
    }
    else
    {
        windowFn = windowFunction;
    }

    // copy the hanning window values to hip memory
    Rpp32f *d_windowFn = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    CHECK_RETURN_STATUS(hipMemcpyAsync(d_windowFn, windowFn, windowLength * sizeof(Rpp32f), hipMemcpyHostToDevice, handle.GetStream()));
    CHECK_RETURN_STATUS(hipStreamSynchronize(handle.GetStream()));

    // compute the number of windows required for each input in the batch
    Rpp32s *numWindowsTensor = reinterpret_cast<Rpp32s*>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
    for (Rpp32u i = 0; i < dstDescPtr->n; i++)
        numWindowsTensor[i] = get_num_windows(srcLengthTensor[i], windowLength, windowStep, centerWindows);

    // find the maximum windows required across all inputs in batch and stride required for window output
    bool vertical = (dstDescPtr->layout == RpptLayout::NFT);
    Rpp32s maxNumWindows = (vertical) ? dstDescPtr->w : dstDescPtr->h;
    Rpp32s windowCenterOffset = (centerWindows) ? (windowLength / 2) : 0;
    if (!nfft) nfft = windowLength;
    Rpp32u windowOutputStride = maxNumWindows * nfft;

    Rpp32f *windowOutput = d_windowFn + windowLength;
    CHECK_RETURN_STATUS(hipMemsetAsync(windowOutput, 0, windowOutputStride * dstDescPtr->n * sizeof(Rpp32f), handle.GetStream()));
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
                       windowOutputStride,
                       d_windowFn,
                       srcLengthTensor,
                       numWindowsTensor,
                       make_int4(nfft, windowLength, windowStep, windowCenterOffset),
                       reflectPadding);

    // compute the sin and cos factors required for FFT
    Rpp32s numBins = (nfft / 2 + 1);
    Rpp32f *cosTensor, *sinTensor;
    cosTensor = windowOutput + dstDescPtr->n * windowOutputStride;
    sinTensor = cosTensor + (nfft * numBins);
    hipLaunchKernelGGL(compute_coefficients_hip_tensor,
                       dim3(ceil((float)nfft/LOCAL_THREADS_X), ceil((float)numBins/LOCAL_THREADS_Y), ceil((float)1/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       cosTensor,
                       sinTensor,
                       numBins,
                       nfft);

    // compute the final output
    globalThreads_x = numBins;
    Rpp32s numTiles = static_cast<int>(ceil((static_cast<float>(nfft) / LOCAL_THREADS_X)));
    hipLaunchKernelGGL(fourier_transform_hip_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       windowOutput,
                       make_uint2(windowOutputStride, nfft),
                       dstPtr,
                       make_uint2(dstDescPtr->strides.nStride, maxNumWindows),
                       numWindowsTensor,
                       cosTensor,
                       sinTensor,
                       make_int4(nfft, numBins, power, numTiles),
                       vertical);

    return RPP_SUCCESS;
}
