/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "hip_tensor_executors.hpp"

// YCbCr conversion coefficients (ITU-R BT.601)
__device__ constexpr float Y_FROM_R = 0.299f;
__device__ constexpr float Y_FROM_G = 0.587f;
__device__ constexpr float Y_FROM_B = 0.114f;

__device__ constexpr float CB_FROM_R = -0.168736f;
__device__ constexpr float CB_FROM_G = -0.331264f;
__device__ constexpr float CB_FROM_B = 0.5f;

__device__ constexpr float CR_FROM_R = 0.5f;
__device__ constexpr float CR_FROM_G = -0.418688f;
__device__ constexpr float CR_FROM_B = -0.081312f;

// YCbCr to RGB conversion coefficients
__device__ constexpr float R_FROM_CR = 1.402f;
__device__ constexpr float G_FROM_CB = -0.344136f;
__device__ constexpr float G_FROM_CR = -0.714136f;
__device__ constexpr float B_FROM_CB = 1.772f;

// Common constants
__device__ constexpr float MAX_PIXEL_VALUE = 255.0f;
__device__ constexpr float CHROMA_OFFSET = 128.0f;
__device__ constexpr int HISTOGRAM_BINS = 256;

// Vectorized constants for SIMD operations
__device__ const float4 yR_f4 = {Y_FROM_R, Y_FROM_R, Y_FROM_R, Y_FROM_R};
__device__ const float4 yG_f4 = {Y_FROM_G, Y_FROM_G, Y_FROM_G, Y_FROM_G};
__device__ const float4 yB_f4 = {Y_FROM_B, Y_FROM_B, Y_FROM_B, Y_FROM_B};

__device__ const float4 cbR_f4 = {CB_FROM_R ,CB_FROM_R, CB_FROM_R, CB_FROM_R};
__device__ const float4 cbG_f4 = {CB_FROM_G ,CB_FROM_G, CB_FROM_G, CB_FROM_G};
__device__ const float4 cbB_f4 = {CB_FROM_B ,CB_FROM_B, CB_FROM_B, CB_FROM_B};

__device__ const float4 crR_f4 = {CR_FROM_R, CR_FROM_R, CR_FROM_R, CR_FROM_R};
__device__ const float4 crG_f4 = {CR_FROM_G, CR_FROM_G, CR_FROM_G, CR_FROM_G};
__device__ const float4 crB_f4 = {CR_FROM_B, CR_FROM_B, CR_FROM_B, CR_FROM_B};

__device__ const float4 maxVal255_f4 = {MAX_PIXEL_VALUE, MAX_PIXEL_VALUE, MAX_PIXEL_VALUE, MAX_PIXEL_VALUE};
__device__ const float4 maxVal128_f4 = {CHROMA_OFFSET, CHROMA_OFFSET, CHROMA_OFFSET, CHROMA_OFFSET};

__device__ inline float4 clamp(float4 v, float lo, float hi)
{
    v.x = fminf(fmaxf(v.x, lo), hi);
    v.y = fminf(fmaxf(v.y, lo), hi);
    v.z = fminf(fmaxf(v.z, lo), hi);
    v.w = fminf(fmaxf(v.w, lo), hi);
    return v;
}

__device__ inline void rgb_to_ycbcr_hip_compute(d_float24 &rgb_f24, d_float8 &y_f8, d_float8 &cb_f8, d_float8 &cr_f8)
{
    // Y = 0.299 * R + 0.587 * G + 0.114 * B
    y_f8.f4[0] = clamp((rgb_f24.f4[0] * yR_f4) + (rgb_f24.f4[2] * yG_f4) + (rgb_f24.f4[4] * yB_f4), 0.0f, MAX_PIXEL_VALUE);
    y_f8.f4[1] = clamp((rgb_f24.f4[1] * yR_f4) + (rgb_f24.f4[3] * yG_f4) + (rgb_f24.f4[5] * yB_f4), 0.0f, MAX_PIXEL_VALUE);

    // Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    cb_f8.f4[0] = clamp((rgb_f24.f4[0] * cbR_f4) + (rgb_f24.f4[2] * cbG_f4) + (rgb_f24.f4[4] * cbB_f4) + maxVal128_f4, 0.0f, MAX_PIXEL_VALUE);
    cb_f8.f4[1] = clamp((rgb_f24.f4[1] * cbR_f4) + (rgb_f24.f4[3] * cbG_f4) + (rgb_f24.f4[5] * cbB_f4) + maxVal128_f4, 0.0f, MAX_PIXEL_VALUE);

    // Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128
    cr_f8.f4[0] = clamp((rgb_f24.f4[0] * crR_f4) + (rgb_f24.f4[2] * crG_f4) + (rgb_f24.f4[4] * crB_f4) + maxVal128_f4, 0.0f, MAX_PIXEL_VALUE);
    cr_f8.f4[1] = clamp((rgb_f24.f4[1] * crR_f4) + (rgb_f24.f4[3] * crG_f4) + (rgb_f24.f4[5] * crB_f4) + maxVal128_f4, 0.0f, MAX_PIXEL_VALUE);
}

__device__ inline void ycbcr_to_rgb_hip_compute(d_float24 &rgb_f24, d_float8 &y_f8, d_float8 &cb_f8, d_float8 &cr_f8)
{
    // Subtract 128 from Cb and Cr
    cb_f8.f4[0] -= MAKE_FLOAT4(CHROMA_OFFSET);
    cb_f8.f4[1] -= MAKE_FLOAT4(CHROMA_OFFSET);
    cr_f8.f4[0] -= MAKE_FLOAT4(CHROMA_OFFSET);
    cr_f8.f4[1] -= MAKE_FLOAT4(CHROMA_OFFSET);

    // R = Y + 1.402 * Cr
    rgb_f24.f4[0] = clamp((y_f8.f4[0] + MAKE_FLOAT4(R_FROM_CR) * cr_f8.f4[0]), 0.0f, MAX_PIXEL_VALUE);
    rgb_f24.f4[1] = clamp((y_f8.f4[1] + MAKE_FLOAT4(R_FROM_CR) * cr_f8.f4[1]), 0.0f, MAX_PIXEL_VALUE);

    // G = Y - 0.344136 * Cb - 0.714136 * Cr
    rgb_f24.f4[2] = clamp((y_f8.f4[0] + (MAKE_FLOAT4(G_FROM_CB) * cb_f8.f4[0]) + (MAKE_FLOAT4(G_FROM_CR) * cr_f8.f4[0])), 0.0f, MAX_PIXEL_VALUE);
    rgb_f24.f4[3] = clamp((y_f8.f4[1] + (MAKE_FLOAT4(G_FROM_CB) * cb_f8.f4[1]) + (MAKE_FLOAT4(G_FROM_CR) * cr_f8.f4[1])), 0.0f, MAX_PIXEL_VALUE);

    // B = Y + 1.772 * Cb
    rgb_f24.f4[4] = clamp((y_f8.f4[0] + MAKE_FLOAT4(B_FROM_CB) * cb_f8.f4[0]), 0.0f, MAX_PIXEL_VALUE);
    rgb_f24.f4[5] = clamp((y_f8.f4[1] + MAKE_FLOAT4(B_FROM_CB) * cb_f8.f4[1]), 0.0f, MAX_PIXEL_VALUE);
}

__global__ void collect_hist_pln_hip_tensor(const unsigned char *__restrict__ srcPtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           uint3 srcStridesNCH,
                                           unsigned int *__restrict__ hist)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint histOffset = id_z * HISTOGRAM_BINS;
    uint8_t pixVal = srcPtr[srcIdx];
    atomicAdd(&hist[histOffset + pixVal], 1);
}

__global__ void build_lut_from_hist_kernel(const unsigned int* __restrict__ hist,
                                          unsigned char* __restrict__ lut,
                                          const int* __restrict__ img_sizes,
                                          int batchSize)
{
    int batch = blockIdx.x;
    if (batch >= batchSize) return;

    int tid = threadIdx.x;
    __shared__ unsigned int cdf_shared[HISTOGRAM_BINS];
    __shared__ unsigned int min_cdf_shared;
    
    if (tid == 0) min_cdf_shared = 0;
    __syncthreads();

    // Initialize shared memory
    for (int i = tid; i < HISTOGRAM_BINS; i += blockDim.x)
    {
        cdf_shared[i] = 0;
    }
    __syncthreads();

    // Load histogram values to shared memory
    for (int i = tid; i < HISTOGRAM_BINS; i += blockDim.x)
    {
        unsigned int val = hist[batch * HISTOGRAM_BINS + i];
        atomicAdd(&cdf_shared[i], val);
    }
    __syncthreads();

    // Compute CDF and find minimum non-zero CDF value
    if (tid == 0)
    {
        unsigned int cdf_accum = 0;
        for (int i = 0; i < HISTOGRAM_BINS; ++i)
        {
            cdf_accum += cdf_shared[i];
            cdf_shared[i] = cdf_accum;
            if (min_cdf_shared == 0 && cdf_shared[i] != 0)
                min_cdf_shared = cdf_shared[i];
        }
    }
    __syncthreads();

    // Build equalization lookup table
    int N = img_sizes[batch];
    for (int i = tid; i < HISTOGRAM_BINS; i += blockDim.x)
    {
        // Branchless computation to avoid thread divergence
        // When min_cdf_shared == N, denominator becomes 0, so we handle it with max()
        float denominator = fmaxf((float)(N - min_cdf_shared), 1.0f);
        // Calculate equalized value
        unsigned char equalized_val = (unsigned char)(roundf((float)((cdf_shared[i] - min_cdf_shared) * MAX_PIXEL_VALUE) / denominator));

        // When min_cdf_shared == N (uniform image), use original value
        // This is done without branching: select between equalized_val and i
        unsigned char is_uniform = (min_cdf_shared == N);
        lut[batch * HISTOGRAM_BINS + i] = is_uniform * i + (1 - is_uniform) * equalized_val;
    }
}

__global__ void apply_lut_pln1_hip_tensor(const unsigned char *__restrict__ srcPtr,
                                         uint3 srcStridesNCH,
                                         unsigned char *__restrict__ dstPtr,
                                         uint3 dstStridesNCH,
                                         const unsigned char* __restrict__ lut,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    unsigned char pixVal = srcPtr[srcIdx];
    dstPtr[dstIdx] = lut[id_z * HISTOGRAM_BINS + pixVal];
}

__global__ void convert_rgb_pkd3_to_ycbcr_pln3(unsigned char *__restrict__ srcPtr,
                                               uint2 srcStridesNH,
                                               unsigned char *__restrict__ yPtr,
                                               unsigned char *__restrict__ cbPtr,
                                               unsigned char *__restrict__ crPtr,
                                               uint2 dstStridesWH,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesWH.y * dstStridesWH.x) + (id_y * dstStridesWH.x) + id_x;
    
    d_float24 rgb_f24;
    d_float8 y_f8, cb_f8, cr_f8;
    
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &rgb_f24);
    rgb_to_ycbcr_hip_compute(rgb_f24, y_f8, cb_f8, cr_f8);
    
    rpp_hip_pack_float8_and_store8(yPtr + dstIdx, &y_f8);
    rpp_hip_pack_float8_and_store8(cbPtr + dstIdx, &cb_f8);
    rpp_hip_pack_float8_and_store8(crPtr + dstIdx, &cr_f8);
}

__global__ void convert_rgb_pln3_to_ycbcr_pln3(unsigned char *__restrict__ srcPtr,
                                               uint3 srcStridesNCH,
                                               unsigned char *__restrict__ yPtr,
                                               unsigned char *__restrict__ cbPtr,
                                               unsigned char *__restrict__ crPtr,
                                               uint2 dstStridesNH,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
    
    d_float24 rgb_f24;
    d_float8 y_f8, cb_f8, cr_f8;
    
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &rgb_f24);
    rgb_to_ycbcr_hip_compute(rgb_f24, y_f8, cb_f8, cr_f8);
    
    rpp_hip_pack_float8_and_store8(yPtr + dstIdx, &y_f8);
    rpp_hip_pack_float8_and_store8(cbPtr + dstIdx, &cb_f8);
    rpp_hip_pack_float8_and_store8(crPtr + dstIdx, &cr_f8);
}

__global__ void convert_ycbcr_pln3_to_rgb_pln3(unsigned char *__restrict__ yPtr,
                                               unsigned char *__restrict__ cbPtr,
                                               unsigned char *__restrict__ crPtr,
                                               uint2 srcStridesWH,
                                               unsigned char *__restrict__ dstPtr,
                                               uint3 dstStridesNCH,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesWH.y * srcStridesWH.x) + (id_y * srcStridesWH.x) + id_x;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 rgb_f24;
    d_float8 y_f8, cb_f8, cr_f8;

    rpp_hip_load8_and_unpack_to_float8(yPtr + srcIdx, &y_f8);
    rpp_hip_load8_and_unpack_to_float8(cbPtr + srcIdx, &cb_f8);
    rpp_hip_load8_and_unpack_to_float8(crPtr + srcIdx, &cr_f8);

    ycbcr_to_rgb_hip_compute(rgb_f24, y_f8, cb_f8, cr_f8);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &rgb_f24);
}

__global__ void convert_ycbcr_pln3_to_rgb_pkd3(unsigned char *__restrict__ yPtr,
                                               unsigned char *__restrict__ cbPtr,
                                               unsigned char *__restrict__ crPtr,
                                               uint2 srcStridesWH,
                                               unsigned char *__restrict__ dstPtr,
                                               uint2 dstStridesNH,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesWH.y * srcStridesWH.x) + (id_y * srcStridesWH.x) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    d_float24 rgb_f24;
    d_float8 y_f8, cb_f8, cr_f8;

    rpp_hip_load8_and_unpack_to_float8(yPtr + srcIdx, &y_f8);
    rpp_hip_load8_and_unpack_to_float8(cbPtr + srcIdx, &cb_f8);
    rpp_hip_load8_and_unpack_to_float8(crPtr + srcIdx, &cr_f8);

    ycbcr_to_rgb_hip_compute(rgb_f24, y_f8, cb_f8, cr_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &rgb_f24);
}

inline void calculate_global_threads(int &globalThreads_x, int &globalThreads_y, int &globalThreads_z,
                                    int width, int height, int batchSize, int vectorizationFactor = 1)
{
    globalThreads_x = (width + vectorizationFactor - 1) / vectorizationFactor;
    globalThreads_y = height;
    globalThreads_z = batchSize;
}

RppStatus hip_exec_histogram_equalize_tensor(Rpp8u *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8u *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int batchSize = dstDescPtr->n;
    
    // Use handle's scratch buffers for histogram and LUT
    unsigned int* d_hist = reinterpret_cast<unsigned int*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
    unsigned char* d_lut = reinterpret_cast<unsigned char*>(d_hist + batchSize * HISTOGRAM_BINS);

    // Process 3-channel images
    if(srcDescPtr->c == 3)
    {
        // Allocate temporary buffers for YCbCr channels
        const size_t planeSize = static_cast<size_t>(srcDescPtr->w) * srcDescPtr->h * srcDescPtr->n;
        Rpp8u *yuvBuf = nullptr;
        hipMalloc(&yuvBuf, planeSize * 3);

        Rpp8u *yBuf = yuvBuf;
        Rpp8u *cbBuf = yuvBuf + planeSize;
        Rpp8u *crBuf = yuvBuf + (planeSize * 2);

        if(srcDescPtr->layout == RpptLayout::NHWC)
        {
            int globalThreads_x, globalThreads_y, globalThreads_z;
            
            // Convert RGB to YCbCr
            globalThreads_x = (srcDescPtr->w + 7) >> 3;
            globalThreads_y = dstDescPtr->h;
            globalThreads_z = handle.GetBatchSize();
            hipLaunchKernelGGL(convert_rgb_pkd3_to_ycbcr_pln3,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), 
                                   ceil((float)globalThreads_y/LOCAL_THREADS_Y), 
                                   ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               yBuf, cbBuf, crBuf,
                               make_uint2(srcDescPtr->w, srcDescPtr->h),
                               roiTensorPtrSrc);

            // Zero histogram buffer
            hipMemsetAsync(d_hist, 0, batchSize * HISTOGRAM_BINS * sizeof(unsigned int), handle.GetStream());

            // Collect histogram for Y channel
            calculate_global_threads(globalThreads_x, globalThreads_y, globalThreads_z,
                                   srcDescPtr->w, srcDescPtr->h, srcDescPtr->n);
                                   
            hipLaunchKernelGGL(collect_hist_pln_hip_tensor,
                              dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), 
                                   ceil((float)globalThreads_y/LOCAL_THREADS_Y), 
                                   ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                              dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                              0,
                              handle.GetStream(),
                              yBuf,
                              roiTensorPtrSrc,
                              make_uint3(srcDescPtr->w * srcDescPtr->h, srcDescPtr->w * srcDescPtr->h, srcDescPtr->w),
                              d_hist);

            // Build LUTs from histograms
            std::vector<int> img_sizes(batchSize);
            for (int b = 0; b < batchSize; ++b)
            {
                int w = roiTensorPtrSrc[b].xywhROI.roiWidth;
                int h = roiTensorPtrSrc[b].xywhROI.roiHeight;
                img_sizes[b] = w * h;
            }
            
            int* d_img_sizes;
            hipMalloc(&d_img_sizes, batchSize * sizeof(int));
            hipMemcpyAsync(d_img_sizes, img_sizes.data(), batchSize * sizeof(int), hipMemcpyHostToDevice, handle.GetStream());

            hipLaunchKernelGGL(build_lut_from_hist_kernel, 
                              dim3(batchSize), 
                              dim3(HISTOGRAM_BINS), 
                              0, 
                              handle.GetStream(),
                              d_hist, d_lut, d_img_sizes, batchSize);
                              
            hipFree(d_img_sizes);

            // Apply LUT to Y channel
            calculate_global_threads(globalThreads_x, globalThreads_y, globalThreads_z,
                                   dstDescPtr->w, dstDescPtr->h, dstDescPtr->n);

            hipLaunchKernelGGL(apply_lut_pln1_hip_tensor,
                              dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), 
                                   ceil((float)globalThreads_y/LOCAL_THREADS_Y), 
                                   ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                              dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                              0,
                              handle.GetStream(),
                              yBuf,
                              make_uint3(srcDescPtr->w * srcDescPtr->h, srcDescPtr->w * srcDescPtr->h, srcDescPtr->w),
                              yBuf,
                              make_uint3(srcDescPtr->w * srcDescPtr->h, srcDescPtr->w * srcDescPtr->h, srcDescPtr->w),
                              d_lut,
                              roiTensorPtrSrc);

            // Convert YCbCr back to RGB
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            globalThreads_y = dstDescPtr->h;
            globalThreads_z = handle.GetBatchSize();
                
            if(dstDescPtr->layout == RpptLayout::NHWC)
            {
                                hipLaunchKernelGGL(convert_ycbcr_pln3_to_rgb_pkd3,
                                dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), 
                                    ceil((float)globalThreads_y/LOCAL_THREADS_Y), 
                                    ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                0,
                                handle.GetStream(),
                                yBuf, cbBuf, crBuf,
                                make_uint2(srcDescPtr->w, srcDescPtr->h),
                                dstPtr,
                                make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                roiTensorPtrSrc);
            }
            else if(dstDescPtr->layout == RpptLayout::NCHW)
            {
                                hipLaunchKernelGGL(convert_ycbcr_pln3_to_rgb_pln3,
                                dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), 
                                    ceil((float)globalThreads_y/LOCAL_THREADS_Y), 
                                    ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                0,
                                handle.GetStream(),
                                yBuf, cbBuf, crBuf,
                                make_uint2(srcDescPtr->w, srcDescPtr->h),
                                dstPtr,
                                make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                roiTensorPtrSrc);
            }
        }
        else if(srcDescPtr->layout == RpptLayout::NCHW)
        {
            int globalThreads_x, globalThreads_y, globalThreads_z;
            
            // Convert RGB to YCbCr
            calculate_global_threads(globalThreads_x, globalThreads_y, globalThreads_z,
                                     srcDescPtr->w, srcDescPtr->h, srcDescPtr->n, 8);

            hipLaunchKernelGGL(convert_rgb_pln3_to_ycbcr_pln3,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), 
                                   ceil((float)globalThreads_y/LOCAL_THREADS_Y), 
                                   ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               yBuf, cbBuf, crBuf,
                               make_uint2(srcDescPtr->w * srcDescPtr->h, srcDescPtr->w),
                               roiTensorPtrSrc);

            // Zero histogram buffer
            hipMemsetAsync(d_hist, 0, batchSize * HISTOGRAM_BINS * sizeof(unsigned int), handle.GetStream());

            // Collect histogram for Y channel
            calculate_global_threads(globalThreads_x, globalThreads_y, globalThreads_z,
                                   srcDescPtr->w, srcDescPtr->h, srcDescPtr->n);
                                   
            hipLaunchKernelGGL(collect_hist_pln_hip_tensor,
                              dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), 
                                   ceil((float)globalThreads_y/LOCAL_THREADS_Y), 
                                   ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                              dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                              0,
                              handle.GetStream(),
                              yBuf,
                              roiTensorPtrSrc,
                              make_uint3(srcDescPtr->w * srcDescPtr->h, srcDescPtr->w * srcDescPtr->h, srcDescPtr->w),
                              d_hist);

            // Build LUTs from histograms
            std::vector<int> img_sizes(batchSize);
            for (int b = 0; b < batchSize; ++b)
            {
                int w = roiTensorPtrSrc[b].xywhROI.roiWidth;
                int h = roiTensorPtrSrc[b].xywhROI.roiHeight;
                img_sizes[b] = w * h;
            }
            
            int* d_img_sizes;
            hipMalloc(&d_img_sizes, batchSize * sizeof(int));
            hipMemcpyAsync(d_img_sizes, img_sizes.data(), batchSize * sizeof(int), hipMemcpyHostToDevice, handle.GetStream());

            hipLaunchKernelGGL(build_lut_from_hist_kernel, 
                              dim3(batchSize), 
                              dim3(HISTOGRAM_BINS), 
                              0, 
                              handle.GetStream(),
                              d_hist, d_lut, d_img_sizes, batchSize);
                              
            hipFree(d_img_sizes);

            // Apply LUT to Y channel
            calculate_global_threads(globalThreads_x, globalThreads_y, globalThreads_z,
                                   dstDescPtr->w, dstDescPtr->h, dstDescPtr->n);

            hipLaunchKernelGGL(apply_lut_pln1_hip_tensor,
                              dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), 
                                   ceil((float)globalThreads_y/LOCAL_THREADS_Y), 
                                   ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                              dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                              0,
                              handle.GetStream(),
                              yBuf,
                              make_uint3(srcDescPtr->w * srcDescPtr->h, srcDescPtr->w * srcDescPtr->h, srcDescPtr->w),
                              yBuf,
                              make_uint3(srcDescPtr->w * srcDescPtr->h, srcDescPtr->w * srcDescPtr->h, srcDescPtr->w),
                              d_lut,
                              roiTensorPtrSrc);

            // Convert YCbCr back to RGB
            calculate_global_threads(globalThreads_x, globalThreads_y, globalThreads_z,
                                   dstDescPtr->w, dstDescPtr->h, dstDescPtr->n, 8);
                
            if(dstDescPtr->layout == RpptLayout::NHWC)
            {
                                hipLaunchKernelGGL(convert_ycbcr_pln3_to_rgb_pkd3,
                                dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), 
                                    ceil((float)globalThreads_y/LOCAL_THREADS_Y), 
                                    ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                0,
                                handle.GetStream(),
                                yBuf, cbBuf, crBuf,
                                make_uint2(srcDescPtr->w, srcDescPtr->h),
                                dstPtr,
                                make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                roiTensorPtrSrc);
            }
            else if(dstDescPtr->layout == RpptLayout::NCHW)
            {
                                hipLaunchKernelGGL(convert_ycbcr_pln3_to_rgb_pln3,
                                dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), 
                                    ceil((float)globalThreads_y/LOCAL_THREADS_Y), 
                                    ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                0,
                                handle.GetStream(),
                                yBuf, cbBuf, crBuf,
                                make_uint2(srcDescPtr->w, srcDescPtr->h),
                                dstPtr,
                                make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                roiTensorPtrSrc);
            }
        }

        hipFree(yuvBuf);
        return RPP_SUCCESS;
    }

    // Process single-channel images
    hipMemsetAsync(d_hist, 0, batchSize * HISTOGRAM_BINS * sizeof(unsigned int), handle.GetStream());

    int globalThreads_x, globalThreads_y, globalThreads_z;
    
    // Collect histogram
    calculate_global_threads(globalThreads_x, globalThreads_y, globalThreads_z,
                           srcDescPtr->w, srcDescPtr->h, srcDescPtr->n);
                           
    hipLaunchKernelGGL(collect_hist_pln_hip_tensor,
                      dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), 
                           ceil((float)globalThreads_y/LOCAL_THREADS_Y), 
                           ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                      dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                      0,
                      handle.GetStream(),
                      srcPtr,
                      roiTensorPtrSrc,
                      make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                      d_hist);

    // Build LUTs
    std::vector<int> img_sizes(batchSize);
    for (int b = 0; b < batchSize; ++b)
    {
        int w = roiTensorPtrSrc[b].xywhROI.roiWidth;
        int h = roiTensorPtrSrc[b].xywhROI.roiHeight;
        img_sizes[b] = w * h;
    }
    
    int* d_img_sizes;
    hipMalloc(&d_img_sizes, batchSize * sizeof(int));
    hipMemcpyAsync(d_img_sizes, img_sizes.data(), batchSize * sizeof(int), hipMemcpyHostToDevice, handle.GetStream());

    hipLaunchKernelGGL(build_lut_from_hist_kernel, 
                      dim3(batchSize), 
                      dim3(HISTOGRAM_BINS), 
                      0, 
                      handle.GetStream(),
                      d_hist, d_lut, d_img_sizes, batchSize);
                      
    hipFree(d_img_sizes);

    // Apply LUT
    calculate_global_threads(globalThreads_x, globalThreads_y, globalThreads_z,
                           dstDescPtr->w, dstDescPtr->h, dstDescPtr->n);

    hipLaunchKernelGGL(apply_lut_pln1_hip_tensor,
                      dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), 
                           ceil((float)globalThreads_y/LOCAL_THREADS_Y), 
                           ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                      dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                      0,
                      handle.GetStream(),
                      srcPtr,
                      make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                      dstPtr,
                      make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                      d_lut,
                      roiTensorPtrSrc);

    return RPP_SUCCESS;
}
