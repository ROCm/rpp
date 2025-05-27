/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#include "host_tensor_executors.hpp"
#include "rpp_cpu_filter.hpp"

inline void rpp_store_filter_3x3_host(Rpp8u *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);
}

inline void rpp_store_filter_3x3_host(Rpp8s *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_f32_to_i8_avx(dstPtrTemp, pDst);
}

inline void rpp_store_filter_3x3_host(Rpp32f *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_f32_to_f32_avx(dstPtrTemp, pDst);
}

inline void rpp_store_filter_3x3_host(Rpp16f *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_f32_to_f16_avx(dstPtrTemp, pDst);
}

inline void create_emboss_kernel_host(Rpp32f* filter, Rpp32f strength)
{
    // Add bias to prevent dark output
    // bias = 2.0f;
    
    Rpp32f embossKernel[9] = {
        2.0f,  1.0f,   0.0f,
        1.0f,  1.0f,  -1.0f,
        0.0f, -1.0f,  -2.0f
    };
    //  Rpp32f embossKernel[9] = {
    //       -1.0f, -1.0f,  0.0f,
    //       -1.0f,  0.0f,  1.0f,
    //        0.0f,  1.0f,  1.0f
    // };

    Rpp32f clampedStrength = (strength > 2.0f) ? 2.0f : strength;

    
    for (int i = 0; i < 9; i++)
        filter[i] = embossKernel[i] * clampedStrength;
        
    // Add bias to center pixel
    // filter[4] += bias;

    // Flip the kernel horizontally and vertically in-place
    // for (int i = 0; i < 3 / 2; ++i) {
    //     for (int j = 0; j < 3; ++j) {
    //         std::swap(filter[i * 3 + j], filter[(3 - 1 - i) * 3 + (3 - 1 - j)]);
    //     }
    // }
}

template<typename T>
RppStatus emboss_host_tensor(T *srcPtr,
                             RpptDescPtr srcDescPtr,
                             T *dstPtr,
                             RpptDescPtr dstDescPtr,
                             Rpp32f *strength,
                             Rpp32f *bias,
                             RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType,
                             RppLayoutParams layoutParams,
                             rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    // set the required masks array needed for shuffle operations
#if __AVX2__
    __m256i pxMaskPln[7] = {avx_pxMaskRotate0To1, avx_pxMaskRotate0To2, avx_pxMaskRotate0To3, avx_pxMaskRotate0To4, avx_pxMaskRotate0To5, avx_pxMaskRotate0To6, avx_pxMaskRotate0To7};
    __m256i pxMaskPkd[7] = {avx_pxMaskRotate0To3, avx_pxMaskRotate0To6, avx_pxMaskRotate0To1, avx_pxMaskRotate0To4, avx_pxMaskRotate0To7, avx_pxMaskRotate0To2, avx_pxMaskRotate0To5};
#endif
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u padLength = 3 / 2;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
        Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;

        Rpp32f biasParam = bias[batchCount];

        Rpp32f *filterTensor = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + batchCount * 3 * 3;
        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        create_emboss_kernel_host(filterTensor, strength[batchCount]);
#if __AVX2__
        int size = 3 * 3;
        __m256 pFilter[size];
        for (int i = 0; i < size; i++)
            pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
        T *srcPtrRow[3], *dstPtrRow;
        for (int i = 0; i < 3; i++)
            srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
        dstPtrRow = dstPtrChannel;

        // emboss filter without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            /* exclude 2 * padLength number of columns from alignedLength calculation
               since padLength number of columns from the beginning and end of each row will be computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
            for (int c = 0; c < srcDescPtr->c; c++)
            {
                srcPtrRow[0] = srcPtrChannel;
                srcPtrRow[1] = srcPtrRow[0] + srcDescPtr->strides.hStride;
                srcPtrRow[2] = srcPtrRow[1] + srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    T *dstPtrTemp = dstPtrRow;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = 3;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, 3, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                    dstPtrTemp += padLength;
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
                    {
                        __m256 pRow[6], pDst[2];
                        rpp_load_filter_3x3_pln_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        pDst[0] = avx_p0;
                        pDst[1] = avx_p0;
                        for (int k = 0, filterIndex = 0, rowIndex = 0; k < 3; k++, filterIndex += 3, rowIndex += 2)
                        {
                            permute_blend_add_3x3<1, 3, 0, 1>(pDst[0], pRow[rowIndex], pRow[rowIndex + 1], &pFilter[filterIndex], pxMaskPln);
                            permute_blend_add_3x3<1, 3, 0, 1>(pDst[1], pRow[rowIndex + 1], avx_p0, &pFilter[filterIndex], pxMaskPln);
                        }
                        if constexpr (std::is_same<T, Rpp32f>::value)
                        {
                            pDst[0] = rpp_pixel_check_0to1_avx(pDst[0]);
                            pDst[1] = rpp_pixel_check_0to1_avx(pDst[1]);
                        }
                        rpp_store_filter_3x3_host(dstPtrTemp, pDst);
                        increment_row_ptrs(srcPtrTemp, 3, 14);
                        dstPtrTemp += 14;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        convolution_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, 3, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                        increment_row_ptrs(srcPtrTemp, 3, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, 3, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1: 0;
                T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                T *dstPtrTemp = dstPtrRow;

                Rpp32s rowKernelLoopLimit = 3;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, 3, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                dstPtrTemp += padLength * 3;
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 pRow[9], pDst[2];
                    rpp_load_filter_3x3_pkd_host(pRow, srcPtrTemp, rowKernelLoopLimit);

                    pDst[0] = avx_p0;
                    pDst[1] = avx_p0;
                    for (int k = 0, filterIndex = 0, rowIndex = 0; k < 3; k++, filterIndex += 3, rowIndex += 3)
                    {
                        permute_blend_add_3x3<7, 63, 0, 1>(pDst[0], pRow[rowIndex], pRow[rowIndex + 1], &pFilter[filterIndex], pxMaskPkd);
                        permute_blend_add_3x3<7, 63, 0, 1>(pDst[1], pRow[rowIndex + 1], pRow[rowIndex + 2], &pFilter[filterIndex], pxMaskPkd);
                    }
                    if constexpr (std::is_same<T, Rpp32f>::value)
                        {
                            pDst[0] = rpp_pixel_check_0to1_avx(pDst[0]);
                            pDst[1] = rpp_pixel_check_0to1_avx(pDst[1]);
                        }
                    increment_row_ptrs(srcPtrTemp, 3, 16);
                    rpp_store_filter_3x3_host(dstPtrTemp, pDst);
                    dstPtrTemp += 16;
                }
#endif
                vectorLoopCount += padLength * 3;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    convolution_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, 3,  padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                    increment_row_ptrs(srcPtrTemp, 3, 1);
                    dstPtrTemp++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next rows
                increment_row_ptrs(srcPtrRow, 3, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
            T *dstPtrChannels[3];
            for (int i = 0; i < 3; i++)
                dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1: 0;
                T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                Rpp32s rowKernelLoopLimit = 3;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, 3, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
#if __AVX2__
                // process remaining columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                {
                    __m256 pRow[9], pDst[2];
                    rpp_load_filter_3x3_pkd_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                    pDst[0] = avx_p0;
                    pDst[1] = avx_p0;
                    for (int k = 0, filterIndex = 0, rowIndex = 0; k < 3; k++, filterIndex += 3, rowIndex += 3)
                    {
                        permute_blend_add_3x3<7, 63, 0, 1>(pDst[0], pRow[rowIndex], pRow[rowIndex + 1], &pFilter[filterIndex], pxMaskPkd);
                        permute_blend_add_3x3<7, 63, 0, 1>(pDst[1], pRow[rowIndex + 1], pRow[rowIndex + 2], &pFilter[filterIndex], pxMaskPkd);
                    }
                    if constexpr (std::is_same<T, Rpp32f>::value)
                        {
                            pDst[0] = rpp_pixel_check_0to1_avx(pDst[0]);
                            pDst[1] = rpp_pixel_check_0to1_avx(pDst[1]);
                        }
                    __m128 pDstPln[3];
                    rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                    rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);
                    increment_row_ptrs(srcPtrTemp, 3, 12);
                    increment_row_ptrs(dstPtrTempChannels, 3, 4);
                }
#endif
                vectorLoopCount += padLength * 3;
                for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++)
                {
                    int channel = c % 3;
                    convolution_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, 3,  padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                    increment_row_ptrs(srcPtrTemp, 3, 1);
                    dstPtrTempChannels[channel]++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next rows
                increment_row_ptrs(srcPtrRow, 3, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            /* exclude (2 * padLength) number of columns from alignedLength calculation
                since padLength number of columns from the beginning and end of each row will be computed using raw c code */
            Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1: 0;
                T *srcPtrTemp[3][3] = {
                                            {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]},
                                            {srcPtrRow[0] + srcDescPtr->strides.cStride, srcPtrRow[1] + srcDescPtr->strides.cStride, srcPtrRow[2] + srcDescPtr->strides.cStride},
                                            {srcPtrRow[0] + 2 * srcDescPtr->strides.cStride, srcPtrRow[1] + 2 * srcDescPtr->strides.cStride, srcPtrRow[2] + 2 * srcDescPtr->strides.cStride}
                                            };

                T *dstPtrTemp = dstPtrRow;
                // get the number of rows needs to be loaded for the corresponding row
                Rpp32s rowKernelLoopLimit = 3;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                // process padLength number of columns in each row
                // left border pixels in image which does not have required pixels in 3x3 box, process them separately
                for (int k = 0; k < padLength; k++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        convolution_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, 3,  padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                        dstPtrTemp++;
                    }
                }
#if __AVX2__
                // process alignedLength number of columns in each row
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
                {
                    __m256 pResult[6];
                    for (int c = 0; c < 3; c++)
                    {
                        int channelStride = c * 2;
                        __m256 pRow[6];
                        rpp_load_filter_3x3_pln_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                        pResult[channelStride] = avx_p0;
                        pResult[channelStride + 1] = avx_p0;
                        for (int k = 0, filterIndex = 0, rowIndex = 0; k < 3; k++, filterIndex += 3, rowIndex += 2)
                        {
                            permute_blend_add_3x3<1, 3, 0, 1>(pResult[channelStride], pRow[rowIndex], pRow[rowIndex + 1], &pFilter[filterIndex], pxMaskPln);
                            permute_blend_add_3x3<1, 3, 0, 1>(pResult[channelStride + 1], pRow[rowIndex + 1], avx_p0, &pFilter[filterIndex], pxMaskPln);
                        }
                        if constexpr (std::is_same<T, Rpp32f>::value)
                        {
                            pResult[channelStride] = rpp_pixel_check_0to1_avx(pResult[channelStride]);
                            pResult[channelStride + 1] = rpp_pixel_check_0to1_avx(pResult[channelStride + 1]);
                        }
                        increment_row_ptrs(srcPtrTemp[c], 3, 14);
                    }
                    
                    // convert result from pln to pkd format and store in output buffer
                    if constexpr (std::is_same<T, Rpp32f>::value)
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, pResult);
                    else if constexpr (std::is_same<T, Rpp16f>::value)
                        rpp_simd_store(rpp_store48_f32pln3_to_f16pkd3_avx, dstPtrTemp, pResult);
                    else if constexpr (std::is_same<T, Rpp8u>::value)
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, pResult);
                    else if constexpr (std::is_same<T, Rpp8s>::value)
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, pResult);
                    dstPtrTemp += 42;
                }
#endif
                vectorLoopCount += padLength;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        convolution_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, 3,  padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                        increment_row_ptrs(srcPtrTemp[c], 3, 1);
                        dstPtrTemp++;
                    }
                }
                // for the first padLength rows, we need not increment the src row pointers to next rows
                increment_row_ptrs(srcPtrRow, 3, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }
    return RPP_SUCCESS;
}

template<typename T>
RppStatus emboss_generic_host_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *strength,
                                     Rpp32f *bias,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f *filterTensor = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + batchCount * 3 * 3;

        Rpp32u padLength = 3 / 2;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
        Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;

        Rpp32f biasParam = bias[batchCount];

        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        T *srcPtrRow[3], *dstPtrRow;
        for (int k = 0; k < 3; k++)
            srcPtrRow[k] = srcPtrChannel + k * srcDescPtr->strides.hStride;
        dstPtrRow = dstPtrChannel;
        create_emboss_kernel_host(filterTensor, strength[batchCount]);
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            for (int c = 0; c < srcDescPtr->c; c++)
            {
                srcPtrRow[0] = srcPtrChannel;
                for (int k = 1; k < 3; k++)
                    srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3];
                    for (int k = 0; k < 3; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    T *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = 3;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, 3, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                    dstPtrTemp += padLength;
                    vectorLoopCount += padLength;
                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        convolution_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, 3,  padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                        increment_row_ptrs(srcPtrTemp, 3, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, 3, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1: 0;
                T *srcPtrTemp[3];
                for (int k = 0; k < 3; k++)
                    srcPtrTemp[k] = srcPtrRow[k];
                T *dstPtrTemp = dstPtrRow;

                Rpp32s rowKernelLoopLimit = 3;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, 3, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                dstPtrTemp += padLength * 3;
                vectorLoopCount += padLength * 3;
                // process remaining columns in each row
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    convolution_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, 3,  padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                    increment_row_ptrs(srcPtrTemp, 3, 1);
                    dstPtrTemp++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next rows
                increment_row_ptrs(srcPtrRow, 3, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1: 0;
                T *srcPtrTemp[3][3];
                for (int c = 0; c < 3; c++)
                {
                    Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                    for (int k = 0; k < 3; k++)
                        srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                }
                T *dstPtrTemp = dstPtrRow;

                Rpp32s rowKernelLoopLimit = 3;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                // process padLength number of columns in each row
                for (int k = 0; k < padLength; k++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        convolution_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, 3,  padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                        dstPtrTemp++;
                    }
                }
                vectorLoopCount += padLength;
                // process remaining columns in each row
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    for (int c = 0; c < srcDescPtr->c; c++)
                    {
                        convolution_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, 3,  padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                        increment_row_ptrs(srcPtrTemp[c], 3, 1);
                        dstPtrTemp++;
                    }
                }
                // for the first padLength rows, we need not increment the src row pointers to next rows
                increment_row_ptrs(srcPtrRow, 3, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            T *dstPtrChannels[3];
            for (int c = 0; c < 3; c++)
                dstPtrChannels[c] = dstPtrChannel + c * dstDescPtr->strides.cStride;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1: 0;
                T *srcPtrTemp[3];
                for (int k = 0; k < 3; k++)
                    srcPtrTemp[k] = srcPtrRow[k];
                T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                Rpp32s rowKernelLoopLimit = 3;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, 3, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                vectorLoopCount += padLength * 3;
                // process remaining columns in each row
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    int channel = vectorLoopCount % 3;
                    convolution_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, 3, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                    increment_row_ptrs(srcPtrTemp, 3, 1);
                    dstPtrTempChannels[channel]++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next rows
                increment_row_ptrs(srcPtrRow, 3, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
            }
        }
    }
    return RPP_SUCCESS;
}

template RppStatus emboss_host_tensor<Rpp8u>(Rpp8u*,
                                             RpptDescPtr,
                                             Rpp8u*,
                                             RpptDescPtr,
                                             Rpp32f*,
                                             Rpp32f*,
                                             RpptROIPtr,
                                             RpptRoiType,
                                             RppLayoutParams,
                                             rpp::Handle&);
template RppStatus emboss_host_tensor<Rpp32f>(Rpp32f*,
                                              RpptDescPtr,
                                              Rpp32f*,
                                              RpptDescPtr,
                                              Rpp32f*,
                                              Rpp32f*,
                                              RpptROIPtr,
                                              RpptRoiType,
                                              RppLayoutParams,
                                              rpp::Handle&);
template RppStatus emboss_host_tensor<Rpp16f>(Rpp16f*,
                                              RpptDescPtr,
                                              Rpp16f*,
                                              RpptDescPtr,
                                              Rpp32f*,
                                              Rpp32f*,
                                              RpptROIPtr,
                                              RpptRoiType,
                                              RppLayoutParams,
                                              rpp::Handle&);
template RppStatus emboss_host_tensor<Rpp8s>(Rpp8s*,
                                             RpptDescPtr,
                                             Rpp8s*,
                                             RpptDescPtr,
                                             Rpp32f*,
                                             Rpp32f*,
                                             RpptROIPtr,
                                             RpptRoiType,
                                             RppLayoutParams,
                                             rpp::Handle&);
template RppStatus emboss_generic_host_tensor<Rpp8u>(Rpp8u*,
                                                     RpptDescPtr,
                                                     Rpp8u*,
                                                     RpptDescPtr,
                                                     Rpp32f*,
                                                     Rpp32f*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     RppLayoutParams,
                                                     rpp::Handle&);
template RppStatus emboss_generic_host_tensor<Rpp32f>(Rpp32f*,
                                                      RpptDescPtr,
                                                      Rpp32f*,
                                                      RpptDescPtr,
                                                      Rpp32f*,
                                                      Rpp32f*,
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      RppLayoutParams,
                                                      rpp::Handle&);
template RppStatus emboss_generic_host_tensor<Rpp16f>(Rpp16f*,
                                                      RpptDescPtr,
                                                      Rpp16f*,
                                                      RpptDescPtr,
                                                      Rpp32f*,
                                                      Rpp32f*,
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      RppLayoutParams,
                                                      rpp::Handle&);
template RppStatus emboss_generic_host_tensor<Rpp8s>(Rpp8s*,
                                                     RpptDescPtr,
                                                     Rpp8s*,
                                                     RpptDescPtr,
                                                     Rpp32f*,
                                                     Rpp32f*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     RppLayoutParams,
                                                     rpp::Handle&);
