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

#include "rppdefs.h"
#include "rpp_cpu_common.hpp"
#include "rpp_cpu_common_geometric.hpp"
#include "rpp_cpu_common_interpolation.hpp"

inline void saturate_pixel(Rpp32f pixel, Rpp8u* dst)
{
    *dst = RPPPIXELCHECK(pixel);
}

inline void saturate_pixel(Rpp32f pixel, Rpp8s* dst)
{
    *dst = (Rpp8s)RPPPIXELCHECKI8(pixel - 128);
}

inline void saturate_pixel(Rpp32f pixel, Rpp32f* dst)
{
    *dst = (Rpp32f)pixel;
}

inline void saturate_pixel(Rpp32f pixel, Rpp16f* dst)
{
    *dst = (Rpp16f)pixel;
}

// Move with resample
inline void set_zeros(__m128 *pVecs, Rpp32s numVecs)
{
    for(int i = 0; i < numVecs; i++)
        pVecs[i] = xmm_p0;
}

inline void set_zeros_avx(__m256 *pVecs, Rpp32s numVecs)
{
    for(int i = 0; i < numVecs; i++)
        pVecs[i] = avx_p0;
}

// Perform resampling along the rows
// If only for resize, add above resize
/// Interpl
template <typename T>
inline void compute_separable_vertical_resample(T *inputPtr, Rpp32f *outputPtr, RpptDescPtr inputDescPtr, RpptDescPtr outputDescPtr,
                                                RpptImagePatch inputImgSize, RpptImagePatch outputImgSize, Rpp32s *index, Rpp32f *coeffs, GenericFilter &filter)
{

    static constexpr Rpp32s maxNumLanes = 16;                                  // Maximum number of pixels that can be present in a vector for U8 type
    static constexpr Rpp32s loadLanes = maxNumLanes / sizeof(T);
    static constexpr Rpp32s storeLanes = maxNumLanes / sizeof(Rpp32f);
    static constexpr Rpp32s numLanes = std::max(loadLanes, storeLanes);        // No of pixels that can be present in a vector wrt data type
    static constexpr Rpp32s numVecs = numLanes * sizeof(Rpp32f) / maxNumLanes; // No of float vectors required to process numLanes pixels

    Rpp32s inputHeightLimit = inputImgSize.height - 1;
    Rpp32s outPixelsPerIter = 4;

    // For PLN3 inputs/outputs
    if (inputDescPtr->c == 3 && inputDescPtr->layout == RpptLayout::NCHW)
    {
        T *inRowPtrR[filter.size];
        T *inRowPtrG[filter.size];
        T *inRowPtrB[filter.size];
        for (int outLocRow = 0; outLocRow < outputImgSize.height; outLocRow++)
        {
            Rpp32f *outRowPtrR = outputPtr + outLocRow * outputDescPtr->strides.hStride;
            Rpp32f *outRowPtrG = outRowPtrR + outputDescPtr->strides.cStride;
            Rpp32f *outRowPtrB = outRowPtrG + outputDescPtr->strides.cStride;
            Rpp32s k0 = outLocRow * filter.size;
            __m128 pCoeff[filter.size];

            // Determine the input row pointers and coefficients to be used for interpolation
            for (int k = 0; k < filter.size; k++)
            {
                Rpp32s inLocRow = index[outLocRow] + k;
                inLocRow = std::min(std::max(inLocRow, 0), inputHeightLimit);
                inRowPtrR[k] = inputPtr + inLocRow * inputDescPtr->strides.hStride;
                inRowPtrG[k] = inRowPtrR[k] + inputDescPtr->strides.cStride;
                inRowPtrB[k] = inRowPtrG[k] + inputDescPtr->strides.cStride;
                pCoeff[k] = _mm_set1_ps(coeffs[k0 + k]);    // Each row is associated with a single coeff
            }
            Rpp32s bufferLength = inputImgSize.width;
            Rpp32s alignedLength = bufferLength &~ (numLanes-1);
            Rpp32s outLocCol = 0;

            // Load the input pixels from filter.size rows
            // Multiply input vec from each row with it's correspondig coefficient
            // Add the results from filter.size rows to obtain the pixels of an output row
            for (; outLocCol + numLanes <= alignedLength; outLocCol += numLanes)
            {
                __m128 pTempR[numVecs], pTempG[numVecs], pTempB[numVecs];
                set_zeros(pTempR, numVecs);
                set_zeros(pTempG, numVecs);
                set_zeros(pTempB, numVecs);
                for (int k = 0; k < filter.size; k++)
                {
                    __m128 pInputR[numVecs], pInputG[numVecs], pInputB[numVecs];

                    // Load numLanes input pixels from each row
                    rpp_resize_load(inRowPtrR[k] + outLocCol, pInputR);
                    rpp_resize_load(inRowPtrG[k] + outLocCol, pInputG);
                    rpp_resize_load(inRowPtrB[k] + outLocCol, pInputB);
                    for (int v = 0; v < numVecs; v++)
                    {
                        pTempR[v] = _mm_fmadd_ps(pCoeff[k], pInputR[v], pTempR[v]);
                        pTempG[v] = _mm_fmadd_ps(pCoeff[k], pInputG[v], pTempG[v]);
                        pTempB[v] = _mm_fmadd_ps(pCoeff[k], pInputB[v], pTempB[v]);
                    }
                }
                for(int vec = 0, outStoreStride = 0; vec < numVecs; vec++, outStoreStride += outPixelsPerIter)    // Since 4 output pixels are stored per iteration
                {
                    rpp_simd_store(rpp_store4_f32_to_f32, outRowPtrR + outLocCol + outStoreStride, pTempR + vec);
                    rpp_simd_store(rpp_store4_f32_to_f32, outRowPtrG + outLocCol + outStoreStride, pTempG + vec);
                    rpp_simd_store(rpp_store4_f32_to_f32, outRowPtrB + outLocCol + outStoreStride, pTempB + vec);
                }
            }

            for (; outLocCol < bufferLength; outLocCol++)
            {
                Rpp32f tempR, tempG, tempB;
                tempR = tempG = tempB = 0;
                for (int k = 0; k < filter.size; k++)
                {
                    Rpp32f coefficient = coeffs[k0 + k];
                    tempR += (inRowPtrR[k][outLocCol] * coefficient);
                    tempG += (inRowPtrG[k][outLocCol] * coefficient);
                    tempB += (inRowPtrB[k][outLocCol] * coefficient);
                }
                outRowPtrR[outLocCol] = tempR;
                outRowPtrG[outLocCol] = tempG;
                outRowPtrB[outLocCol] = tempB;
            }
        }
    }
    // For PKD3 and PLN1 inputs/outputs
    else
    {
        T *inRowPtr[filter.size];
        for (int outLocRow = 0; outLocRow < outputImgSize.height; outLocRow++)
        {
            __m128 pCoeff[filter.size];
            Rpp32s k0 = outLocRow * filter.size;
            Rpp32f *outRowPtr = outputPtr + outLocRow * outputDescPtr->strides.hStride;

            // Determine the input row pointers and coefficients to be used for interpolation
            for (int k = 0; k < filter.size; k++)
            {
                Rpp32s inLocRow = index[outLocRow] + k;
                inLocRow = std::min(std::max(inLocRow, 0), inputHeightLimit);
                inRowPtr[k] = inputPtr + inLocRow * inputDescPtr->strides.hStride;
                pCoeff[k] = _mm_set1_ps(coeffs[k0 + k]);    // Each row is associated with a single coeff
            }
            Rpp32s bufferLength = inputImgSize.width * inputDescPtr->strides.wStride;
            Rpp32s alignedLength = bufferLength &~ (numLanes-1);
            Rpp32s outLocCol = 0;

            // Load the input pixels from filter.size rows
            // Multiply input vec from each row with it's correspondig coefficient
            // Add the results from filter.size rows to obtain the pixels of an output row
            for (; outLocCol + numLanes <= alignedLength; outLocCol += numLanes)
            {
                __m128 pTemp[numVecs];
                set_zeros(pTemp, numVecs);
                for (int k = 0; k < filter.size; k++)
                {
                    __m128 pInput[numVecs];
                    rpp_resize_load(inRowPtr[k] + outLocCol, pInput);   // Load numLanes input pixels from each row
                    for (int v = 0; v < numVecs; v++)
                        pTemp[v] = _mm_fmadd_ps(pInput[v], pCoeff[k], pTemp[v]);
                }
                for(int vec = 0, outStoreStride = 0; vec < numVecs; vec++, outStoreStride += outPixelsPerIter)     // Since 4 output pixels are stored per iteration
                    rpp_simd_store(rpp_store4_f32_to_f32, outRowPtr + outLocCol + outStoreStride, &pTemp[vec]);
            }

            for (; outLocCol < bufferLength; outLocCol++)
            {
                Rpp32f temp = 0;
                for (int k = 0; k < filter.size; k++)
                    temp += (inRowPtr[k][outLocCol] * coeffs[k0 + k]);
                outRowPtr[outLocCol] = temp;
            }
        }
    }
}

// Remap, resize, rotate, warp affine, persp - geometric for interpolation be part of separate headers
// Perform resampling along the columns
template <typename T>
inline void compute_separable_horizontal_resample(Rpp32f *inputPtr, T *outputPtr, RpptDescPtr inputDescPtr, RpptDescPtr outputDescPtr,
                        RpptImagePatch inputImgSize, RpptImagePatch outputImgSize, Rpp32s *index, Rpp32f *coeffs, GenericFilter &filter)
{
    static constexpr Rpp32s maxNumLanes = 16;                                   // Maximum number of pixels that can be present in a vector
    static constexpr Rpp32s numLanes = maxNumLanes / sizeof(T);                 // No of pixels that can be present in a vector wrt data type
    static constexpr Rpp32s numVecs = numLanes * sizeof(Rpp32f) / maxNumLanes;  // No of float vectors required to process numLanes pixels
    Rpp32s numOutPixels, filterKernelStride;
    numOutPixels = filterKernelStride = 4;
    Rpp32s filterKernelSizeOverStride = filter.size % filterKernelStride;
    Rpp32s filterKernelRadiusWStrided = (Rpp32s)(filter.radius) * inputDescPtr->strides.wStride;

    Rpp32s inputWidthLimit = (inputImgSize.width - 1) * inputDescPtr->strides.wStride;
    __m128i pxInputWidthLimit = _mm_set1_epi32(inputWidthLimit);

    // For PLN3 inputs
    if(inputDescPtr->c == 3 && inputDescPtr->layout == RpptLayout::NCHW)
    {
        for (int outLocRow = 0; outLocRow < outputImgSize.height; outLocRow++)
        {
            T *outRowPtrR = outputPtr + outLocRow * outputDescPtr->strides.hStride;
            T *outRowPtrG = outRowPtrR + outputDescPtr->strides.cStride;
            T *outRowPtrB = outRowPtrG + outputDescPtr->strides.cStride;
            Rpp32f *inRowPtrR = inputPtr + outLocRow * inputDescPtr->strides.hStride;
            Rpp32f *inRowPtrG = inRowPtrR + inputDescPtr->strides.cStride;
            Rpp32f *inRowPtrB = inRowPtrG + inputDescPtr->strides.cStride;
            Rpp32s bufferLength = outputImgSize.width;
            Rpp32s alignedLength = bufferLength &~ (numLanes-1);
            __m128 pFirstValR = _mm_set1_ps(inRowPtrR[0]);
            __m128 pFirstValG = _mm_set1_ps(inRowPtrG[0]);
            __m128 pFirstValB = _mm_set1_ps(inRowPtrB[0]);
            bool breakLoop = false;
            Rpp32s outLocCol = 0;

            // Load filter.size consecutive pixels from a location in the row
            // Multiply with corresponding coeffs and add together to obtain the output pixel
            for (; outLocCol + numLanes <= alignedLength; outLocCol += numLanes)
            {
                __m128 pOutputChannel[numVecs * 3];
                set_zeros(pOutputChannel, numVecs * 3);
                __m128 *pOutputR = pOutputChannel;
                __m128 *pOutputG = pOutputChannel + numVecs;
                __m128 *pOutputB = pOutputChannel + (numVecs * 2);
                for(int vec = 0, x = outLocCol; vec < numVecs; vec++, x += numOutPixels)
                {
                    Rpp32s coeffIdx = (x * filter.size);
                    if(index[x] < 0)
                    {
                        __m128i pxIdx[numOutPixels];
                        pxIdx[0] = _mm_set1_epi32(index[x]);
                        pxIdx[1] = _mm_set1_epi32(index[x + 1]);
                        pxIdx[2] = _mm_set1_epi32(index[x + 2]);
                        pxIdx[3] = _mm_set1_epi32(index[x + 3]);
                        for(int k = 0; k < filter.size; k += filterKernelStride)
                        {
                            // Generate mask to determine the negative indices in the iteration
                            __m128i pxNegativeIndexMask[numOutPixels];
                            __m128i pxKernelIdx = _mm_set1_epi32(k);
                            __m128 pInputR[numOutPixels], pInputG[numOutPixels], pInputB[numOutPixels], pCoeffs[numOutPixels];
                            Rpp32s kernelAdd = (k + filterKernelStride) > filter.size ? filterKernelSizeOverStride : filterKernelStride;
                            set_zeros(pInputR, numOutPixels);
                            set_zeros(pInputG, numOutPixels);
                            set_zeros(pInputB, numOutPixels);
                            set_zeros(pCoeffs, numOutPixels);

                            for(int l = 0; l < numOutPixels; l++)
                            {
                                pxNegativeIndexMask[l] = _mm_cmplt_epi32(_mm_add_epi32(_mm_add_epi32(pxIdx[l], pxKernelIdx), xmm_pDstLocInit), xmm_px0);    // Generate mask to determine the negative indices in the iteration
                                Rpp32s srcx = index[x + l] + k;

                                // Load filterKernelStride(4) consecutive pixels
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtrR + srcx, pInputR + l);
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtrG + srcx, pInputG + l);
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtrB + srcx, pInputB + l);
                                pCoeffs[l] = _mm_loadu_ps(&(coeffs[coeffIdx + ((l + k) * 4)]));        // Load coefficients

                                // If negative index is present replace the input pixel value with first value in the row
                                pInputR[l] = _mm_blendv_ps(pInputR[l], pFirstValR, pxNegativeIndexMask[l]);
                                pInputG[l] = _mm_blendv_ps(pInputG[l], pFirstValG, pxNegativeIndexMask[l]);
                                pInputB[l] = _mm_blendv_ps(pInputB[l], pFirstValB, pxNegativeIndexMask[l]);
                            }

                            // Perform transpose operation to arrange input pixels from different output locations in each vector
                            _MM_TRANSPOSE4_PS(pInputR[0], pInputR[1], pInputR[2], pInputR[3]);
                            _MM_TRANSPOSE4_PS(pInputG[0], pInputG[1], pInputG[2], pInputG[3]);
                            _MM_TRANSPOSE4_PS(pInputB[0], pInputB[1], pInputB[2], pInputB[3]);
                            for (int l = 0; l < kernelAdd; l++)
                            {
                                pOutputR[vec] = _mm_fmadd_ps(pCoeffs[l], pInputR[l], pOutputR[vec]);
                                pOutputG[vec] = _mm_fmadd_ps(pCoeffs[l], pInputG[l], pOutputG[vec]);
                                pOutputB[vec] = _mm_fmadd_ps(pCoeffs[l], pInputB[l], pOutputB[vec]);
                            }
                        }
                    }
                    else if(index[x + 3] >= (inputWidthLimit - filterKernelRadiusWStrided))    // If the index value exceeds the limit, break the loop
                    {
                        breakLoop = true;
                        break;
                    }
                    else
                    {
                        // Considers a 4x1 window for computation each time
                        for(int k = 0; k < filter.size; k += filterKernelStride)
                        {
                            __m128 pInputR[numOutPixels], pInputG[numOutPixels], pInputB[numOutPixels];
                            __m128 pCoeffs[numOutPixels];
                            Rpp32s kernelAdd = (k + filterKernelStride) > filter.size ? filterKernelSizeOverStride : filterKernelStride;
                            for (int l = 0; l < numOutPixels; l++)
                            {
                                pInputR[l] = pInputG[l] = pInputB[l] = pCoeffs[l] = xmm_p0;
                                pCoeffs[l] = _mm_loadu_ps(&(coeffs[coeffIdx + ((l + k) * 4)])); // Load coefficients
                                Rpp32s srcx = index[x + l] + k;
                                srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                                // Load filterKernelStride(4) consecutive pixels
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtrR + srcx, pInputR + l);
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtrG + srcx, pInputG + l);
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtrB + srcx, pInputB + l);
                            }

                            // Perform transpose operation to arrange input pixels from different output locations in each vector
                            _MM_TRANSPOSE4_PS(pInputR[0], pInputR[1], pInputR[2], pInputR[3]);
                            _MM_TRANSPOSE4_PS(pInputG[0], pInputG[1], pInputG[2], pInputG[3]);
                            _MM_TRANSPOSE4_PS(pInputB[0], pInputB[1], pInputB[2], pInputB[3]);
                            for (int l = 0; l < kernelAdd; l++)
                            {
                                pOutputR[vec] = _mm_fmadd_ps(pCoeffs[l], pInputR[l], pOutputR[vec]);
                                pOutputG[vec] = _mm_fmadd_ps(pCoeffs[l], pInputG[l], pOutputG[vec]);
                                pOutputB[vec] = _mm_fmadd_ps(pCoeffs[l], pInputB[l], pOutputB[vec]);
                            }
                        }
                    }
                }
                if(breakLoop) break;
                Rpp32s xStride = outLocCol * outputDescPtr->strides.wStride;
                if(outputDescPtr->layout == RpptLayout::NCHW)       // For PLN3 outputs
                    rpp_resize_store_pln3(outRowPtrR + xStride, outRowPtrG + xStride, outRowPtrB + xStride, pOutputChannel);
                else if(outputDescPtr->layout == RpptLayout::NHWC)  // For PKD3 outputs
                    rpp_resize_store_pkd3(outRowPtrR + xStride, pOutputChannel);
            }
            Rpp32s k0 = 0;
            for (; outLocCol < outputImgSize.width; outLocCol++)
            {
                Rpp32s x0 = index[outLocCol];
                k0 = outLocCol % 4 == 0 ? outLocCol * filter.size : k0 + 1; // Since coeffs are stored in continuously for 4 dst locations
                Rpp32f sumR, sumG, sumB;
                sumR = sumG = sumB = 0;
                for (int k = 0; k < filter.size; k++)
                {
                    Rpp32s srcx = x0 + k;
                    srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                    Rpp32s kPos = (k * 4);      // Since coeffs are stored in continuously for 4 dst locations
                    sumR += (coeffs[k0 + kPos] * inRowPtrR[srcx]);
                    sumG += (coeffs[k0 + kPos] * inRowPtrG[srcx]);
                    sumB += (coeffs[k0 + kPos] * inRowPtrB[srcx]);
                }
                Rpp32s xStride = outLocCol * outputDescPtr->strides.wStride;
                saturate_pixel(sumR, outRowPtrR + xStride);
                saturate_pixel(sumG, outRowPtrG + xStride);
                saturate_pixel(sumB, outRowPtrB + xStride);
            }
        }
    }
    // For PKD3 inputs
    else if(inputDescPtr->c == 3 && inputDescPtr->layout == RpptLayout::NHWC)
    {
        for (int outLocRow = 0; outLocRow < outputImgSize.height; outLocRow++)
        {
            T *outRowPtrR = outputPtr + outLocRow * outputDescPtr->strides.hStride;
            T *outRowPtrG = outRowPtrR + outputDescPtr->strides.cStride;
            T *outRowPtrB = outRowPtrG + outputDescPtr->strides.cStride;
            Rpp32f *inRowPtr = inputPtr + outLocRow * inputDescPtr->strides.hStride;
            Rpp32s bufferLength = outputImgSize.width;
            Rpp32s alignedLength = bufferLength &~ (numLanes-1);
            Rpp32s outLocCol = 0;

            // Load filter.size consecutive pixels from a location in the row
            // Multiply with corresponding coeffs and add together to obtain the output pixel
            for (; outLocCol + numLanes <= alignedLength; outLocCol += numLanes)
            {
                __m128 pOutputChannel[numVecs * 3];
                set_zeros(pOutputChannel, numVecs * 3);
                __m128 *pOutputR = pOutputChannel;
                __m128 *pOutputG = pOutputChannel + numVecs;
                __m128 *pOutputB = pOutputChannel + (numVecs * 2);
                for(int vec = 0, x = outLocCol; vec < numVecs; vec++, x += numOutPixels)   // 4 dst pixels processed per iteration
                {
                    Rpp32s coeffIdx = (x * filter.size);
                    for(int k = 0, kStrided = 0; k < filter.size; k ++, kStrided = k * 3)
                    {
                        __m128 pInput[numOutPixels];
                        __m128 pCoeffs = _mm_loadu_ps(&(coeffs[coeffIdx + (k * numOutPixels)]));
                        for (int l = 0; l < numOutPixels; l++)
                        {
                            Rpp32s srcx = index[x + l] + kStrided;
                            srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                            rpp_simd_load(rpp_load4_f32_to_f32, &inRowPtr[srcx], &pInput[l]);   // Load RGB pixel from a src location
                        }

                        // Perform transpose operation to arrange input pixels by R,G and B separately in each vector
                        _MM_TRANSPOSE4_PS(pInput[0], pInput[1], pInput[2], pInput[3]);
                        pOutputR[vec] = _mm_fmadd_ps(pCoeffs, pInput[0], pOutputR[vec]);
                        pOutputG[vec] = _mm_fmadd_ps(pCoeffs, pInput[1], pOutputG[vec]);
                        pOutputB[vec] = _mm_fmadd_ps(pCoeffs, pInput[2], pOutputB[vec]);
                    }
                }

                Rpp32s xStride = outLocCol * outputDescPtr->strides.wStride;
                if(outputDescPtr->layout == RpptLayout::NCHW)       // For PLN3 outputs
                    rpp_resize_store_pln3(outRowPtrR + xStride, outRowPtrG + xStride, outRowPtrB + xStride, pOutputChannel);
                else if(outputDescPtr->layout == RpptLayout::NHWC)  // For PKD3 outputs
                    rpp_resize_store_pkd3(outRowPtrR + xStride, pOutputChannel);
            }
            Rpp32s k0 = 0;
            for (; outLocCol < outputImgSize.width; outLocCol++)
            {
                Rpp32s x0 = index[outLocCol];
                k0 = outLocCol % 4 == 0 ? outLocCol * filter.size : k0 + 1;  // Since coeffs are stored in continuously for 4 dst locations
                Rpp32f sumR, sumG, sumB;
                sumR = sumG = sumB = 0;
                for (int k = 0; k < filter.size; k++)
                {
                    Rpp32s srcx = x0 + (k * 3);
                    srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                    Rpp32s kPos = (k * 4);      // Since coeffs are stored in continuously for 4 dst locations
                    sumR += (coeffs[k0 + kPos] * inRowPtr[srcx]);
                    sumG += (coeffs[k0 + kPos] * inRowPtr[srcx + 1]);
                    sumB += (coeffs[k0 + kPos] * inRowPtr[srcx + 2]);
                }
                Rpp32s xStride = outLocCol * outputDescPtr->strides.wStride;
                saturate_pixel(sumR, outRowPtrR + xStride);
                saturate_pixel(sumG, outRowPtrG + xStride);
                saturate_pixel(sumB, outRowPtrB + xStride);
            }
        }
    }
    else
    {
        for (int outLocRow = 0; outLocRow < outputImgSize.height; outLocRow++)
        {
            T *out_row = outputPtr + outLocRow * outputDescPtr->strides.hStride;
            Rpp32f *inRowPtr = inputPtr + outLocRow * inputDescPtr->strides.hStride;
            Rpp32s bufferLength = outputImgSize.width;
            Rpp32s alignedLength = bufferLength &~ (numLanes-1);
            __m128 pFirstVal = _mm_set1_ps(inRowPtr[0]);
            bool breakLoop = false;
            Rpp32s outLocCol = 0;

            // Load filter.size consecutive pixels from a location in the row
            // Multiply with corresponding coeffs and add together to obtain the output pixel
            for (; outLocCol + numLanes <= alignedLength; outLocCol += numLanes)
            {
                __m128 pOutput[numVecs];
                set_zeros(pOutput, numVecs);
                for(int vec = 0, x = outLocCol; vec < numVecs; vec++, x += numOutPixels)
                {
                    Rpp32s coeffIdx = (x * filter.size);
                    if(index[x] < 0)
                    {
                        __m128i pxIdx[numOutPixels];
                        pxIdx[0] = _mm_set1_epi32(index[x]);
                        pxIdx[1] = _mm_set1_epi32(index[x + 1]);
                        pxIdx[2] = _mm_set1_epi32(index[x + 2]);
                        pxIdx[3] = _mm_set1_epi32(index[x + 3]);
                        for(int k = 0; k < filter.size; k += filterKernelStride)
                        {
                            __m128i pxNegativeIndexMask[numOutPixels];
                            __m128i pxKernelIdx = _mm_set1_epi32(k);
                            __m128 pInput[numOutPixels], pCoeffs[numOutPixels];
                            Rpp32s kernelAdd = (k + filterKernelStride) > filter.size ? filterKernelSizeOverStride : filterKernelStride;
                            set_zeros(pInput, numOutPixels);
                            set_zeros(pCoeffs, numOutPixels);
                            for(int l = 0; l < numOutPixels; l++)
                            {
                                pxNegativeIndexMask[l] = _mm_cmplt_epi32(_mm_add_epi32(_mm_add_epi32(pxIdx[l], pxKernelIdx), xmm_pDstLocInit), xmm_px0);    // Generate mask to determine the negative indices in the iteration
                                rpp_simd_load(rpp_load4_f32_to_f32, &inRowPtr[index[x + l] + k], &pInput[l]);   // Load filterKernelStride(4) consecutive pixels
                                pCoeffs[l] = _mm_loadu_ps(&(coeffs[coeffIdx + ((l + k) * 4)]));                 // Load coefficients
                                pInput[l] = _mm_blendv_ps(pInput[l], pFirstVal, pxNegativeIndexMask[l]);        // If negative index is present replace the pixel value with first value in the row
                            }
                            _MM_TRANSPOSE4_PS(pInput[0], pInput[1], pInput[2], pInput[3]);  // Perform transpose operation to arrange input pixels from different output locations in each vector
                            for (int l = 0; l < kernelAdd; l++)
                                pOutput[vec] = _mm_fmadd_ps(pCoeffs[l], pInput[l], pOutput[vec]);
                        }
                    }
                    else if(index[x + 3] >= (inputWidthLimit - filterKernelRadiusWStrided))   // If the index value exceeds the limit, break the loop
                    {
                        breakLoop = true;
                        break;
                    }
                    else
                    {
                        for(int k = 0; k < filter.size; k += filterKernelStride)
                        {
                            __m128 pInput[numOutPixels], pCoeffs[numOutPixels];
                            Rpp32s kernelAdd = (k + filterKernelStride) > filter.size ? filterKernelSizeOverStride : filterKernelStride;
                            for (int l = 0; l < numOutPixels; l++)
                            {
                                pInput[l] = pCoeffs[l] = xmm_p0;
                                pCoeffs[l] = _mm_loadu_ps(&(coeffs[coeffIdx + ((l + k) * 4)]));     // Load coefficients
                                Rpp32s srcx = index[x + l] + k;
                                srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                                rpp_simd_load(rpp_load4_f32_to_f32, inRowPtr + srcx, pInput + l);   // Load filterKernelStride(4) consecutive pixels
                            }
                            _MM_TRANSPOSE4_PS(pInput[0], pInput[1], pInput[2], pInput[3]);  // Perform transpose operation to arrange input pixels from different output locations in each vector
                            for (int l = 0; l < kernelAdd; l++)
                                pOutput[vec] = _mm_fmadd_ps(pCoeffs[l], pInput[l], pOutput[vec]);
                        }
                    }
                }
                if(breakLoop) break;
                rpp_resize_store(out_row + outLocCol, pOutput);
            }
            Rpp32s k0 = 0;
            for (; outLocCol < bufferLength; outLocCol++)
            {
                Rpp32s x0 = index[outLocCol];
                k0 = outLocCol % 4 == 0 ? outLocCol * filter.size : k0 + 1;  // Since coeffs are stored in continuously for 4 dst locations
                Rpp32f sum = 0;
                for (int k = 0; k < filter.size; k++)
                {
                    Rpp32s srcx = x0 + k;
                    srcx = std::min(std::max(srcx, 0), inputWidthLimit);
                    sum += (coeffs[k0 + (k * 4)] * inRowPtr[srcx]);
                }
                saturate_pixel(sum, out_row + outLocCol);
            }
        }
    }
}

inline void compute_bicubic_coefficient(Rpp32f weight, Rpp32f &coeff)
{
    Rpp32f x = fabsf(weight);
    coeff = (x >= 2) ? 0 : ((x > 1) ? (x * x * (-0.5f * x + 2.5f) - 4.0f * x + 2.0f) : (x * x * (1.5f * x - 2.5f) + 1.0f));
}

inline Rpp32f sinc(Rpp32f x)
{
    x *= M_PI;
    return (std::abs(x) < 1e-5f) ? (1.0f - x * x * ONE_OVER_6) : std::sin(x) / x;
}

inline void compute_lanczos3_coefficient(Rpp32f weight, Rpp32f &coeff)
{
    coeff = fabs(weight) >= 3 ? 0.0f : (sinc(weight) * sinc(weight * 0.333333f));
}

inline void compute_gaussian_coefficient(Rpp32f weight, Rpp32f &coeff)
{
    coeff = expf(weight * weight * -4.0f);
}

inline void compute_triangular_coefficient(Rpp32f weight, Rpp32f &coeff)
{
    coeff = 1 - std::fabs(weight);
    coeff = coeff < 0 ? 0 : coeff;
}

inline void compute_coefficient(RpptInterpolationType interpolationType, Rpp32f weight, Rpp32f &coeff)
{
    switch (interpolationType)
    {
    case RpptInterpolationType::BICUBIC:
    {
        compute_bicubic_coefficient(weight, coeff);
        break;
    }
    case RpptInterpolationType::LANCZOS:
    {
        compute_lanczos3_coefficient(weight, coeff);
        break;
    }
    case RpptInterpolationType::GAUSSIAN:
    {
        compute_gaussian_coefficient(weight, coeff);
        break;
    }
    case RpptInterpolationType::TRIANGULAR:
    {
        compute_triangular_coefficient(weight, coeff);
        break;
    }
    default:
        break;
    }
}

// Computes the row coefficients for separable resampling
inline void compute_row_coefficients(RpptInterpolationType interpolationType, GenericFilter &filter , Rpp32f weight, Rpp32f *coeffs, Rpp32u srcStride = 1)
{
    Rpp32f sum = 0;
    weight = weight - filter.radius;
    for(int k = 0; k < filter.size; k++)
    {
        compute_coefficient(interpolationType, (weight + k) * filter.scale, coeffs[k]);
        sum += coeffs[k];
    }
    if(sum)
    {
        sum = 1 / sum;
        for(int k = 0; k < filter.size; k++)
            coeffs[k] = coeffs[k] * sum;
    }
}

// Computes the column coefficients for separable resampling
inline void compute_col_coefficients(RpptInterpolationType interpolationType, GenericFilter &filter, Rpp32f weight, Rpp32f *coeffs, Rpp32u srcStride = 1)
{
    Rpp32f sum = 0;
    weight = weight - filter.radius;

    // The coefficients are computed for 4 dst locations and stored consecutively for ease of access
    for(int k = 0, kPos = 0; k < filter.size; k++, kPos += 4)
    {
        compute_coefficient(interpolationType, (weight + k) * filter.scale, coeffs[kPos]);
        sum += coeffs[kPos];
    }
    if(sum)
    {
        sum = 1 / sum;
        for(int k = 0, kPos = 0; k < filter.size; k++, kPos += 4)
            coeffs[kPos] = coeffs[kPos] * sum;
    }
}


/************* NEAREST NEIGHBOR INTERPOLATION *************/

template <typename T>
RppStatus resize_separable_host_tensor(T *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       T *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f * tempPtr,
                                       RpptDescPtr tempDescPtr,
                                       RpptImagePatchPtr dstImgSize,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams srcLayoutParams,
                                       RpptInterpolationType interpolationType,
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

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        GenericFilter vFilter(interpolationType, roi.xywhROI.roiHeight, dstImgSize[batchCount].height, hRatio);    // Initialize vertical resampling filter
        GenericFilter hFilter(interpolationType, roi.xywhROI.roiWidth, dstImgSize[batchCount].width, wRatio);      // Initialize Horizontal resampling filter
        Rpp32f hOffset = (hRatio - 1) * 0.5f - vFilter.radius;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - hFilter.radius;

        Rpp32s rowIndex[dstImgSize[batchCount].height], colIndex[dstImgSize[batchCount].width];
        Rpp32f rowCoeffs[dstImgSize[batchCount].height * vFilter.size];
        Rpp32f colCoeffs[((dstImgSize[batchCount].width + 3) & ~3) * hFilter.size]; // Buffer size is made a multiple of 4 inorder to allocate sufficient memory for Horizontal coefficients

        // Pre-compute row index and coefficients
        for(int indexCount = 0, coeffCount = 0; indexCount < dstImgSize[batchCount].height; indexCount++, coeffCount += vFilter.size)
        {
            Rpp32f weightParam;
            compute_resize_src_loc(indexCount, hRatio, rowIndex[indexCount], weightParam, hOffset);
            compute_row_coefficients(interpolationType, vFilter, weightParam, &rowCoeffs[coeffCount]);
        }
        // Pre-compute col index and coefficients
        for(int indexCount = 0, coeffCount = 0; indexCount < dstImgSize[batchCount].width; indexCount++)
        {
            Rpp32f weightParam;
            compute_resize_src_loc(indexCount, wRatio, colIndex[indexCount], weightParam, wOffset, srcDescPtr->strides.wStride);
            coeffCount = (indexCount % 4 == 0) ? (indexCount * hFilter.size) : coeffCount + 1;
            compute_col_coefficients(interpolationType, hFilter, weightParam, &colCoeffs[coeffCount], srcDescPtr->strides.wStride);
        }

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32f * tempPtrImage = tempPtr + batchCount * tempDescPtr->strides.nStride;
        srcPtrImage = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);

        RpptImagePatch srcImgSize;
        srcImgSize.width = roi.xywhROI.roiWidth;
        srcImgSize.height = roi.xywhROI.roiHeight;

        // The intermediate result from Vertical Resampling will have the src width and dst height
        RpptImagePatch tempImgSize;
        tempImgSize.width = roi.xywhROI.roiWidth;
        tempImgSize.height = dstImgSize[batchCount].height;

        compute_separable_vertical_resample(srcPtrImage, tempPtrImage, srcDescPtr, tempDescPtr, srcImgSize, tempImgSize, rowIndex, rowCoeffs, vFilter);
        compute_separable_horizontal_resample(tempPtrImage, dstPtrImage, tempDescPtr, dstDescPtr, tempImgSize, dstImgSize[batchCount], colIndex, colCoeffs, hFilter);
    }

    return RPP_SUCCESS;
}

RppStatus resize_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams srcLayoutParams,
                                      rpp::Handle& handle);

RppStatus resize_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptImagePatchPtr dstImgSize,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams srcLayoutParams,
                                        rpp::Handle& handle);

RppStatus resize_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8s *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams srcLayoutParams,
                                      rpp::Handle& handle);

RppStatus resize_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp16f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptImagePatchPtr dstImgSize,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams srcLayoutParams,
                                        rpp::Handle& handle);

/************* BILINEAR INTERPOLATION *************/

RppStatus resize_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8u *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSize,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            RppLayoutParams srcLayoutParams,
                                            rpp::Handle& handle);

RppStatus resize_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              RpptImagePatchPtr dstImgSize,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams srcLayoutParams,
                                              rpp::Handle& handle);

RppStatus resize_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp16f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              RpptImagePatchPtr dstImgSize,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams srcLayoutParams,
                                              rpp::Handle& handle);

RppStatus resize_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8s *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSize,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            RppLayoutParams srcLayoutParams,
                                            rpp::Handle& handle);