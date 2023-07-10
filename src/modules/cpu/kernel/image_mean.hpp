#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus image_mean_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *imageSumArr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 8) * 8;
        Rpp32u vectorIncrement = 8;
        Rpp32f totalPixelsPerChannel = roi.xywhROI.roiWidth * roi.xywhROI.roiHeight;

        // Image Sum without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f mean = 0.0, sum = 0.0;
            Rpp32f sumAvx[8] = {0.0};

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow;
                srcPtrRow = srcPtrChannel;
#if __AVX2__
                __m256 psum = _mm256_setzero_ps();
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p1[1];
                        rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrTemp, p1);
                        compute_sum_8_host(p1, &psum);
                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        sum += (Rpp32f)(*srcPtrTemp);
                        srcPtrTemp++;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
#if __AVX2__
                rpp_simd_store(rpp_store8_f32_to_f32_avx, sumAvx, &psum);
                for(int i=0;i<4;i++)
                    sum += (sumAvx[i] + sumAvx[i + 4]);
#endif
            }
            mean = sum / totalPixelsPerChannel;
            imageSumArr[batchCount] = (Rpp32f)mean;
        }
    }

    return RPP_SUCCESS;
}