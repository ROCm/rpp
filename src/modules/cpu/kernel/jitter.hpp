#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline void compute_jitter_param_initialize_4_host_sse(Rpp32u &kernelSize, __m128 &pJitterParam)
{
    pJitterParam = _mm_set1_ps(kernelSize);
}

void printVec8(__m128i vec)
{
    int i;
    //char array[16];
    int array[4];
    _mm_store_si128( (__m128i *) array, vec);
    /*for(i=0; i<16; i++){
        printf("%3u , " , array[i]);
    }*/
    for(i=0; i<4; i++){
        printf("%5d , " , array[i]);
    }
    printf("\n");

}
RppStatus jitter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u *kernelSizeTensor,
                                       RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    __m128 pSrcChannel = _mm_set1_ps(srcDescPtr->c);
    __m128 pSrcStride = _mm_set1_ps(srcDescPtr->strides.hStride);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize-1)/2;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - bound;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        //Rpp32u bufferLength = widthLimit * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = widthLimit & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;
        RpptXorwowStateBoxMuller xorwowState;
        Rpp32s jitterSrcLocArray[4] = {0}; 

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pJitterParam;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_jitter_param_initialize_4_host_sse(kernelSize, pJitterParam);
        __m128 pDstLoc = xmm_pDstLocInit;
        //__m128 roix = _mm_set1_ps(roi.xywhROI.xy.x);
        //__m128 roiy = _mm_set1_ps(roi.xywhROI.xy.y);

        // Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < heightLimit; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                /*for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m128 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    compute_jitter_48_host(p, );  // jitter
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_jitter_48_host(p, );  // jitter
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }*/

                for (; vectorLoopCount < widthLimit; vectorLoopCount += 3)
                {

                    Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    Rpp16u nhx = randomNumberFloat * kernelSize;
                    randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    Rpp16u nhy = randomNumberFloat * kernelSize;
                    int rowLoc = (roi.xywhROI.xy.y + i + nhy) * srcDescPtr->strides.hStride;
                    int colLoc = (roi.xywhROI.xy.x + vectorLoopCount + nhx) * srcDescPtr->c;
                    int rowcolLoc = rowLoc + colLoc;
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[0] + rowcolLoc));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[1] + rowcolLoc));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[2] + rowcolLoc));

                    //srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                //srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < heightLimit; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                /*for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m128 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_jitter_48_host(p, );  // jitter
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_jitter_48_host(p, );  // jitter
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }*/

                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    Rpp16u nhx = randomNumberFloat * kernelSize;
                    randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    Rpp16u nhy = randomNumberFloat * kernelSize;
                    int rowLoc = (roi.xywhROI.xy.y + i + nhy) * srcDescPtr->strides.hStride;
                    int colLoc = (roi.xywhROI.xy.x + vectorLoopCount + nhx) * srcDescPtr->c;
                    int rowcolLoc = rowLoc + colLoc;
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((Rpp32f) *(srcPtrTempR + rowcolLoc));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((Rpp32f) *(srcPtrTempG + rowcolLoc));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((Rpp32f) *(srcPtrTempB + rowcolLoc));

                    //srcPtrTempR++;
                    //srcPtrTempG++;
                    //srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                //srcPtrRowR += srcDescPtr->strides.hStride;
                //srcPtrRowG += srcDescPtr->strides.hStride;
                //srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Jitter without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;


            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m128 randomNumberFloatx, randomNumberFloaty, nhx, nhy, x, y, j;
                __m128 pdstLocRow = _mm_set1_ps(dstLocRow);
                //y = _mm_add_ps(roiy,pdstLocRow);
                y = _mm_set1_ps(dstLocRow);

                printf("%d\n",dstLocRow);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    j = _mm_set1_ps(vectorLoopCount);
                    //x = _mm_add_ps(roix,_mm_add_ps(j,pDstLoc));
                    x = _mm_add_ps(j,pDstLoc);
                    randomNumberFloatx = rpp_host_rng_xorwow_4_f32_sse(pxXorwowStateX, &pxXorwowStateCounter);
                    nhx = _mm_mul_ps(randomNumberFloatx,pJitterParam);
                    randomNumberFloaty = rpp_host_rng_xorwow_4_f32_sse(pxXorwowStateX, &pxXorwowStateCounter);
                    nhy = _mm_mul_ps(randomNumberFloaty,pJitterParam);


                    __m128 rowTemp = _mm_add_ps(y,nhy);
                    __m128 colTemp = _mm_add_ps(x,nhx);
                    printf("\nrowTemp\n");
                    printVec8(_mm_cvtps_epi32(rowTemp));
                    printf("\ncolTemp\n");
                    printVec8(_mm_cvtps_epi32(colTemp));
                    
                    compute_jitter_loc(rowTemp, colTemp, jitterSrcLocArray, pSrcStride, pSrcChannel);
                    printf("\ncompute loc done\n");
                    rpp_simd_load(rpp_nn_load_u8pkd3, srcPtrChannel, jitterSrcLocArray, pRow);
                    printf("simd load done\n");
                    rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTemp, pRow);
                    printf("simd store done\n");
                    dstPtrTemp += vectorIncrement;
                }

                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    Rpp32u nhx = randomNumberFloat * kernelSize;
                    randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    Rpp32u nhy = randomNumberFloat * kernelSize;
                    int rowLoc = (roi.xywhROI.xy.y + dstLocRow + nhy) * srcDescPtr->strides.hStride;
                    int colLoc = (roi.xywhROI.xy.x + vectorLoopCount + nhx) * srcDescPtr->c;
                    int rowcolLoc = rowLoc + colLoc;
                    *dstPtrTemp++ = (Rpp8u) RPPPIXELCHECK((Rpp32f) *(srcPtrChannel + rowcolLoc));
                    *dstPtrTemp++ = (Rpp8u) RPPPIXELCHECK((Rpp32f) *(srcPtrChannel + rowcolLoc + 1));
                    *dstPtrTemp++ = (Rpp8u) RPPPIXELCHECK((Rpp32f) *(srcPtrChannel + rowcolLoc + 2));
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
            /*int i=0;
            for(; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;                    
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    wyhash16_x = seedVal;
                    Rpp16u nhx = rand_range16(kernelSize);
                    Rpp16u nhy = rand_range16(kernelSize);

                    Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    Rpp16u nhx = randomNumberFloat * kernelSize;
                    randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    Rpp16u nhy = randomNumberFloat * kernelSize;


                    int rowLoc = (roi.xywhROI.xy.y + i + nhy) * srcDescPtr->strides.hStride;
                    int colLoc = (roi.xywhROI.xy.x + vectorLoopCount + nhx) * srcDescPtr->c;
                    int rowcolLoc = rowLoc + colLoc;
                    *dstPtrTemp++ = (Rpp8u) RPPPIXELCHECK((Rpp32f) *(srcPtrTemp + rowcolLoc));
                    *dstPtrTemp++ = (Rpp8u) RPPPIXELCHECK((Rpp32f) *(srcPtrTemp + rowcolLoc + 1));
                    *dstPtrTemp++ = (Rpp8u) RPPPIXELCHECK((Rpp32f) *(srcPtrTemp + rowcolLoc + 2));
                }
                *dstPtrTemp+=(3*bufferLengthMax-bufferLength);
                dstPtrRow += dstDescPtr->strides.hStride;
                //srcPtrTempRow += srcDescPtr->strides.hStride;
            }
            for(;i<roi.xywhROI.roiHeight;i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;                    
                srcPtrTemp = srcPtrTempRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for(; vectorLoopCount<bufferLengthMax; vectorLoopCount++)
                {
                    int inc = vectorLoopCount*srcDescPtr->c;
                    *dstPtrTemp++ = *(srcPtrTemp + inc);
                    *dstPtrTemp++ = *(srcPtrTemp + inc + 1);
                    *dstPtrTemp++ = *(srcPtrTemp + inc + 2);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
                srcPtrTempRow += srcDescPtr->strides.hStride;
            }
            for(; i < roi.xywhROI.roiHeight; i++)
            {
                dstPtrRow += roi.xywhROI.roiWidth;
            }*/
        }
        else
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < heightLimit; i++)
                {
                    Rpp8u *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    /*for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
#if __AVX2__
                        __m128 p[2];

                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_jitter_16_host(p, );  // jitter adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores
#else
                        __m128 p[4];

                        rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);    // simd loads
                        compute_jitter_16_host(p, );  // jitter adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);    // simd stores
#endif
                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }*/

                    for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                    {
                        Rpp32f randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                        Rpp16u nhx = randomNumberFloat * kernelSize;
                        randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                        Rpp16u nhy = randomNumberFloat * kernelSize;
                        int rowLoc = (roi.xywhROI.xy.y + i + nhy) * srcDescPtr->strides.hStride;
                        int colLoc = (roi.xywhROI.xy.x + vectorLoopCount + nhx) * srcDescPtr->c;
                        int rowcolLoc = rowLoc + colLoc;
                        *dstPtrTemp++ = (Rpp8u) RPPPIXELCHECK((Rpp32f) *(srcPtrTemp + rowcolLoc));
                        *dstPtrTemp++ = (Rpp8u) RPPPIXELCHECK((Rpp32f) *(srcPtrTemp + rowcolLoc + 1));
                        *dstPtrTemp++ = (Rpp8u) RPPPIXELCHECK((Rpp32f) *(srcPtrTemp + rowcolLoc + 2));
                    }

                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus jitter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32u *kernelSizeTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize-1);
        Rpp32u widthLimit = roi.xywhROI.roiWidth - bound;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = widthLimit * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        int seedVal;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;
#endif

        // Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < heightLimit; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                /*for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m128 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_jitter_24_host(p, );  // jitter adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[3];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_jitter_12_host(p, );  // jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }*/
                seedVal = rand() % (65536);
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    wyhash16_x = seedVal;
                    Rpp16u nhx = rand_range16(kernelSize);
                    Rpp16u nhy = rand_range16(kernelSize);
                    *dstPtrTempR = (srcPtrTemp[0] + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));
                    *dstPtrTempG = (srcPtrTemp[1] + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));
                    *dstPtrTempB = (srcPtrTemp[2] + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < heightLimit; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                /*for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m128 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_jitter_24_host(p, );  // jitter adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_jitter_12_host(p, );  // jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }*/

                seedVal = rand() % (65536);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    wyhash16_x = seedVal;
                    Rpp16u nhx = rand_range16(kernelSize);
                    Rpp16u nhy = rand_range16(kernelSize);
                    dstPtrTemp[0] = *(srcPtrTempR + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));
                    dstPtrTemp[1] = *(srcPtrTempG + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));
                    dstPtrTemp[2] = *(srcPtrTempB + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Jitter without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < heightLimit; i++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    /*for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
#if __AVX2__
                        __m128 p[1];

                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_jitter_8_host(p, );  // jitter adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
#else
                        __m128 p[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp, p);    // simd loads
                        compute_jitter_4_host(p, );  // jitter adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, p);    // simd stores
#endif
                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }*/

                    seedVal = rand() % (65536);
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        wyhash16_x = seedVal;
                        Rpp16u nhx = rand_range16(kernelSize);
                        Rpp16u nhy = rand_range16(kernelSize);
                        *dstPtrTemp = *(srcPtrTemp + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus jitter_f16_f16_host_tensor(Rpp16f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp16f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32u *kernelSizeTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize-1);
        Rpp32u widthLimit = roi.xywhROI.roiWidth - bound;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = widthLimit * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        int seedVal;

#if __AVX2__

#else

#endif

        // Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < heightLimit; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                /*for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];

                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

#if __AVX2__
                    __m128 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    compute_jitter_24_host(p, );  // jitter adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
#else
                    __m128 p[3];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    compute_jitter_12_host(p, );  // jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
#endif

                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }*/

                seedVal = rand() % (65536);
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    wyhash16_x = seedVal;
                    Rpp16u nhx = rand_range16(kernelSize);
                    Rpp16u nhy = rand_range16(kernelSize);
                    *dstPtrTempR = (srcPtrTemp[0] + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));
                    *dstPtrTempG = (srcPtrTemp[1] + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));
                    *dstPtrTempB = (srcPtrTemp[2] + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < heightLimit; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                /*for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }
#if __AVX2__
                    __m128 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_jitter_24_host(p, );  // jitter adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_jitter_12_host(p, );  // jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }*/

                seedVal = rand() % (65536);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    wyhash16_x = seedVal;
                    Rpp16u nhx = rand_range16(kernelSize);
                    Rpp16u nhy = rand_range16(kernelSize);
                    dstPtrTemp[0] = *(srcPtrTempR + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));
                    dstPtrTemp[1] = *(srcPtrTempG + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));
                    dstPtrTemp[2] = *(srcPtrTempB + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Jitter without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            //Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < heightLimit; i++)
                {
                    Rpp16f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    /*for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];

                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        {
                            srcPtrTemp_ps[cnt] = (Rpp16f) srcPtrTemp[cnt];
                        }
#if __AVX2__
                        __m128 p[1];

                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp_ps, p);    // simd loads
                        compute_jitter_8_host(p, );  // jitter adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, p);    // simd stores
#else
                        __m128 p[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp_ps, p);    // simd loads
                        compute_jitter_4_host(p, );  // jitter adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp_ps, p);    // simd stores
#endif

                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        {
                            dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                        }

                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }*/

                    seedVal = rand() % (65536);
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        wyhash16_x = seedVal;
                        Rpp16u nhx = rand_range16(kernelSize);
                        Rpp16u nhy = rand_range16(kernelSize);
                        *dstPtrTemp = *(srcPtrTemp + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus jitter_i8_i8_host_tensor(Rpp8s *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8s *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u *kernelSizeTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize-1);
        Rpp32u widthLimit = roi.xywhROI.roiWidth - bound;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = widthLimit * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        int seedVal;

#if __AVX2__

#else

#endif

        // Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < heightLimit; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                /*for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m128 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    compute_jitter_48_host(p, );  // jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_jitter_48_host(p, );  // jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }*/

                seedVal = rand() % (65536);
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    wyhash16_x = seedVal;
                    Rpp16u nhx = rand_range16(kernelSize);
                    Rpp16u nhy = rand_range16(kernelSize);
                    *dstPtrTempR = (srcPtrTemp[0] + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));   /* Add subtract 128?*/
                    *dstPtrTempG = (srcPtrTemp[1] + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));
                    *dstPtrTempB = (srcPtrTemp[2] + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < heightLimit; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                /*for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m128 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_jitter_48_host(p, );  // jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_jitter_48_host(p, );  // jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }*/

                seedVal = rand() % (65536);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    wyhash16_x = seedVal;
                    Rpp16u nhx = rand_range16(kernelSize);
                    Rpp16u nhy = rand_range16(kernelSize);
                    dstPtrTemp[0] = *(srcPtrTempR + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));
                    dstPtrTemp[1] = *(srcPtrTempG + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));
                    dstPtrTemp[2] = *(srcPtrTempB + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Jitter without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < heightLimit; i++)
                {
                    Rpp8s *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    /*for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
#if __AVX2__
                        __m128 p[2];

                        rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_jitter_16_host(p, );  // jitter adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores
#else
                        __m128 p[4];

                        rpp_simd_load(rpp_load16_i8_to_f32, srcPtrTemp, p);    // simd loads
                        compute_jitter_16_host(p, );  // jitter adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);    // simd stores
#endif

                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }*/

                    seedVal = rand() % (65536);
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        wyhash16_x = seedVal;
                        Rpp16u nhx = rand_range16(kernelSize);
                        Rpp16u nhy = rand_range16(kernelSize);
                        *dstPtrTemp = *(srcPtrTemp + (roi.xywhROI.xy.y + nhy)*srcDescPtr->strides.hStride + (roi.xywhROI.xy.x + nhx));

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}
