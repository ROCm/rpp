#ifndef HOST_ADVANCED_AUGMENTATIONS_H
#define HOST_ADVANCED_AUGMENTATIONS_H

#include "cpu/rpp_cpu_simd.hpp"
#include <cpu/rpp_cpu_common.hpp>

/**************** water ***************/

template <typename T>
RppStatus water_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                         Rpp32f *batch_ampl_x, Rpp32f *batch_ampl_y, 
                         Rpp32f *batch_freq_x, Rpp32f *batch_freq_y, 
                         Rpp32f *batch_phase_x, Rpp32f *batch_phase_y, 
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            
            Rpp32f ampl_x = batch_ampl_x[batchCount];
            Rpp32f ampl_y = batch_ampl_y[batchCount];
            Rpp32f freq_x = batch_freq_x[batchCount];
            Rpp32f freq_y = batch_freq_y[batchCount];
            Rpp32f phase_x = batch_phase_x[batchCount];
            Rpp32f phase_y = batch_phase_y[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            T *srcPtrImageR, *srcPtrImageG, *srcPtrImageB;
            srcPtrImageR = srcPtrImage;
            srcPtrImageG = srcPtrImageR + imageDimMax;
            srcPtrImageB = srcPtrImageG + imageDimMax;

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrImage + (i * elementsInRowMax);
                dstPtrTempG = dstPtrTempR + imageDimMax;
                dstPtrTempB = dstPtrTempG + imageDimMax;

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128i const zero = _mm_setzero_si128();
                __m128 pI = _mm_set1_ps((Rpp32f)i);
                __m128 pJ, pWaterI, pWaterJ;
                __m128 pAmplX = _mm_set1_ps(ampl_x);
                __m128 pAmplY = _mm_set1_ps(ampl_y);
                __m128 pFreqX = _mm_set1_ps(freq_x);
                __m128 pFreqY = _mm_set1_ps(freq_y);
                __m128 pPhaseX = _mm_set1_ps(phase_x);
                __m128 pPhaseY = _mm_set1_ps(phase_y);
                __m128 p0, p1, p2, p3;

                Rpp32f waterI[4], waterJ[4];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pJ = _mm_setr_ps((Rpp32f)(vectorLoopCount), (Rpp32f)(vectorLoopCount + 1), (Rpp32f)(vectorLoopCount + 2), (Rpp32f)(vectorLoopCount + 3));
                    p0 = _mm_mul_ps(pFreqX, pI);
                    p0 = _mm_add_ps(p0, pPhaseX);
                    sincos_ps(p0, &p1, &p2);
                    p1 = _mm_mul_ps(p1, pAmplX);
                    pWaterJ = _mm_add_ps(pJ, p1);
                    p0 = _mm_mul_ps(pFreqY, pJ);
                    p0 = _mm_add_ps(p0, pPhaseY);
                    sincos_ps(p0, &p1, &p2);
                    p2 = _mm_mul_ps(p2, pAmplY);
                    pWaterI = _mm_add_ps(pI, p2);

                    _mm_storeu_ps(waterI, pWaterI);
                    _mm_storeu_ps(waterJ, pWaterJ);

                    for (int count = 0; count < 4; count++)
                    {
                        Rpp32u waterIint, waterJint;
                        waterIint = (Rpp32u) RPPPRANGECHECK(waterI[count], 0, batch_srcSize[batchCount].height - 1);
                        waterJint = (Rpp32u) RPPPRANGECHECK(waterJ[count], 0, batch_srcSize[batchCount].width - 1);
                        
                        *dstPtrTempR = *(srcPtrImageR + (waterIint * elementsInRowMax) + waterJint);
                        dstPtrTempR++;
                        
                        if (channel == 3)
                        {
                            *dstPtrTempG = *(srcPtrImageG + (waterIint * elementsInRowMax) + waterJint);
                            *dstPtrTempB = *(srcPtrImageB + (waterIint * elementsInRowMax) + waterJint);
                            dstPtrTempG++;
                            dstPtrTempB++;
                        }
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f locI = (Rpp32f) i;
                    Rpp32f locJ = (Rpp32f) vectorLoopCount;

                    Rpp32f waterLocJ = locJ + ampl_x * sin((freq_x * locI) + phase_x);
                    Rpp32f waterLocI = locI + ampl_y * cos((freq_y * locJ) + phase_y);

                    Rpp32u waterLocIint, waterLocJint;
                    waterLocIint = (Rpp32u) RPPPRANGECHECK(waterLocI, 0, batch_srcSize[batchCount].height);
                    waterLocJint = (Rpp32u) RPPPRANGECHECK(waterLocJ, 0, batch_srcSize[batchCount].width);
                    
                    *dstPtrTempR = *(srcPtrImageR + (waterLocIint * elementsInRowMax) + waterLocJint);
                    dstPtrTempR++;
                    
                    if (channel == 3)
                    {
                        *dstPtrTempG = *(srcPtrImageG + (waterLocIint * elementsInRowMax) + waterLocJint);
                        *dstPtrTempB = *(srcPtrImageB + (waterLocIint * elementsInRowMax) + waterLocJint);
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);
                
                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            
            Rpp32f ampl_x = batch_ampl_x[batchCount];
            Rpp32f ampl_y = batch_ampl_y[batchCount];
            Rpp32f freq_x = batch_freq_x[batchCount];
            Rpp32f freq_y = batch_freq_y[batchCount];
            Rpp32f phase_x = batch_phase_x[batchCount];
            Rpp32f phase_y = batch_phase_y[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *dstPtrTemp;
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128i const zero = _mm_setzero_si128();
                __m128 pI = _mm_set1_ps((Rpp32f)i);
                __m128 pJ, pWaterI, pWaterJ;
                __m128 pAmplX = _mm_set1_ps(ampl_x);
                __m128 pAmplY = _mm_set1_ps(ampl_y);
                __m128 pFreqX = _mm_set1_ps(freq_x);
                __m128 pFreqY = _mm_set1_ps(freq_y);
                __m128 pPhaseX = _mm_set1_ps(phase_x);
                __m128 pPhaseY = _mm_set1_ps(phase_y);
                __m128 p0, p1, p2, p3;

                Rpp32f waterI[4], waterJ[4];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pJ = _mm_setr_ps((Rpp32f)(vectorLoopCount), (Rpp32f)(vectorLoopCount + 1), (Rpp32f)(vectorLoopCount + 2), (Rpp32f)(vectorLoopCount + 3));
                    p0 = _mm_mul_ps(pFreqX, pI);
                    p0 = _mm_add_ps(p0, pPhaseX);
                    sincos_ps(p0, &p1, &p2);
                    p1 = _mm_mul_ps(p1, pAmplX);
                    pWaterJ = _mm_add_ps(pJ, p1);
                    p0 = _mm_mul_ps(pFreqY, pJ);
                    p0 = _mm_add_ps(p0, pPhaseY);
                    sincos_ps(p0, &p1, &p2);
                    p2 = _mm_mul_ps(p2, pAmplY);
                    pWaterI = _mm_add_ps(pI, p2);

                    _mm_storeu_ps(waterI, pWaterI);
                    _mm_storeu_ps(waterJ, pWaterJ);

                    for (int count = 0; count < 4; count++)
                    {
                        Rpp32u waterIint, waterJint;
                        waterIint = (Rpp32u) RPPPRANGECHECK(waterI[count], 0, batch_srcSize[batchCount].height - 1);
                        waterJint = (Rpp32u) RPPPRANGECHECK(waterJ[count], 0, batch_srcSize[batchCount].width - 1);
                        
                        for (int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp = *(srcPtrImage + (waterIint * elementsInRowMax) + (waterJint * channel) + c);
                            dstPtrTemp++;
                        }
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f locI = (Rpp32f) i;
                    Rpp32f locJ = (Rpp32f) vectorLoopCount;

                    Rpp32f waterLocJ = locJ + ampl_x * sin((freq_x * locI) + phase_x);
                    Rpp32f waterLocI = locI + ampl_y * cos((freq_y * locJ) + phase_y);

                    Rpp32u waterLocIint, waterLocJint;
                    waterLocIint = (Rpp32u) RPPPRANGECHECK(waterLocI, 0, batch_srcSize[batchCount].height);
                    waterLocJint = (Rpp32u) RPPPRANGECHECK(waterLocJ, 0, batch_srcSize[batchCount].width);
                    
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = *(srcPtrImage + (waterLocIint * elementsInRowMax) + (waterLocJint * channel) + c);
                        dstPtrTemp++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);
                
                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** non_linear_blend ***************/

template <typename T>
RppStatus non_linear_blend_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                         Rpp32f *batch_std_dev,
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            
            Rpp32f std_dev = batch_std_dev[batchCount];
            Rpp32f multiplier = - 1.0 / (2.0 * std_dev * std_dev);
            
            T *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;
            Rpp32s subtrahendI = (Rpp32s) (batch_srcSizeMax[batchCount].height >> 1);
            Rpp32s subtrahendJ = (Rpp32s) (batch_srcSizeMax[batchCount].width >> 1);

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32f locI = (Rpp32f) (i - subtrahendI);
                Rpp32f relativeGaussian[4];

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128i const zero = _mm_setzero_si128();
                __m128 pMultiplier = _mm_set1_ps(multiplier);
                __m128 pZero = _mm_set1_ps(0.0);
                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pExpI = _mm_set1_ps(locI * locI * multiplier);
                __m128 pJ, pExpJ, pRelGaussian;
                __m128i px0, px1, px2, px3;
                __m128 p0, p1, p2, p3, p4, p5, p6, p7;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pJ = _mm_setr_ps(
                        (Rpp32f)(vectorLoopCount - subtrahendJ), 
                        (Rpp32f)(vectorLoopCount + 1 - subtrahendJ), 
                        (Rpp32f)(vectorLoopCount + 2 - subtrahendJ), 
                        (Rpp32f)(vectorLoopCount + 3 - subtrahendJ)
                    );
                    pExpJ = _mm_mul_ps(pJ, pJ);
                    pExpJ = _mm_mul_ps(pExpJ, pMultiplier);
                    pRelGaussian = _mm_add_ps(pExpI, pExpJ);
                    pRelGaussian = fast_exp_sse(pRelGaussian);
                    
                    for (int c = 0; c < channel; c++)
                    {
                        T *srcPtr1TempChannel, *srcPtr2TempChannel, *dstPtrTempChannel;
                        srcPtr1TempChannel = srcPtr1Temp + (c * imageDimMax);
                        srcPtr2TempChannel = srcPtr2Temp + (c * imageDimMax);
                        dstPtrTempChannel = dstPtrTemp + (c * imageDimMax);

                        px0 =  _mm_loadu_si128((__m128i *)srcPtr1TempChannel);
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3

                        px0 =  _mm_loadu_si128((__m128i *)srcPtr2TempChannel);
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3

                        p0 = _mm_mul_ps(pRelGaussian, p0);
                        p1 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian), p1);
                        p0 = _mm_add_ps(p0, p1);

                        px0 = _mm_cvtps_epi32(p0);
                        px1 = _mm_cvtps_epi32(pZero);
                        px2 = _mm_cvtps_epi32(pZero);
                        px3 = _mm_cvtps_epi32(pZero);

                        px0 = _mm_packus_epi32(px0, px1);
                        px1 = _mm_packus_epi32(px2, px3);

                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTempChannel, px0);
                    }

                    srcPtr1Temp += 4;
                    srcPtr2Temp += 4;
                    dstPtrTemp += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s locI = i - subtrahendI;
                    Rpp32s locJ = vectorLoopCount - subtrahendJ;

                    Rpp32f gaussianValue = gaussian_2d_relative(locI, locJ, std_dev);

                    for (int c = 0; c < channel; c++)
                    {
                        T *srcPtr1TempChannel, *srcPtr2TempChannel, *dstPtrTempChannel;
                        srcPtr1TempChannel = srcPtr1Temp + (c * imageDimMax);
                        srcPtr2TempChannel = srcPtr2Temp + (c * imageDimMax);
                        dstPtrTempChannel = dstPtrTemp + (c * imageDimMax);

                        *dstPtrTempChannel = (Rpp8u) RPPPIXELCHECK((gaussianValue * ((Rpp32f) *srcPtr1TempChannel)) + ((1 - gaussianValue) * ((Rpp32f) *srcPtr2TempChannel)));
                    }

                    srcPtr1Temp++;
                    srcPtr2Temp++;
                    dstPtrTemp++;
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);
                
                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            
            Rpp32f std_dev = batch_std_dev[batchCount];
            Rpp32f multiplier = - 1.0 / (2.0 * std_dev * std_dev);
            
            T *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32s subtrahendI = (Rpp32s) (batch_srcSizeMax[batchCount].height >> 1);
            Rpp32s subtrahendJ = (Rpp32s) (batch_srcSizeMax[batchCount].width >> 1);

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32f locI = (Rpp32f) (i - subtrahendI);
                Rpp32f relativeGaussian[4];

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128i const zero = _mm_setzero_si128();
                __m128 pMultiplier = _mm_set1_ps(multiplier);
                __m128 pZero = _mm_set1_ps(0.0);
                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pExpI = _mm_set1_ps(locI * locI * multiplier);
                __m128 pJ, pExpJ, pRelGaussian;
                __m128i px0, px1, px2, px3;
                __m128 p0, p1, p2, p3, p4, p5, p6, p7;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pJ = _mm_setr_ps(
                        (Rpp32f)(vectorLoopCount - subtrahendJ), 
                        (Rpp32f)(vectorLoopCount + 1 - subtrahendJ), 
                        (Rpp32f)(vectorLoopCount + 2 - subtrahendJ), 
                        (Rpp32f)(vectorLoopCount + 3 - subtrahendJ)
                    );
                    pExpJ = _mm_mul_ps(pJ, pJ);
                    pExpJ = _mm_mul_ps(pExpJ, pMultiplier);
                    pRelGaussian = _mm_add_ps(pExpI, pExpJ);
                    pRelGaussian = fast_exp_sse(pRelGaussian);
                    
                    _mm_storeu_ps(relativeGaussian, pRelGaussian);

                    __m128 pRelGaussian0 = _mm_setr_ps(relativeGaussian[0], relativeGaussian[0], relativeGaussian[0], relativeGaussian[1]);
                    __m128 pRelGaussian1 = _mm_setr_ps(relativeGaussian[1], relativeGaussian[1], relativeGaussian[2], relativeGaussian[2]);
                    __m128 pRelGaussian2 = _mm_setr_ps(relativeGaussian[2], relativeGaussian[3], relativeGaussian[3], relativeGaussian[3]);

                    px0 =  _mm_loadu_si128((__m128i *)srcPtr1Temp);
                    px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                    px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                    p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                    p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                    p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                    px0 =  _mm_loadu_si128((__m128i *)srcPtr2Temp);
                    px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                    px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    p4 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                    p5 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                    p6 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                    p7 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                    p0 = _mm_mul_ps(pRelGaussian0, p0);
                    p4 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian0), p4);
                    p0 = _mm_add_ps(p0, p4);

                    p1 = _mm_mul_ps(pRelGaussian1, p1);
                    p5 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian1), p5);
                    p1 = _mm_add_ps(p1, p5);

                    p2 = _mm_mul_ps(pRelGaussian2, p2);
                    p6 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian2), p6);
                    p2 = _mm_add_ps(p2, p6);

                    px0 = _mm_cvtps_epi32(p0);
                    px1 = _mm_cvtps_epi32(p1);
                    px2 = _mm_cvtps_epi32(p2);
                    px3 = _mm_cvtps_epi32(pZero);

                    px0 = _mm_packus_epi32(px0, px1);
                    px1 = _mm_packus_epi32(px2, px3);

                    px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                    srcPtr1Temp += 12;
                    srcPtr2Temp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s locI = i - subtrahendI;
                    Rpp32s locJ = vectorLoopCount - subtrahendJ;

                    Rpp32f gaussianValue = gaussian_2d_relative(locI, locJ, std_dev);

                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((gaussianValue * ((Rpp32f) *srcPtr1Temp)) + ((1 - gaussianValue) * ((Rpp32f) *srcPtr2Temp)));
                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);
                
                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}
#endif