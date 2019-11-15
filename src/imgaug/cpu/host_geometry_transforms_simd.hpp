//
// Created by svcbuild on 11/15/19.
//

#ifndef AMD_RPP_HOST_GEOMETRY_TRANSFORMS_SIMD_HPP
#define AMD_RPP_HOST_GEOMETRY_TRANSFORMS_SIMD_HPP
#include <cpu/rpp_cpu_common.hpp>

#if ENABLE_SIMD_INTRINSICS

/**************** Flip ***************/

template <>
RppStatus flip_host(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
                    RppiAxis flipAxis,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp8u *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
       int alignW = (srcSize.width + 31) & ~31;
       if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {
            for (int c = 0; c < channel; c++)
            {
                srcPtrTemp = srcPtr + ((c + 1) * srcSize.height * srcSize.width) - srcSize.width;
                for (int i = 0; i < srcSize.height; i++)
                {
                    int j=0;
#if __AVX2__
                    __m256i p0;
                    for (; j < alignW; j+=32) {
                        p0 = _mm256_loadu_si256((const __m256i*)srcPtrTemp);
                        _mm256_storeu_si256((__m256i *)dstPtrTemp, p0);
                        srcPtrTemp +=32, dstPtrTemp += 32;
                    }
#endif
                    for (; j < srcSize.width; j++)
                    {
                        *dstPtrTemp++ = *srcPtrTemp++;
                    }
                    srcPtrTemp = srcPtrTemp - (2 * srcSize.width);
                }
            }
        }
        else if (flipAxis == RPPI_VERTICAL_AXIS)
        {
            for (int c = 0; c < channel; c++)
            {
                srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width) + srcSize.width;
                for (int i = 0; i < srcSize.height; i++)
                {
                    int j=0;
#if __AVX2__
                    __m256i p0;
                    for (; j < alignW; j+=32) {
                        srcPtrTemp -= 32;
                        p0 = _mm256_loadu_si256((const __m256i*)srcPtrTemp);
                        _mm256_storeu_si256((__m256i *)dstPtrTemp, p0);
                        dstPtrTemp += 32;
                    }
#endif
                    for (; j < srcSize.width; j++)
                    {
                        srcPtrTemp--;
                        *dstPtrTemp++ = *srcPtrTemp;
                    }
                    srcPtrTemp = srcPtrTemp + (2 * srcSize.width);
                }
            }
        }
        else if (flipAxis == RPPI_BOTH_AXIS)
        {
            for (int c = 0; c < channel; c++)
            {
                srcPtrTemp = srcPtr + ((c + 1) * srcSize.height * srcSize.width);
                for (int i = 0; i < srcSize.height; i++)
                {
                    int j=0;
#if __AVX2__
                    __m256i p0;
                    for (; j < alignW; j+=32) {
                        srcPtrTemp -= 32;
                        p0 = _mm256_loadu_si256((const __m256i*)srcPtrTemp);
                        _mm256_storeu_si256((__m256i *)dstPtrTemp, p0);
                        dstPtrTemp += 32;
                    }

#endif
                    for (; j < srcSize.width; j++)
                    {
                        srcPtrTemp--;
                        *dstPtrTemp++ = *srcPtrTemp;
                        dstPtrTemp++;
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        //todo:: if channel is 3, call reference
        int alignW = srcSize.width / 10; // proces 10 pixels in one load
        if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {
            srcPtrTemp = srcPtr + (channel * ((srcSize.height-1) * srcSize.width));
            for (int i = 0; i < srcSize.height; i++)
            {
                Rpp8u* pSrc = srcPtrTemp;
                int j=0;
#if __AVX2__
                __m256i p0;
                // copy multiple of 3 bytes for 3 channels
                for (; j < alignW; j+=10) {
                    p0 = _mm256_loadu_si256((const __m256i*)pSrc);
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p0);
                    pSrc +=30, dstPtrTemp += 30;
                }

                for (; j < srcSize.width; j++)
                {
                    *dstPtrTemp++ = pSrc[0];
                    *dstPtrTemp++ = pSrc[1];
                    *dstPtrTemp++ = pSrc[2];
                    pSrc += 3;
                }
                srcPtrTemp -= srcSize.width*channel;
#endif
                for (; j < srcSize.width; j++)
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = *srcPtrTemp;
                        dstPtrTemp++;
                        srcPtrTemp++;
                    }
                }


                srcPtrTemp = srcPtrTemp - (channel * (2 * srcSize.width));
            }
        }
        else if (flipAxis == RPPI_VERTICAL_AXIS)
        {
            srcPtrTemp = srcPtr + channel * srcSize.width;
            for (int i = 0; i < srcSize.height; i++)
            {
                Rpp8u* pSrc = srcPtrTemp;
                int j=0;
#if __AVX2__
                __m256i p0;
                // copy multiple of 3 bytes for 3 channels
                for (; j < alignW; j+=10) {
                    pSrc -= 30;
                    p0 = _mm256_loadu_si256((const __m256i*)pSrc);
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p0);
                    dstPtrTemp += 30;
                }
#endif
                for (; j < srcSize.width; j++)
                {
                    pSrc -= 3;
                    *dstPtrTemp++ = pSrc[0];
                    *dstPtrTemp++ = pSrc[1];
                    *dstPtrTemp++ = pSrc[2];
                }
                srcPtrTemp += srcSize.width*channel;
            }
        }
        else if (flipAxis == RPPI_BOTH_AXIS)
        {
            srcPtrTemp = srcPtr + (channel * (srcSize.height * srcSize.width));
            for (int i = 0; i < srcSize.height; i++)
            {
                Rpp8u* pSrc = srcPtrTemp;
                int j=0;
#if __AVX2__
                __m256i p0;
                // copy multiple of 3 bytes for 3 channels
                for (; j < alignW; j+= 10) {
                    pSrc -= 30;
                    p0 = _mm256_loadu_si256((const __m256i*)pSrc);
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p0);
                    dstPtrTemp += 30;
                }
#endif
                for (; j < srcSize.width; j++)
                {
                    pSrc -= 3;
                    *dstPtrTemp++ = pSrc[0];
                    *dstPtrTemp++ = pSrc[1];
                    *dstPtrTemp++ = pSrc[2];
                }
                srcPtrTemp -= srcSize.width*channel;
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** Rotate ***************/


template <>
RppStatus rotate_host(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiSize dstSize,
                           Rpp32f angleDeg,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f angleRad = -RAD(angleDeg);
    Rpp32f rotate[4] = {0};
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = -sin(angleRad);
    rotate[3] = cos(angleRad);

    Rpp8u *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f divisor = (rotate[1] * rotate[2]) - (rotate[0] * rotate[3]);
    Rpp32f srcLocationRow, srcLocationColumn, srcLocationRowTerm1, srcLocationColumnTerm1, pixel;
    Rpp32s srcLocationRowFloor, srcLocationColumnFloor;

    Rpp32f halfSrcHeight = srcSize.height / 2;
    Rpp32f halfSrcWidth = srcSize.width / 2;
    Rpp32f halfDstHeight = dstSize.height / 2;
    Rpp32f halfDstWidth = dstSize.width / 2;
    Rpp32f halfHeightDiff = halfSrcHeight - halfDstHeight;
    Rpp32f halfWidthDiff = halfSrcWidth - halfDstWidth;

    Rpp32f srcLocationRowParameter = (rotate[0] * halfSrcHeight) + (rotate[1] * halfSrcWidth) - halfSrcHeight + halfHeightDiff;
    Rpp32f srcLocationColumnParameter = (rotate[2] * halfSrcHeight) + (rotate[3] * halfSrcWidth) - halfSrcWidth + halfWidthDiff;
    Rpp32f srcLocationRowParameter2 = (-rotate[3] * (Rpp32s)srcLocationRowParameter) + (rotate[1] * (Rpp32s)srcLocationColumnParameter);
    Rpp32f srcLocationColumnParameter2 = (rotate[2] * (Rpp32s)srcLocationRowParameter) + (-rotate[0] * (Rpp32s)srcLocationColumnParameter);
    Rpp32f div_mul_factor = 1.f/divisor;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {
                srcLocationRowTerm1 = -rotate[3] * i;
                srcLocationColumnTerm1 = rotate[2] * i;
                for (int j = 0; j < dstSize.width; j++)
                {
                    srcLocationRow = (srcLocationRowTerm1 + (rotate[1] * j) + srcLocationRowParameter2) * div_mul_factor;
                    srcLocationColumn = (srcLocationColumnTerm1 + (-rotate[0] * j) + srcLocationColumnParameter2) * div_mul_factor;

                    if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                    else
                    {
                        srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                        srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                        Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (Rpp8u) round(pixel);
                        dstPtrTemp ++;
                    }
                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRowTerm1 = -rotate[3] * i;
            srcLocationColumnTerm1 = rotate[2] * i;
            int j = 0;
#if 0// todo;;__AVX2__
           int alignedWidth = dstSize.width & ~7 ; //process 8 pixels in inner loop
           // for (; j < alignedWidth; j+= 8) {

           // }
#endif
            for (; j < dstSize.width; j++)
            {
                srcLocationRow = (srcLocationRowTerm1 + (rotate[1] * j) + srcLocationRowParameter2) * div_mul_factor;
                srcLocationColumn = (srcLocationColumnTerm1 + (-rotate[0] * j) + srcLocationColumnParameter2) * div_mul_factor;

                if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                {
                        *dstPtrTemp++ = 0;
                        *dstPtrTemp++ = 0;
                        *dstPtrTemp++ = 0;
                }
                else
                {
                    srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
                    srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;
                    Rpp32f mul0 = (1 - weightedHeight) * (1 - weightedWidth);
                    Rpp32f mul1 = (1 - weightedHeight) * weightedWidth;
                    Rpp32f mul2 = weightedHeight * (1 - weightedWidth);
                    Rpp32f mul3 = weightedHeight * weightedWidth;
                    Rpp32f R, G, B;

                    R =   (srcPtrTopRow[srcLocColFloorChanneled] * mul0)
                        + (srcPtrTopRow[srcLocColFloorChanneled+channel] * mul1)
                        + (srcPtrBottomRow[srcLocColFloorChanneled] * mul2)
                        + (srcPtrBottomRow[srcLocColFloorChanneled + channel] * mul3);
                    G =   (srcPtrTopRow[srcLocColFloorChanneled + 1] * mul0)
                        + (srcPtrTopRow[srcLocColFloorChanneled + channel + 1] * mul1)
                        + (srcPtrBottomRow[srcLocColFloorChanneled + 1] * mul2)
                        + (srcPtrBottomRow[srcLocColFloorChanneled + channel + 1] * mul3);
                    B =   (srcPtrTopRow[srcLocColFloorChanneled + 2] * mul0)
                        + (srcPtrTopRow[srcLocColFloorChanneled+channel + 2] * mul1)
                        + (srcPtrBottomRow[srcLocColFloorChanneled + 2] * mul2)
                        + (srcPtrBottomRow[srcLocColFloorChanneled + channel + 2] * mul3);

                    *dstPtrTemp++ = (Rpp8u) round(R);
                    *dstPtrTemp++ = (Rpp8u) round(G);
                    *dstPtrTemp++ = (Rpp8u) round(B);
                }
            }
        }
    }

    return RPP_SUCCESS;
}

#endif
#endif //AMD_RPP_HOST_GEOMETRY_TRANSFORMS_SIMD_HPP
