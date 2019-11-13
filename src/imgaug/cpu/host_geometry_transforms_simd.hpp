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
        int alignW = (srcSize.width*channel + 31) & ~31;
        if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {
//            srcPtrTemp = srcPtr + (channel * (srcSize.height * srcSize.width) - srcSize.width));
            srcPtrTemp = srcPtr + (channel * ((srcSize.height-1) * srcSize.width));
            for (int i = 0; i < srcSize.height; i++)
            {
                Rpp8u* pSrc = srcPtrTemp;
                int j=0;
#if __AVX2__                    
                __m256i p0;
                // copy multiple of 3 bytes for 3 channels
                for (; j < alignW; j+=30) {
                    p0 = _mm256_loadu_si256((const __m256i*)pSrc);
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p0);
                    pSrc +=30, dstPtrTemp += 30;
                }
#endif

                for (; j < srcSize.width; j++)
                {
                    *dstPtrTemp++ = *pSrc++;
                    *dstPtrTemp++ = *pSrc++;
                    *dstPtrTemp++ = *pSrc++;
                }
                srcPtrTemp -= srcSize.width*channel;

//                srcPtrTemp = srcPtrTemp - (channel * (2 * srcSize.width));
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
                for (; j < alignW; j+=30) {
                    pSrc -= 30
                    p0 = _mm256_loadu_si256((const __m256i*)pSrc);
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p0);
                    dstPtrTemp += 30;
                }
#endif
                for (j < srcSize.width*channel; j++)
                {
                    pSrc -= 3;
                    *dstPtrTemp++ = pSrc[0];
                    *dstPtrTemp++ = pSrc[1];
                    *dstPtrTemp++ = pSrc[2];
                }
                srcPtrTemp -= srcSize.width*channel;
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
                for (; j < alignW; j+=30) {
                    pSrc -= 30
                    p0 = _mm256_loadu_si256((const __m256i*)pSrc);
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p0);
                    dstPtrTemp += 30;
                }
#endif
                for (j < srcSize.width*channel; j++)
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

#endif

