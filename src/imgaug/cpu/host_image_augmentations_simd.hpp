#include <cpu/rpp_cpu_common.hpp>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#if ENABLE_SIMD_INTRINSICS

#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif

static inline Rpp32u HorMin(__m128i pmin)
{
    pmin = _mm_min_epi8(pmin, _mm_shuffle_epi32(pmin, _MM_SHUFFLE(3, 2, 3, 2)));
    pmin = _mm_min_epi8(pmin, _mm_shuffle_epi32(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin = _mm_min_epi8(pmin, _mm_shufflelo_epi16(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin = _mm_min_epi8(pmin, _mm_srli_epi16(pmin, 8));
    return (_mm_cvtsi128_si32(pmin) & 0x000000FF);    
}

static inline Rpp32u HorMax(__m128i pmax)
{
    pmax = _mm_max_epi8(pmax, _mm_shuffle_epi32(pmax, _MM_SHUFFLE(3, 2, 3, 2)));
    pmax = _mm_max_epi8(pmax, _mm_shuffle_epi32(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax = _mm_max_epi8(pmax, _mm_shufflelo_epi16(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax = _mm_max_epi8(pmax, _mm_srli_epi16(pmax, 8));
    return (_mm_cvtsi128_si32(pmax) & 0x000000FF);    
}

/**************** Contrast ***************/


RppStatus contrast_host_simd(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, 
                        Rpp32f new_min, Rpp32f new_max,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp8u *srcPtrTemp;
    Rpp8u *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f pixel, min, max;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            int length = (srcSize.height * srcSize.width);
            srcPtrTemp = srcPtr + (c * length);
            dstPtrTemp = dstPtr + (c * length);
            int alignedlength = length & ~15;
            Rpp32u min, max;
            __m128i px0, px1, px2, px3, pmin, pmax;
            __m128i const zero = _mm_setzero_si128();
            pmin = _mm_loadu_si128((__m128i *)srcPtrTemp);
            pmax = pmin;
            srcPtrTemp += 16;
            int i = 16;
            for (; i < alignedlength; i+=16)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                pmin = _mm_min_epi8(pmin, px0);
                pmax = _mm_max_epi8(pmax, px0);
                srcPtrTemp += 16;
            }
            min = HorMin(pmin);
            max = HorMax(pmax);
            for (; i < length; i++, srcPtrTemp++)
            {
                if (*srcPtrTemp < min)
                {
                    min = *srcPtrTemp;
                }
                if (*srcPtrTemp > max)
                {
                    max = *srcPtrTemp;
                }
            }
            srcPtrTemp = srcPtr + (c * length);
            Rpp32f contrast_factor = (Rpp32f)(new_max - new_min) / (max - min); 
            for (int i = 0; i < length; i++)
            {
                //*dstPtrTemp = (U) (((((Rpp32f) (*srcPtrTemp)) - min) * ((new_max - new_min) / (max - min))) + new_min);
                *dstPtrTemp = (Rpp8u) ((Rpp32f) ((*srcPtrTemp - min) * contrast_factor) + new_min);
                srcPtrTemp++;
                dstPtrTemp++;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        // todo;; code looks fishy: double check
        for(int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + c;
            dstPtrTemp = dstPtr + c;
            min = *srcPtrTemp;
            max = *srcPtrTemp;
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                if (*srcPtrTemp < min)
                {
                    min = *srcPtrTemp;
                }
                if (*srcPtrTemp > max)
                {
                    max = *srcPtrTemp;
                }
                srcPtrTemp = srcPtrTemp + channel;
            }

            srcPtrTemp = srcPtr + c;
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                *dstPtrTemp = (Rpp8u) (((((Rpp32f) (*srcPtrTemp)) - min) * ((new_max - new_min) / (max - min))) + new_min);
                srcPtrTemp = srcPtrTemp + channel;
                dstPtrTemp = dstPtrTemp + channel;
            }
        }
    }

    return RPP_SUCCESS;
}

/************ Brightness ************/

// TODO:: add AVX if supported
RppStatus brightness_host_simd(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
                          Rpp32f alpha, Rpp32f beta,
                          RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp8u *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    int length = (channel * srcSize.height * srcSize.width);
    int alignedlength = length & ~15;
    __m128i const zero = _mm_setzero_si128();
    __m128 pMul = _mm_set1_ps(alpha), pAdd = _mm_set1_ps(beta);
    __m128 p0, p1, p2, p3;
    __m128i px0, px1, px2, px3;
    int i = 0;
    for (; i < alignedlength; i+=16)
    {
        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp); // todo: check if we can use _mm_load_si128 instead (aligned)
        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
        p2 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));   // pixels 4-7
        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));  // pixels 0-3
        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));   // pixels 12-15
        p1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));  // pixels 8-11
        p0 = _mm_mul_ps(p0, pMul);
        p2 = _mm_mul_ps(p2, pMul);
        p1 = _mm_mul_ps(p1, pMul);
        p3 = _mm_mul_ps(p3, pMul);
        px0 = _mm_cvtps_epi32(_mm_add_ps(p0, pAdd));
        px2 = _mm_cvtps_epi32(_mm_add_ps(p2, pAdd));
        px1 = _mm_cvtps_epi32(_mm_add_ps(p1, pAdd));
        px3 = _mm_cvtps_epi32(_mm_add_ps(p3, pAdd));
        px0 = _mm_packus_epi32(px0, px2);
        px1 = _mm_packus_epi32(px1, px3);
        px0 = _mm_packus_epi16(px0, px1);       // pix 0-15
        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);      // todo: check if we can use _mm_store_si128 instead (aligned)
        srcPtrTemp +=16, dstPtrTemp +=16;
    }
    for (; i < length; i++) {
        Rpp32f pixel = ((Rpp32f) (*srcPtrTemp++)) * alpha + beta;
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp++ = (Rpp8u) round(pixel);
    }
    return RPP_SUCCESS;

}


/**************** Blend ***************/

RppStatus blend_host_simd(Rpp8u* srcPtr1, Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, 
                        Rpp32f alpha, RppiChnFormat chnFormat, 
                        unsigned int channel)
{
//#pragma omp parallel for
    Rpp8u *srcPtrTemp1, *srcPtrTemp2, *dstPtrTemp;
    srcPtrTemp1 = srcPtr1;
    srcPtrTemp2 = srcPtr2;
    dstPtrTemp = dstPtr;    
    int length = (channel * srcSize.height * srcSize.width);
    int alignedlength = length & ~15;
    __m128i const zero = _mm_setzero_si128();
    __m128 pMul1 = _mm_set1_ps(1.0-alpha), pMul2 = _mm_set1_ps(alpha);
    __m128 p0, p1, p2, p3, p4, p5, p6, p7;
    __m128i px0, px1, px2, px3;

    int i = 0;
    for (; i < alignedlength; i+=16)
    {
        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp1); // todo: check if we can use _mm_load_si128 instead (aligned)
        px2 =  _mm_loadu_si128((__m128i *)srcPtrTemp2); // todo: check if we can use _mm_load_si128 instead (aligned)
        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
        px3 = _mm_unpackhi_epi8(px2, zero);    // pixels 8-15
        px2 = _mm_unpacklo_epi8(px2, zero);    // pixels 0-7
        p2 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));   // pixels 4-7
        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));  // pixels 0-3
        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));   // pixels 12-15
        p1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));  // pixels 8-11
        p6 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px2, zero));   // pixels 4-7
        p4 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px2, zero));  // pixels 0-3
        p7 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px3, zero));   // pixels 12-15
        p5 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px3, zero));  // pixels 8-11

        p0 = _mm_mul_ps(p0, pMul1);
        p2 = _mm_mul_ps(p2, pMul1);
        p1 = _mm_mul_ps(p1, pMul1);
        p3 = _mm_mul_ps(p3, pMul1);
        p4 = _mm_mul_ps(p4, pMul2);
        p5 = _mm_mul_ps(p5, pMul2);
        p6 = _mm_mul_ps(p6, pMul2);
        p7 = _mm_mul_ps(p7, pMul2);

        px0 = _mm_cvtps_epi32(_mm_add_ps(p0, p4));
        px2 = _mm_cvtps_epi32(_mm_add_ps(p2, p6));
        px1 = _mm_cvtps_epi32(_mm_add_ps(p1, p5));
        px3 = _mm_cvtps_epi32(_mm_add_ps(p3, p7));
        px0 = _mm_packus_epi32(px0, px2);
        px1 = _mm_packus_epi32(px1, px3);
        px0 = _mm_packus_epi16(px0, px1);       // pix 0-15
        _mm_storeu_si128((__m128i *)dstPtr, px0);      // todo: check if we can use _mm_store_si128 instead (aligned)
        srcPtrTemp1 +=16, srcPtrTemp2 += 16, dstPtr +=16;

    }
    for (; i < length; i++) {
        *dstPtr++ = ((1. - alpha) * (*srcPtrTemp1++)) + (alpha * (*srcPtrTemp2++));
    }

    return RPP_SUCCESS;  
}

#endif