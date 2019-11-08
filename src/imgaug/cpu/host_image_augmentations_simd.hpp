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
    pmin = _mm_min_epu8(pmin, _mm_shuffle_epi32(pmin, _MM_SHUFFLE(3, 2, 3, 2)));
    pmin = _mm_min_epu8(pmin, _mm_shuffle_epi32(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin = _mm_min_epu8(pmin, _mm_shufflelo_epi16(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin = _mm_min_epu8(pmin, _mm_srli_epi16(pmin, 8));
    return (_mm_cvtsi128_si32(pmin) & 0x000000FF);    
}

static inline Rpp32u HorMax(__m128i pmax)
{
    pmax = _mm_min_epu8(pmax, _mm_shuffle_epi32(pmax, _MM_SHUFFLE(3, 2, 3, 2)));
    pmax = _mm_min_epu8(pmax, _mm_shuffle_epi32(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax = _mm_min_epu8(pmax, _mm_shufflelo_epi16(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax = _mm_min_epu8(pmax, _mm_srli_epi16(pmax, 8));
    return (_mm_cvtsi128_si32(pmax) & 0x000000FF);
}

#if __AVX__
static inline Rpp32u HorMin256(__m256i pmin)
{
    __m128i pmin_128;
    pmin = _mm256_min_epu8(pmin, _mm256_permute4x64_epi64(pmin, _MM_SHUFFLE(3, 2, 3, 2)));
    pmin = _mm256_min_epu8(pmin, _mm256_permute4x64_epi64(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin_128 = M256I(pmin).m256i_i128[0];
    pmin_128 = _mm_min_epu8(pmin_128, _mm_shufflelo_epi16(pmin_128, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin_128 = _mm_min_epu8(pmin_128, _mm_srli_epi16(pmin_128, 8));
    return (_mm_cvtsi128_si32(pmin_128) & 0x000000FF);
}

static inline Rpp32u HorMax256(__m256i pmax)
{
    __m128i pmax_128;
    pmax = _mm256_max_epu8(pmax, _mm256_permute4x64_epi64(pmax, _MM_SHUFFLE(3, 2, 3, 2)));
    pmax = _mm256_max_epu8(pmax, _mm256_permute4x64_epi64(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax_128 = M256I(pmax).m256i_i128[0];
    pmax_128 = _mm_max_epi8(pmax_128, _mm_shufflelo_epi16(pmax_128, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax_128 = _mm_max_epi8(pmax_128, _mm_srli_epi16(pmax_128, 8));
    return (_mm_cvtsi128_si32(pmax_128) & 0x000000FF);    
}
#endif

/**************** Contrast ***************/


RppStatus contrast_host_simd(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, 
                        Rpp32f new_min, Rpp32f new_max,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp8u *srcPtrTemp, *dstPtrTemp;

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
            __m128 p0, p1, p2, p3;
            __m128i const zero = _mm_setzero_si128();
            pmin = _mm_loadu_si128((__m128i *)srcPtrTemp);
            pmax = pmin;
            srcPtrTemp += 16;
            int i = 16;
            for (; i < alignedlength; i+=16)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                pmin = _mm_min_epu8(pmin, px0);
                pmax = _mm_max_epu8(pmax, px0);
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
            i = 0;
            pmin = _mm_set1_epi16(min);
            pmax = _mm_set1_epi16(new_min);
            __m128 p_cf = _mm_set1_ps(contrast_factor);
            for (; i < alignedlength; i++)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                px1 = _mm_sub_epi16(_mm_unpackhi_epi8(px0, zero), pmin);    // pixels 8-15
                px0 = _mm_sub_epi16(_mm_unpacklo_epi8(px0, zero), pmin);    // pixels 0-7
                p2 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));   // pixels 4-7
                p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));  // pixels 0-3
                p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));   // pixels 12-15
                p1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));  // pixels 8-11
                p0 = _mm_mul_ps(p0, p_cf);
                p2 = _mm_mul_ps(p2, p_cf);
                p1 = _mm_mul_ps(p1, p_cf);
                p3 = _mm_mul_ps(p3, p_cf);
                px0 = _mm_packus_epi32(_mm_cvtps_epi32(p0), _mm_cvtps_epi32(p2));
                px1 = _mm_packus_epi32(_mm_cvtps_epi32(p1), _mm_cvtps_epi32(p3));
                px0 = _mm_add_epi16(px0, pmax);
                px1 = _mm_add_epi16(px1, pmax);
                _mm_storeu_si128((__m128i *)dstPtrTemp, _mm_packus_epi16(px0,px1));
                srcPtrTemp += 16, dstPtrTemp += 16;
            }
            for (; i < length; i++)
            {
                //*dstPtrTemp = (U) (((((Rpp32f) (*srcPtrTemp)) - min) * ((new_max - new_min) / (max - min))) + new_min);
                *dstPtrTemp = (Rpp8u) ((Rpp32f) ((*srcPtrTemp - min) * contrast_factor) + new_min);
                srcPtrTemp++;
                dstPtrTemp++;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED) {
#if __AVX2__
        if (channel == 3) {
            int length = (srcSize.height * srcSize.width *3);
            int alignedlength = length & ~31;
           __m256i mask1 = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10, 13, 16, 19, 22, 
                                                2, 5, 8, 11, 14, 17, 20, 23, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
           __m256i pmin = _mm256_set1_epi8(0xFF);
           __m256i pmax = _mm256_set1_epi8(0x0);
           __m128i pminB = _mm_set1_epi8(0xFF);
           __m128i pmaxB = _mm_set1_epi8(0x0);
           __m256i px0, px1, px2;
            Rpp32u minR, maxR, minG, maxG, minB, maxB;
            int i = 0;
            for (; i < alignedlength; i+=48)
            {
                px0 = _mm256_loadu_si256((__m256i *)srcPtrTemp);
                px1 = _mm256_loadu_si256((__m256i *)(srcPtrTemp+24));
                px0 = _mm256_shuffle_epi8(px0, mask1);  // R(8), G(8), B(8), xx(8)
                px1 = _mm256_shuffle_epi8(px1, mask1);  // R(8), G(8), B(8), xx(8)
                px2 = _mm256_unpacklo_epi64(px0, px1);      // R(16),  G(16)
                px0 = _mm256_unpackhi_epi64(px0, px1);      // B(16) , xx(16)
                pmin = _mm256_min_epu8(pmin, px2);
                pmax = _mm256_max_epu8(pmax, px2);
                pminB = _mm_min_epu8(pminB, M256I(px0).m256i_i128[0]);
                pmaxB = _mm_max_epu8(pmaxB, M256I(px0).m256i_i128[0]);
                srcPtrTemp += 48;
            }
            minR = HorMin(M256I(pmin).m256i_i128[0]);
            maxR = HorMax(M256I(pmax).m256i_i128[0]);
            minG = HorMin(M256I(pmin).m256i_i128[1]);
            maxG = HorMax(M256I(pmax).m256i_i128[1]);
            minB = HorMin(pminB);
            maxB = HorMax(pmaxB);
            for (; i < length; i+=3)
            {
                Rpp8u R, G, B;
                R = srcPtrTemp[0], G =  srcPtrTemp[1], B = srcPtrTemp[2];
                if (R < minR) minR = R;
                if (R > maxR) maxR = R;
                if (G < minG) minG = G;
                if (G > maxG) maxG = G;
                if (R < minB) minB = B;
                if (R > maxB) maxB = B;
                srcPtrTemp += 3;
            }
            Rpp32f contrast_factor_R = (Rpp32f)(new_max - new_min) / (maxR - minR);
            Rpp32f contrast_factor_G = (Rpp32f)(new_max - new_min) / (maxG - minG);
            Rpp32f contrast_factor_B = (Rpp32f)(new_max - new_min) / (maxB - minB);
            // todo:: do the following code to SSE/AVX
            length = srcSize.height * srcSize.width*3;
            Rpp8u *srcR = srcPtr, *srcG = srcPtr+1, *srcB = srcPtr+2;
            Rpp8u *dstR = dstPtr, *dstG = dstPtr+1, *dstB = dstPtr+2;    
        //#pragma omp parallel for
            for (i = 0; i < length; i+=3)
            {
                *dstR = (Rpp8u) ((Rpp32f) ((*srcR - minR) * contrast_factor_R) + new_min);
                *dstG = (Rpp8u) ((Rpp32f) ((*srcG - minG) * contrast_factor_G) + new_min);
                *dstB = (Rpp8u) ((Rpp32f) ((*srcB - minB) * contrast_factor_B) + new_min);
                srcR += 3, srcG += 3, srcB += 3;
                dstR += 3, dstG += 3, dstB += 3;
            }
        }else {
#endif
            for(int c = 0; c < channel; c++)
            {
                Rpp32u min, max;
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
                Rpp32f contrast_factor = (Rpp32f)(new_max - new_min) / (max - min);
    #pragma omp parallel for simd
                for (int i = 0; i < (srcSize.height * srcSize.width); i += channel)
                {
                    *dstPtrTemp = (Rpp8u) ((Rpp32f) ((*srcPtrTemp - min) * contrast_factor) + new_min);
                }
            }
#if __AVX__
        }
#endif
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

/**************** Exposure Modification ***************/

RppStatus exposure_host_simd(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
                    Rpp32f exposureFactor,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp8u *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    Rpp32f pixel;
    Rpp32f exposure_mul = pow(2,exposureFactor);

#if __AVX2__
    int length = (channel * srcSize.height * srcSize.width);
    int alignedlength = length & ~31;
    __m256 pMul1 = _mm256_set1_ps(exposure_mul);
    __m256i const zero = _mm256_setzero_si256();
    __m256i px0, px1, px2, px3;
    __m256 p0, p1, p2, p3;
    int i = 0;
    for (; i < alignedlength; i+=32)
    {
        px0 =  _mm256_loadu_si256((__m256i *)srcPtrTemp); // todo: check if we can use _mm_load_si128 instead (aligned)
        px1 = _mm256_unpackhi_epi8(px0, zero);    // pixels 16-31
        px0 = _mm256_unpacklo_epi8(px0, zero);    // pixels 0-15
        p2 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(px0, zero));   // pixels 8-15
        p0 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(px0, zero));  // pixels 0-7
        p3 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(px1, zero));   // pixels 24-31
        p1 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(px1, zero));  // pixels 16-23
        px0 = _mm256_cvtps_epi32(_mm256_mul_ps(p0, pMul1));
        px2 = _mm256_cvtps_epi32(_mm256_mul_ps(p2, pMul1));
        px1 = _mm256_cvtps_epi32(_mm256_mul_ps(p1, pMul1));
        px3 = _mm256_cvtps_epi32(_mm256_mul_ps(p3, pMul1));
        px0 = _mm256_packus_epi32(px0, px2);
        px1 = _mm256_packus_epi32(px1, px3);
        px0 = _mm256_packus_epi16(px0, px1);       // pix 0-31
        _mm256_storeu_si256((__m256i *)dstPtr, px0);      // todo: check if we can use _mm_store_si128 instead (aligned)
        srcPtrTemp +=32, dstPtr +=32;
    }
    for (; i < length; i++) {
        pixel = ((Rpp32f) (srcPtrTemp[i])) * exposure_mul;
        dstPtrTemp[i] = (Rpp8u)RPPPIXELCHECK(pixel);
    }

#else
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32f) (srcPtrTemp[i])) * exposure_mul;
        dstPtrTemp[i] = (Rpp8u)RPPPIXELCHECK(pixel);
    }
#endif
    return RPP_SUCCESS;
}

#endif