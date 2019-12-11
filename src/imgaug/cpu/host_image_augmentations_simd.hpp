#include <cpu/rpp_cpu_common.hpp>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <omp.h>
#include "cpu/rpp_cpu_simd.hpp"

#if ENABLE_SIMD_INTRINSICS

static unsigned int g_seed;
//Used to seed the generator.
inline void fast_srand( int seed )
{
    g_seed = seed;
}

//fastrand routine returns one integer, similar output value range as C lib. :: taken from intel
inline int fastrand()
{
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16)&0x7FFF;
}

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

#define  FOGVALUE_11(x, val)          (int)(x*(1.5 + val) + 3*val)
#define  FOGVALUE_12(x, val)          (int)(x*(1.5 + val) + 7*val)
#define  FOGVALUE_13(x, val)          (int)(x*(1.5 + val) + 11*val)
#define  FOGVALUE_21(x, val)          (int)(x*(1.5 + val*val) + 126*val)
#define  FOGVALUE_22(x, val)          (int)(x*(1.5 + val*val) + 130*val)
#define  FOGVALUE_23(x, val)          (int)(x*(1.5 + val*val) + 134*val)
#define  FOGVALUE_31(x, val)          (int)(x*(1.5 + val*(val*1.414)) + 96*val + 20)
#define  FOGVALUE_32(x, val)          (int)(x*(1.5 + val*(val*1.414)) + 100*val + 20)
#define  FOGVALUE_33(x, val)          (int)(x*(1.5 + val*(val*1.414)) + 104*val)

/**************** Contrast ***************/

template<>
RppStatus contrast_host(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
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
            srcPtrTemp = srcPtr ;
            dstPtrTemp = dstPtr ;
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
template<>
RppStatus brightness_host(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
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
template<>
RppStatus blend_host(Rpp8u* srcPtr1, Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr,
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
template<>
RppStatus exposure_host(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
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

/**************** Gamma Correction ***************/

template <>
RppStatus gamma_correction_host(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
                                Rpp32f gamma,
                                RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp8u *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    Rpp8u gamma_mat[256];       // gamma lookup_table

    for (int i = 0; i < 256; i++) {
        gamma_mat[i] = RPPPIXELCHECK((int)(pow( ((Rpp32f) i / 255.0), gamma ) * 255));
    }
    int length = (channel * srcSize.height * srcSize.width);

#pragma omp parallel for
    for (int i = 0; i < length ; i++)
    {
        dstPtrTemp[i] = (Rpp8u) gamma_mat[srcPtrTemp[i]];
    }
    return RPP_SUCCESS;
}

/************ Blur************/

#define CHECK_SIMD 0

template <>
RppStatus blur_host(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
                    Rpp32f stdDev, Rpp32u kernelSize,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }

    float one_ninth = (float)1.0/9;
#if __AVX2__
    __m256i const one_nineth = _mm256_set1_epi16((short)7282);
    __m256i const zero = _mm256_setzero_si256();
#endif      

    if (chnFormat == RPPI_CHN_PLANAR)
    {
#pragma omp parallel for
        for (int c = 0; c < channel; c++)
        {
            Rpp8u *src_ptr = srcPtr + c*srcSize.width*srcSize.height;
            Rpp8u *dst_ptr = dstPtr + c*srcSize.width*srcSize.height;

            int alignedWidth = srcSize.width & ~15;

            for (int row = 0; row < srcSize.height; row++)
            {
                Rpp8u* row0 = &src_ptr[srcSize.width*row];
                Rpp8u* rowp = (row > 0) ? row0 - srcSize.width : row0;
                Rpp8u* rown = (row < (srcSize.height -1)) ? row0 + srcSize.width : row0;
                Rpp8u* dst = dst_ptr + srcSize.width*row;
                int x = 0;
#if __AVX2__
                for (; x < alignedWidth; x+=16) {
                    __m256i s0 ,s1 ,s2;
                    __m256i r0, r1, r2;            
                    s0 = _mm256_permute4x64_epi64(_mm256_loadu2_m128i( (__m128i*)(rowp-1), (__m128i*)(row0-1)), 0xd8);
                    s1 = _mm256_permute4x64_epi64(_mm256_loadu2_m128i( (__m128i*)(rowp+1), (__m128i*)(row0+1)), 0xd8);
                    s2 = _mm256_permute4x64_epi64(_mm256_loadu2_m128i( (__m128i*)(rowp), (__m128i*)(row0)), 0xd8);
                    r0 = _mm256_unpacklo_epi8(s0, zero);
                    r1 = _mm256_unpacklo_epi8(s1, zero);
                    r2 = _mm256_unpacklo_epi8(s0, zero);

                    r0 = _mm256_unpacklo_epi8(s0, zero);
                    r1 = _mm256_unpacklo_epi8(s1, zero);
                    r2 = _mm256_unpacklo_epi8(s2, zero);

                    r0 = _mm256_add_epi16( _mm256_add_epi16( r0, r1 ), r2 );

                    M256I(s0).m256i_i128[0] = _mm_loadu_si128( (__m128i*)(rown-1) );
                    M256I(s1).m256i_i128[0] = _mm_loadu_si128( (__m128i*)(rown+1) );
                    M256I(s2).m256i_i128[0] = _mm_loadu_si128( (__m128i*)(rown) );
                    s0 = _mm256_unpacklo_epi8(_mm256_permute4x64_epi64(s0, 0xd8), zero);
                    s1 = _mm256_unpacklo_epi8(_mm256_permute4x64_epi64(s1, 0xd8), zero);
                    s2 = _mm256_unpacklo_epi8(_mm256_permute4x64_epi64(s2, 0xd8), zero);
                    r2 = _mm256_add_epi16( _mm256_add_epi16( s0, s1 ), s2 );
                    s0 = _mm256_mulhi_epi16( _mm256_add_epi16( _mm256_add_epi16( r0, r1 ), r2 ), one_nineth );
                    s0 = _mm256_packus_epi16(s0, _mm256_permute4x64_epi64(s0, 0x4e));

                    _mm_storeu_si128( (__m128i *)dst, M256I(s0).m256i_i128[0]);

                    row0 += 16;
                    rowp += 16;
                    rown += 16;
                    dst += 16;                
                }
#endif                
                for (; x < srcSize.width; x++)
                {
                    int val =  row0[0] + rowp[0] + rown[0];
                    val += row0[-1] + rowp[-1] + rown[-1];
                    if (x < (srcSize.width-1))
                        val += row0[1] + rowp[1] + rown[1];
                    else
                        val += row0[0] + rowp[0] + rown[0];          // don't exceed boundary          

                    *dst++ =  (Rpp8u)(val * one_ninth);
                    row0 ++, rowp++, rown++;
                }

            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
#pragma omp parallel for
        for (int row = 0; row < srcSize.height; row++)
        {
            int width = srcSize.width*channel;
            int alignedWidth = (srcSize.width/5)*5*channel;         // processing 5 pixels at a time
            Rpp8u* row0 = &srcPtr[width*row];
            Rpp8u* rowp = (row > 0) ? row0 - width : row0;
            Rpp8u* rown = (row < (srcSize.height -1)) ? row0 + width : row0;
            Rpp8u* dst = dstPtr + width*row;
            int x = 0;
            if (channel != 3) alignedWidth = 0;

#if __AVX2__
            for (; x < alignedWidth; x+=15) {
                __m256i s0 ,s1 ,s2;
                __m256i r0, r1, r2;            
                s0 = _mm256_permute4x64_epi64(_mm256_loadu2_m128i( (__m128i*)(rowp-3), (__m128i*)(row0-3)), 0xd8);
                s1 = _mm256_permute4x64_epi64(_mm256_loadu2_m128i( (__m128i*)(rowp+3), (__m128i*)(row0+3)), 0xd8);
                s2 = _mm256_permute4x64_epi64(_mm256_loadu2_m128i( (__m128i*)(rowp), (__m128i*)(row0)), 0xd8);
                r0 = _mm256_unpacklo_epi8(s0, zero);
                r1 = _mm256_unpacklo_epi8(s1, zero);
                r2 = _mm256_unpacklo_epi8(s2, zero);

                r0 = _mm256_add_epi16( _mm256_add_epi16( r0, r1 ), r2 );

                s0 = _mm256_unpackhi_epi8(s0, zero);
                s1 = _mm256_unpackhi_epi8(s1, zero);
                s2 = _mm256_unpackhi_epi8(s2, zero);
                r1 = _mm256_add_epi16( _mm256_add_epi16( s0, s1 ), s2 );

                M256I(s0).m256i_i128[0] = _mm_loadu_si128( (__m128i*)(rown-3) );
                M256I(s1).m256i_i128[0] = _mm_loadu_si128( (__m128i*)(rown+3) );
                M256I(s2).m256i_i128[0] = _mm_loadu_si128( (__m128i*)(rown) );
                s0 = _mm256_unpacklo_epi8(_mm256_permute4x64_epi64(s0, 0xd8), zero);
                s1 = _mm256_unpacklo_epi8(_mm256_permute4x64_epi64(s1, 0xd8), zero);
                s2 = _mm256_unpacklo_epi8(_mm256_permute4x64_epi64(s2, 0xd8), zero);
                r2 = _mm256_add_epi16( _mm256_add_epi16( s0, s1 ), s2 );
                s0 = _mm256_mulhi_epi16( _mm256_add_epi16( _mm256_add_epi16( r0, r1 ), r2 ), one_nineth );
                s0 = _mm256_packus_epi16(s0, _mm256_permute4x64_epi64(s0, 0x4e));

                _mm_storeu_si128( (__m128i *)dst, M256I(s0).m256i_i128[0]);

                row0 += 15;
                rowp += 15;
                rown += 15;
                dst += 15;                
            }
#elif __SSE4_2__
            __m128i const one_nineth = _mm_set1_epi16((short)7282);
            __m128i const zero = _mm_setzero_si128();
            for (; x < alignedWidth; x+=15) {
                __m128i s0 ,s1 ,s2;
                __m128i r0, r1, r2, r3, r4, r5, r6;            
                s0 = _mm_loadu_si128( (__m128i*)(row0-3));
                s1 = _mm_loadu_si128( (__m128i*)(row0+3));
                s2 = _mm_loadu_si128( (__m128i*)(row0));
                r0 = _mm_unpacklo_epi8(s0, zero);
                r1 = _mm_unpacklo_epi8(s1, zero);
                r2 = _mm_unpacklo_epi8(s2, zero);
                r0 = _mm_add_epi16( _mm_add_epi16( r0, r1 ), r2 );      // pix 0- 7 for row0

                s0 = _mm_unpackhi_epi8(s0, zero);
                s1 = _mm_unpackhi_epi8(s1, zero);
                s2 = _mm_unpackhi_epi8(s2, zero);
                r1 = _mm_add_epi16( _mm_add_epi16( s0, s1 ), s2 );   // // pix 8 - 15 for row0
                
                s0 = _mm_loadu_si128( (__m128i*)(rowp-3));
                s1 = _mm_loadu_si128( (__m128i*)(rowp+3));
                s2 = _mm_loadu_si128( (__m128i*)(rowp));
                r2 = _mm_unpacklo_epi8(s0, zero);
                r3 = _mm_unpacklo_epi8(s1, zero);
                r4 = _mm_unpacklo_epi8(s2, zero);                
                r2 = _mm_add_epi16( _mm_add_epi16( r2, r3 ), r4 );      // pix 0- 7 for rowp
                
                s0 = _mm_unpackhi_epi8(s0, zero);
                s1 = _mm_unpackhi_epi8(s1, zero);
                s2 = _mm_unpackhi_epi8(s2, zero);
                r3 = _mm_add_epi16( _mm_add_epi16( s0, s1 ), s2 );      // pix 8- 15 for rowp

                s0 = _mm_loadu_si128( (__m128i*)(rown-3) );
                s1 = _mm_loadu_si128( (__m128i*)(rown+3) );
                s2 = _mm_loadu_si128( (__m128i*)(rown) );
                r4 = _mm_unpacklo_epi8(s0, zero);                
                r5 = _mm_unpacklo_epi8(s1, zero);
                r6 = _mm_unpacklo_epi8(s2, zero);
                r4 = _mm_add_epi16( _mm_add_epi16( r4, r5 ), r6 );      // pix 0- 7 for rown

                s0 = _mm_unpackhi_epi8(s0, zero);
                s1 = _mm_unpackhi_epi8(s1, zero);
                s2 = _mm_unpackhi_epi8(s2, zero);
                r0 = _mm_mulhi_epi16( _mm_add_epi16( _mm_add_epi16( r0, r2 ), r4 ), one_nineth );
                s0 = _mm_add_epi16( _mm_add_epi16( s0, s1 ), s2 );   // // pix 8 - 15 for rown
                r1 = _mm_mulhi_epi16( _mm_add_epi16( _mm_add_epi16( r1, r3 ), s0 ), one_nineth );

                _mm_storeu_si128( (__m128i *)dst, _mm_packus_epi16(r0, r1));

                row0 += 15;
                rowp += 15;
                rown += 15;
                dst += 15;                
            }            
#endif         
            for (; x < width; x++)
            {
                unsigned short val =  row0[0] + rowp[0] + rown[0];
                val += row0[-3] + rowp[-3] + rown[-3];
                if (x < (width-3))
                    val += row0[3] + rowp[3] + rown[3];
                else
                    val += row0[0] + rowp[0] + rown[0];          // don't exceed boundary          
                dst[x] =  (Rpp8u)((val*7282)>>16);
            }

#if CHECK_SIMD
            Rpp8u *pdstRef =  dstPtr + width*row;
            row0 = &srcPtr[width*row];
            rowp = (row > 0) ? row0 - width : row0;
            rown = (row < (srcSize.height -1)) ? row0 + width : row0;
            for (x=0; x < width; x++) {
                unsigned short val =  row0[x] + rowp[x] + rown[x];
                val += row0[x-3] + rowp[x-3] + rown[x-3];
                if (x < (width-3))
                    val += row0[x+3] + rowp[x+3] + rown[x+3];
                else
                    val += row0[x] + rowp[x] + rown[x];          // don't exceed boundary   
                val =  ((val*7282)>>16); 
                if ( !row && (Rpp8u)val !=  pdstRef[x] )     
                    printf("Error: pixel mismatch at %d: %d != %d(ref)\n", x, pdstRef[x], (Rpp8u)val);
            }
#endif

        }
    }
    return RPP_SUCCESS;
}

/**************** Fog ***************/
template <>
RppStatus fog_host(Rpp8u* srcPtr, RppiSize srcSize,
                   Rpp32f fogValue,
                   RppiChnFormat chnFormat,  unsigned int channel, Rpp8u* temp)
{
    if(fogValue <= 0)
    {
        int length = srcSize.height * srcSize.width * channel;
        int i = 0;
#if __AVX2__  
        Rpp8u *src = temp;
        Rpp8u *dst = srcPtr;
        int alignedLength = length & ~63;
        for (; i < alignedLength; i+=64)
        {
            _mm256_storeu_si256((__m256i *)dst, _mm256_loadu_si256((__m256i *) src));
            _mm256_storeu_si256((__m256i *)(dst+32), _mm256_loadu_si256((__m256i *) (src+32)));
            dst += 64;
            src += 64;
        }
#endif        
        for(; i < length; i++)
        {
            *dst++ = *src++;
        }
    }

    if(fogValue != 0)
    {
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            Rpp8u *srcPtr1, *srcPtr2;
            if(channel > 1)
            {
                srcPtr1 = srcPtr + (srcSize.width * srcSize.height);
                srcPtr2 = srcPtr + (srcSize.width * srcSize.height * 2);
            }
#pragma omp parallel for
            for (int i = 0; i < (srcSize.width * srcSize.height); i++)
            {
                Rpp32f check= *srcPtr;
                if(channel > 1)
                    check = (check + *srcPtr1 + *srcPtr2) / 3;
                if (check < 240) {
                    if (check>=(170)) {
                        *srcPtr =  (Rpp8u) std::clamp(FOGVALUE_11(*srcPtr, fogValue), 0, 255);
                        *srcPtr1 = (Rpp8u) std::clamp(FOGVALUE_12(*srcPtr1, fogValue), 0, 255);
                        *srcPtr2 = (Rpp8u) std::clamp(FOGVALUE_13(*srcPtr2, fogValue), 0, 255);
                    }
                    else if (check<=(85)) {
                        *srcPtr =  (Rpp8u) std::clamp(FOGVALUE_21(*srcPtr, fogValue), 0, 255);
                        *srcPtr1 = (Rpp8u) std::clamp(FOGVALUE_22(*srcPtr1, fogValue), 0, 255);
                        *srcPtr2 = (Rpp8u) std::clamp(FOGVALUE_23(*srcPtr2, fogValue), 0, 255);                        
                    }
                    else {
                        *srcPtr =  (Rpp8u) std::clamp(FOGVALUE_31(*srcPtr, fogValue), 0, 255);
                        *srcPtr1 = (Rpp8u) std::clamp(FOGVALUE_32(*srcPtr1, fogValue), 0, 255);
                        *srcPtr2 = (Rpp8u) std::clamp(FOGVALUE_33(*srcPtr2, fogValue), 0, 255);                                                
                    }
                }
                srcPtr++;
                srcPtr1++;
                srcPtr2++;
            }
        }
        else
        {
            Rpp8u * srcPtr1 = srcPtr + 1;
            Rpp8u * srcPtr2 = srcPtr + 2;
#pragma omp parallel for
            for (int i = 0; i < (srcSize.width * srcSize.height * channel); i += 3)
            {
                Rpp32f check = (srcPtr[i] + srcPtr1[i] + srcPtr2[i]) / 3;
                if (check < 240) {
                    if (check>=(170)) {
                        srcPtr[i] =  (Rpp8u) std::clamp(FOGVALUE_11(srcPtr[i], fogValue), 0, 255);
                        srcPtr1[i] = (Rpp8u) std::clamp(FOGVALUE_12(srcPtr1[i], fogValue), 0, 255);
                        srcPtr2[i] = (Rpp8u) std::clamp(FOGVALUE_13(srcPtr2[i], fogValue), 0, 255);
                    }
                    else if (check<=(85)) {
                        srcPtr[i] =  (Rpp8u) std::clamp(FOGVALUE_21(srcPtr[i], fogValue), 0, 255);
                        srcPtr1[i] = (Rpp8u) std::clamp(FOGVALUE_22(srcPtr1[i], fogValue), 0, 255);
                        srcPtr2[i] = (Rpp8u) std::clamp(FOGVALUE_23(srcPtr2[i], fogValue), 0, 255);
                    }
                    else {
                        srcPtr[i] =  (Rpp8u) std::clamp(FOGVALUE_21(srcPtr[i], fogValue), 0, 255);
                        srcPtr1[i] = (Rpp8u) std::clamp(FOGVALUE_22(srcPtr1[i], fogValue), 0, 255);
                        srcPtr2[i] = (Rpp8u) std::clamp(FOGVALUE_23(srcPtr2[i], fogValue), 0, 255);
                    }
                }
            }

        }
    }
    return RPP_SUCCESS;
}

/**************** Rain ***************/
template <>
RppStatus rain_host(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
                    Rpp32f rainPercentage, Rpp32f rainWidth, Rpp32f rainHeight, Rpp32f transparency,
                    RppiChnFormat chnFormat,   unsigned int channel)
{ 
    rainPercentage *= 0.004;
    transparency *= 0.2;

    const Rpp32u rainDrops = (Rpp32u)(rainPercentage * srcSize.width * srcSize.height * channel);
    // seed the random gen
    fast_srand(std::time(0));
    const unsigned rand_len = srcSize.width;
    unsigned int col_rand[rand_len];
    unsigned int row_rand[rand_len];
    for(int i = 0; i<  rand_len; i++)
    {
        col_rand[i] = fastrand() % srcSize.width;
        row_rand[i] = fastrand() % srcSize.height;
    }
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
#pragma omp parallel for
        for(int i = 0 ; i < rainDrops ; i++)
        {
            int rand_idx = i%rand_len;
            Rpp32u row = row_rand[rand_idx];
            Rpp32u column = col_rand[rand_idx];
            Rpp32f pixel;
            Rpp8u *dst = &dstPtr[(row * srcSize.width) + column];
            Rpp8u *dst1, *dst2;
            if (channel > 1){
                dst1 = dst + srcSize.width*srcSize.height;
                dst2 = dst1 + srcSize.width*srcSize.height;
            }
            for(int j = 0;j < rainHeight;j++)
            {
                for(int m = 0;m < rainWidth;m++)
                {
                    if ( (row + rainHeight) < srcSize.height && (column + rainWidth) < srcSize.width)
                    {
                        int idx = srcSize.width*j + m; 
                        dst[idx] = 196;
                        if (channel > 1) {
                            dst1[idx] = 226, dst2[idx] = 255;
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
#pragma omp parallel for
        for(int i = 0 ; i < rainDrops ; i++)
        {
            int rand_idx = i%rand_len;
            Rpp32u row = row_rand[rand_idx];
            Rpp32u column = col_rand[rand_idx];
            Rpp32f pixel;
            Rpp8u *dst = &dstPtr[(row * srcSize.width*channel) + column*channel];
            for(int j = 0;j < rainHeight; j++)
            {
                for(int m = 0;m < rainWidth; m++)
                {
                    if ((row + rainHeight) < srcSize.height && (column + rainWidth) < srcSize.width )
                    {
                        int idx = (j*srcSize.width*channel) + m*channel;
                        dst[idx] = 196;
                        if (channel > 1) {
                            dst[idx+1] = 226;
                            dst[idx+2] = 255;
                        }
                    }
                } 
            }
        }
    }

    Rpp8u *src = &srcPtr[0];
    Rpp8u *dst = &dstPtr[0];
    int length = channel * srcSize.width * srcSize.height;
    int i=0;
#if __AVX2__  
    int alignedLength = length & ~31;
    __m256i const trans = _mm256_set1_epi16((unsigned short)(transparency*65535));       // 1/5th
    __m256i const zero = _mm256_setzero_si256();    
    for (; i < alignedLength; i+=32)
    {
        __m256i s0, s1, r0;
        s0 = _mm256_loadu_si256((__m256i *) dst);
        r0 = _mm256_loadu_si256((__m256i *) src);
        s1 = _mm256_unpacklo_epi8(s0, zero);
        s0 = _mm256_unpackhi_epi8(s0, zero);
        s1 = _mm256_mulhi_epi16(s1, trans);
        s0 = _mm256_mulhi_epi16(s0, trans);
        s1 = _mm256_add_epi16(s1, _mm256_unpacklo_epi8(r0, zero));
        s0 = _mm256_add_epi16(s0, _mm256_unpackhi_epi8(r0, zero));
        _mm256_storeu_si256((__m256i *)dst, _mm256_packus_epi16(s1, s0));
        dst += 32;
        src += 32;
    }
#else
    int alignedLength = length & ~15;
    __m128i const trans = _mm_set1_epi16((unsigned short)(transparency*65535));       // trans factor 
    __m128i const zero = _mm_setzero_si128();
    for (; i < alignedLength; i+=16)
    {
        __m128i s0, s1, r0;
        s0 = _mm_loadu_si128((__m128i *) dst);
        r0 = _mm_loadu_si128((__m128i *) src);
        s1 = _mm_unpacklo_epi8(s0, zero);
        s0 = _mm_unpackhi_epi8(s0, zero);
        s1 = _mm_mulhi_epi16(s1, trans);
        s0 = _mm_mulhi_epi16(s0, trans);
        s1 = _mm_add_epi16(s1, _mm_unpacklo_epi8(r0, zero));
        s0 = _mm_add_epi16(s0, _mm_unpackhi_epi8(r0, zero));
        _mm_storeu_si128((__m128i *)dst, _mm_packus_epi16(s1, s0));
        dst += 16;
        src += 16;
    }
#endif        

    for (; i < length; i++)
    {
        dst[i] = (Rpp8u) std::clamp((int)(src[i] + transparency * dst[i]), 0, 255);
    }
    return RPP_SUCCESS;
}

#endif
