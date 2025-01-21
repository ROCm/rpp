#ifndef RPP_CPU_COMMON_GEOMETRIC_H
#define RPP_CPU_COMMON_GEOMETRIC_H

inline void compute_offset_i8_1c_avx(__m256 &p)
{
    p = _mm256_add_ps(p, avx_p128);
}

inline void compute_offset_i8_3c_avx(__m256 *p)
{
    compute_offset_i8_1c_avx(p[0]);
    compute_offset_i8_1c_avx(p[1]);
    compute_offset_i8_1c_avx(p[2]);
}

/* Resize helper functions */
inline void compute_dst_size_cap_host(RpptImagePatchPtr dstImgSize, RpptDescPtr dstDescPtr)
{
    dstImgSize->width = std::min(dstImgSize->width, dstDescPtr->w);
    dstImgSize->height = std::min(dstImgSize->height, dstDescPtr->h);
}

// Resize
inline void compute_resize_src_loc(Rpp32s dstLocation, Rpp32f scale, Rpp32s &srcLoc, Rpp32f &weight, Rpp32f offset = 0, Rpp32u srcStride = 1)
{
    Rpp32f srcLocationFloat = ((Rpp32f) dstLocation) * scale + offset;
    Rpp32s srcLocation = (Rpp32s) std::ceil(srcLocationFloat);
    weight = srcLocation - srcLocationFloat;
    srcLoc = srcLocation * srcStride;
}

inline void compute_resize_nn_src_loc(Rpp32s dstLocation, Rpp32f scale, Rpp32u limit, Rpp32s &srcLoc, Rpp32f offset = 0, Rpp32u srcStride = 1)
{
    Rpp32f srcLocation = ((Rpp32f) dstLocation) * scale + offset;
    Rpp32s srcLocationFloor = std::floor(srcLocation);
    srcLoc = ((srcLocationFloor > limit) ? limit : srcLocationFloor) * srcStride;
}

inline void compute_resize_bilinear_src_loc_and_weights(Rpp32s dstLocation, Rpp32f scale, Rpp32s &srcLoc, Rpp32f *weight, Rpp32f offset = 0, Rpp32u srcStride = 1)
{
    compute_resize_src_loc(dstLocation, scale, srcLoc, weight[1], offset, srcStride);
    weight[0] = 1 - weight[1];
}

inline void compute_resize_nn_src_loc_sse(__m128 &pDstLoc, __m128 &pScale, __m128 &pLimit, Rpp32s *srcLoc, __m128 pOffset = xmm_p0, bool hasRGBChannels = false)
{
    __m128 pLoc = _mm_fmadd_ps(pDstLoc, pScale, pOffset);
    pDstLoc = _mm_add_ps(pDstLoc, xmm_p4);
    __m128 pLocFloor = _mm_floor_ps(pLoc);
    pLocFloor = _mm_max_ps(_mm_min_ps(pLocFloor, pLimit), xmm_p0);
    if(hasRGBChannels)
        pLocFloor = _mm_mul_ps(pLocFloor, xmm_p3);
    __m128i pxLocFloor = _mm_cvtps_epi32(pLocFloor);
    _mm_storeu_si128((__m128i*) srcLoc, pxLocFloor);
}

inline void compute_resize_bilinear_src_loc_and_weights_avx(__m256 &pDstLoc, __m256 &pScale, Rpp32s *srcLoc, __m256 *pWeight, __m256i &pxLoc, __m256 pOffset = avx_p0, bool hasRGBChannels = false)
{
    __m256 pLocFloat = _mm256_fmadd_ps(pDstLoc, pScale, pOffset);
    pDstLoc = _mm256_add_ps(pDstLoc, avx_p8);
    __m256 pLoc = _mm256_ceil_ps(pLocFloat);
    pWeight[1] = _mm256_sub_ps(pLoc, pLocFloat);
    pWeight[0] = _mm256_sub_ps(avx_p1, pWeight[1]);
    if(hasRGBChannels)
        pLoc = _mm256_mul_ps(pLoc, avx_p3);
    pxLoc = _mm256_cvtps_epi32(pLoc);
    _mm256_storeu_si256((__m256i*) srcLoc, pxLoc);
}

inline void compute_resize_bilinear_src_loc_and_weights_mirror_avx(__m256 &pDstLoc, __m256 &pScale, Rpp32s *srcLoc, __m256 *pWeight, __m256i &pxLoc, __m256 pOffset = avx_p0, bool hasRGBChannels = false)
{
    __m256 pLocFloat = _mm256_fmadd_ps(pDstLoc, pScale, pOffset);
    pDstLoc = _mm256_sub_ps(pDstLoc, avx_p8);
    __m256 pLoc = _mm256_ceil_ps(pLocFloat);
    pWeight[1] = _mm256_sub_ps(pLoc, pLocFloat);
    pWeight[0] = _mm256_sub_ps(avx_p1, pWeight[1]);
    if (hasRGBChannels)
        pLoc = _mm256_mul_ps(pLoc, avx_p3);
    pxLoc = _mm256_cvtps_epi32(pLoc);
    _mm256_storeu_si256((__m256i *)srcLoc, pxLoc);
}
#endif