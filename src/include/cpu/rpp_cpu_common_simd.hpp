#ifndef RPP_CPU_COMMON_SIMD_H
#define RPP_CPU_COMMON_SIMD_H


#include "rpp_cpu_simd.hpp"

#if ENABLE_SIMD_INTRINSICS
#define CHECK_SIMD      0

#define FP_BITS     16
#define FP_MUL      (1<<FP_BITS)

template<>
inline RppStatus resize_kernel_host(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiSize dstSize,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
   // return RPP_SUCCESS;
    if (dstSize.height < 0 || dstSize.width < 0 )
    {
        return RPP_ERROR;
    }
    // call ref host implementation
    //if (channel > 3 )
    //    return resize_kernel_host(srcPtr, dstPtr, dstSize, chnFormat, channel);
#if CHECK_SIMD
    Rpp8u *tmpBuff = new Rpp8u[dstSize.width*dstSize.height*3];
#endif

    Rpp8u *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    Rpp32f yscale = ((Rpp32f) (srcSize.height)) / ((Rpp32f) (dstSize.height));    // yscale
    Rpp32f xscale = ((Rpp32f) (srcSize.width)) / ((Rpp32f) (dstSize.width));      //xscale
    int alignW = (dstSize.width + 15) & ~15;
    // generate maps for fixed point computations
    unsigned int *Xmap = new unsigned int[alignW*2];
    unsigned short *Xf = (unsigned short *)(Xmap + alignW);
    unsigned short *Xf1 = Xf + alignW;
    int xpos = (int)(FP_MUL * (xscale*0.5 - 0.5));
    int xinc = (int)(FP_MUL * xscale);
    int yinc = (int)(FP_MUL * yscale);      // to convert to fixed point
    unsigned int aligned_width = dstSize.width;

    // generate xmap
    for (unsigned int x = 0; x < dstSize.width; x++, xpos += xinc)
    {
        int xf;
        int xmap = (xpos >> FP_BITS);
        if (xmap >= (int)(srcSize.width - 8)){
            aligned_width = x;
        }
        if (xmap >= (int)(srcSize.width - 1)){
            Xmap[x] = (chnFormat == RPPI_CHN_PLANAR)? (srcSize.width - 1):(srcSize.width - 1)*3;
        }
        else
            Xmap[x] = (xmap<0)? 0: (chnFormat == RPPI_CHN_PLANAR)? xmap: xmap*3;
        xf = ((xpos & 0xffff) + 0x80) >> 8;
        Xf[x] = xf;
        Xf1[x] = (0x100 - xf);
    }
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;
        for (int c = 0; c < channel; c++)
        {
            int dstride = dstSize.width;
            int sstride = srcSize.width;
            Rpp8u *pSrcBorder = srcPtrTemp + (srcSize.height*sstride);    // points to the last pixel

            int ypos = (int)(FP_MUL * (yscale*0.5 - 0.5));
            for (int y = 0; y < (int)dstSize.height; y++, ypos += yinc)
            {
                int ym, fy, fy1;
                Rpp8u *pSrc1, *pSrc2;
                Rpp8u *pdst = dstPtrTemp + y*dstride;

                ym = (ypos >> FP_BITS);
                fy = ((ypos & 0xffff) + 0x80) >> 8;
                fy1 = (0x100 - fy);
                if (ym > (int)(srcSize.height - 1)){
                    pSrc1 = pSrc2 = srcPtrTemp + (srcSize.height - 1)*sstride;
                }
                else
                {
                    pSrc1 = (ym<0)? srcPtrTemp : (srcPtr + ym*sstride);
                    pSrc2 = pSrc1 + sstride;
                }
                for (int x=0; x < dstSize.width; x++) {
                    int result;
                    const unsigned char *p0 = pSrc1 + Xmap[x];
                    const unsigned char *p01 = p0 + channel;
                    const unsigned char *p1 = pSrc2 + Xmap[x];
                    const unsigned char *p11 = p1 + channel;
                    if (p0 > pSrcBorder) p0 = pSrcBorder;
                    if (p1 > pSrcBorder) p1 = pSrcBorder;
                    if (p01 > pSrcBorder) p01 = pSrcBorder;
                    if (p11 > pSrcBorder) p11 = pSrcBorder;
                    result = ((Xf1[x] * fy1*p0[0]) + (Xf[x] * fy1*p01[0]) + (Xf1[x] * fy*p1[0]) + (Xf[x] * fy*p11[0]) + 0x8000) >> 16;
                    *pdst++ = (Rpp8u) std::max(0, std::min(result, 255));
                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
            dstPtrTemp += dstSize.width * dstSize.height;
        }
    }

    else if (chnFormat == RPPI_CHN_PACKED)
    {
        aligned_width &= ~3;
        int dstride = dstSize.width * channel;
        int sstride = srcSize.width * channel;
        Rpp8u *pSrcBorder = srcPtr + (srcSize.height*sstride) - channel;    // points to the last pixel
#if __AVX2__
        const __m256i mm_zeros = _mm256_setzero_si256();
        const __m256i mm_round = _mm256_set1_epi32((int)0x80);
        const __m256i pmask1 = _mm256_setr_epi32(0, 1, 4, 2, 3, 6, 5, 7);
        const __m256i pmask2 = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
#endif
        int ypos = (int)(FP_MUL * (yscale*0.5 - 0.5));
        for (int y = 0; y < (int)dstSize.height; y++, ypos += yinc)
        {
            int ym, fy, fy1;
            Rpp8u *pSrc1, *pSrc2;
            Rpp8u *pdst = dstPtrTemp + y*dstride;

            ym = (ypos >> FP_BITS);
            fy = ((ypos & 0xffff) + 0x80) >> 8;
            fy1 = (0x100 - fy);
            if (ym > (int)(srcSize.height - 1)){
                pSrc1 = pSrc2 = srcPtrTemp + (srcSize.height - 1)*sstride;
            }
            else
            {
                pSrc1 = (ym<0)? srcPtrTemp : (srcPtrTemp + ym*sstride);
                pSrc2 = pSrc1 + sstride;
            }
            unsigned int x = 0;
#if __AVX2__

            __m256i w_y = _mm256_setr_epi32(fy1, fy, fy1, fy, fy1, fy, fy1, fy);
            __m256i p01, p23, ps01, ps23, px0, px1, ps2, ps3;
            for (; x < aligned_width; x += 4)
            {
                // load 2 pixels each
                M256I(p01).m256i_i128[0] = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x]]);
                M256I(p23).m256i_i128[0] = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x+1]]);
                M256I(ps01).m256i_i128[0] = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x]]);
                M256I(ps23).m256i_i128[0] = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x+1]]);

                M256I(p01).m256i_i128[1] = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x+2]]);
                M256I(p23).m256i_i128[1] = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x+3]]);
                M256I(ps01).m256i_i128[1] = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x+2]]);
                M256I(ps23).m256i_i128[1] = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x+3]]);

                // unpcklo for p01 and ps01
                p01 = _mm256_unpacklo_epi8(p01, ps01);
                p23 = _mm256_unpacklo_epi8(p23, ps23);
                p01 = _mm256_unpacklo_epi16(p01, _mm256_srli_si256(p01, 6));     //R0R1R2R3 G0G1G2G3 B0B1B2B3 XXXX for 1st and 3rd pixel
                p23 = _mm256_unpacklo_epi16(p23, _mm256_srli_si256(p23, 6));      //R0R1R2R3 G0G1G2G3 B0B1B2B3 XXXX for 2nd and 4th pixel
                p01 = _mm256_permutevar8x32_epi32(p01, pmask2); // R0,G0, B0, R1, G1, B1, xx, xx
                p23 = _mm256_permutevar8x32_epi32(p23, pmask2); // R2, G2, B2, R3, G3, B3, xx, xx,

                // load and get the weights in place
                ps01 = _mm256_setr_epi32(Xf1[x], Xf1[x], Xf[x], Xf[x], Xf1[x+2], Xf1[x+2], Xf[x+2], Xf[x+2]);            // xfxfxf1xf1
                ps23 = _mm256_setr_epi32(Xf1[x+1], Xf1[x+1], Xf[x+1], Xf[x+1], Xf1[x+3], Xf1[x+3], Xf[x+3], Xf[x+3]);
                ps01 = _mm256_mullo_epi32(ps01, w_y);                      // W0W1W2W3 for 1st and 3rd
                ps23 = _mm256_mullo_epi32(ps23, w_y);                      // W0W1W2W3 for 2nd and 4th
                ps01 = _mm256_srli_epi32(ps01, 8);                 // convert to 16bit
                ps23 = _mm256_srli_epi32(ps23, 8);                 // convert to 16bit
                ps01 = _mm256_packus_epi32(ps01, ps01);                 // pack to 16bit (w0w1w2w3(0), (w0w1w2w3(2), w0w1w2w3(0), (w0w1w2w3(2)))
                ps23 = _mm256_packus_epi32(ps23, ps23);                 // pack to 16bit (w0w1w2w3(1), (w0w1w2w3(3), w0w1w2w3(1), (w0w1w2w3(3))
                ps01 = _mm256_permute4x64_epi64(ps01, 0xe0);            // (w0w1w2w3(0), (w0w1w2w3(0), w0w1w2w3(0), w0w1w2w3(2)
                ps23 = _mm256_permute4x64_epi64(ps23, 0xe0);            // (w0w1w2w3(1), (w0w1w2w3(1), w0w1w2w3(1), w0w1w2w3(3)
                ps2  = _mm256_permute4x64_epi64(ps01, 0xff);            // (w0w1w2w3(2), (w0w1w2w3(2), w0w1w2w3(2), w0w1w2w3(2)
                ps3  = _mm256_permute4x64_epi64(ps23, 0xff);            // (w0w1w2w3(3), (w0w1w2w3(3), w0w1w2w3(3), w0w1w2w3(3)

                // get pixels in place for interpolation
                px0 = _mm256_unpacklo_epi8(p01, mm_zeros);        // R0R1R2R3(0), G0G1G2G3(0), B0B1B2B3(0), xxxxx
                p01 = _mm256_unpackhi_epi8(p01, mm_zeros);        // R0R1R2R3(2), G0G1G2G3(2), B0B1B2B3(2), xxxxx
                px1 = _mm256_unpacklo_epi8(p23, mm_zeros);        // R0R1R2R3(1), G0G1G2G3(1), B0B1B2B3(1), xxxxx
                p23 = _mm256_unpackhi_epi8(p23, mm_zeros);        // R0R1R2R3(3), G0G1G2G3(3), B0B1B2B3(3), xxxxx

                px0 = _mm256_madd_epi16(px0, ps01);                  // pix0: (W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3), (W0*B0+W1*B1), (W2*B2+W3*B3), xx, xx
                px1 = _mm256_madd_epi16(px1, ps23);                  // pix1: (W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3), (W0*B0+W1*B1), (W2*B2+W3*B3), xx, xx
                p01 = _mm256_madd_epi16(p01, ps2);                  // pix2: (W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3), (W0*B0+W1*B1), (W2*B2+W3*B3), xx, xx
                p23 = _mm256_madd_epi16(p23, ps3);                  // pix3: (W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3), (W0*B0+W1*B1), (W2*B2+W3*B3), xx, xx
                px0 = _mm256_hadd_epi32(px0, px1);      // R0,G0, R1, G1, B0, xx, B1, xx (32bit)
                p01 = _mm256_hadd_epi32(p01, p23);      // R2,G2, R3, G3, B2, xx, B3, xx (32bit)
                px0 = _mm256_permutevar8x32_epi32(px0, pmask1); // R0,G0, B0, R1, G1, B1, xx, xx
                p01 = _mm256_permutevar8x32_epi32(p01, pmask1); // R2, G2, B2, R3, G3, B3, xx, xx,
                px0 = _mm256_add_epi32(px0, mm_round);
                p01 = _mm256_add_epi32(p01, mm_round);
                px0 = _mm256_srli_epi32(px0, 8);      // /256
                p01 = _mm256_srli_epi32(p01, 8);      // /256
                px0 = _mm256_packus_epi32(px0, p01); //R0G0B0R1 R2G2B2R3 G1B1xx B3G3xx
                p01 = _mm256_permutevar8x32_epi32(px0, pmask1); //R0G0B0R1G1B1R2G2B2R3B3G3xxxx
                px0 = _mm256_permute4x64_epi64(p01, 0xaa);
                px0 = _mm256_packus_epi16(p01, px0); //R0G0B0R1G1B1R2G2B2R3G3B3xxxx ....
                _mm_storeu_si128((__m128i *)pdst, M256I(px0).m256i_i128[0]);      // write 12 bytes
                pdst += 12;
            }
#endif
            for (; x < dstSize.width; x++) {
                int result;
                const unsigned char *p0 = pSrc1 + Xmap[x];
                const unsigned char *p01 = p0 + channel;
                const unsigned char *p1 = pSrc2 + Xmap[x];
                const unsigned char *p11 = p1 + channel;
                if (p0 > pSrcBorder) p0 = pSrcBorder;
                if (p1 > pSrcBorder) p1 = pSrcBorder;
                if (p01 > pSrcBorder) p01 = pSrcBorder;
                if (p11 > pSrcBorder) p11 = pSrcBorder;
                result = ((Xf1[x] * fy1*p0[0]) + (Xf[x] * fy1*p01[0]) + (Xf1[x] * fy*p1[0]) + (Xf[x] * fy*p11[0]) + 0x8000) >> 16;
                *pdst++ = (Rpp8u) std::max(0, std::min(result, 255));
                result = ((Xf1[x] * fy1*p0[1]) + (Xf[x] * fy1*p01[1]) + (Xf1[x] * fy*p1[1]) + (Xf[x] * fy*p11[1]) + 0x8000) >> 16;
                *pdst++ = (Rpp8u)std::max(0, std::min(result, 255));
                result = ((Xf1[x] * fy1*p0[2]) + (Xf[x] * fy1*p01[2]) + (Xf1[x] * fy*p1[2]) + (Xf[x] * fy*p11[2]) + 0x8000) >> 16;
                *pdst++ = (Rpp8u)std::max(0, std::min(result, 255));
            }
#if CHECK_SIMD
            Rpp8u *pdstRef =  dstPtrTemp + y*dstride;
            Rpp8u *tmpRef = tmpBuff + y*dstride;
            for (x=0; x < dstSize.width; x++) {
                int result;
                const unsigned char *p0 = pSrc1 + Xmap[x];
                const unsigned char *p01 = p0 + channel;
                const unsigned char *p1 = pSrc2 + Xmap[x];
                const unsigned char *p11 = p1 + channel;
                if (p0 > pSrcBorder) p0 = pSrcBorder;
                if (p1 > pSrcBorder) p1 = pSrcBorder;
                if (p01 > pSrcBorder) p01 = pSrcBorder;
                if (p11 > pSrcBorder) p11 = pSrcBorder;
                result = ((Xf1[x] * fy1*p0[0]) + (Xf[x] * fy1*p01[0]) + (Xf1[x] * fy*p1[0]) + (Xf[x] * fy*p11[0]) + 0x8000) >> 16;
                tmpRef[x*3] = (Rpp8u) std::max(0, std::min(result, 255));
                result = ((Xf1[x] * fy1*p0[1]) + (Xf[x] * fy1*p01[1]) + (Xf1[x] * fy*p1[1]) + (Xf[x] * fy*p11[1]) + 0x8000) >> 16;
                tmpRef[x*3+1] = (Rpp8u)std::max(0, std::min(result, 255));
                result = ((Xf1[x] * fy1*p0[2]) + (Xf[x] * fy1*p01[2]) + (Xf1[x] * fy*p1[2]) + (Xf[x] * fy*p11[2]) + 0x8000) >> 16;
                tmpRef[x*3+2] = (Rpp8u)std::max(0, std::min(result, 255));
                if ( y==0 && (pdstRef[x*3] != tmpRef[x*3] || pdstRef[x*3+1] != tmpRef[x*3+1] || pdstRef[x*3+2] != tmpRef[x*3+2]))
                    printf("Error: pixel mismatch at %d: %x %x %x != %x %x %x(ref)\n", x, pdstRef[x*3], pdstRef[x*3+1], pdstRef[x*3+2], tmpRef[x*3], tmpRef[x*3+1], tmpRef[x*3+2]);
            }
#endif
        }
    }
    if (Xmap) delete[] Xmap;
#if CHECK_SIMD
    delete [] tmpBuff;
#endif

    return RPP_SUCCESS;
}
#endif

#endif //RPP_CPU_COMMON_H
