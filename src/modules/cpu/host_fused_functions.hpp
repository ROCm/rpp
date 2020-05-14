#ifndef HOST_FUSED_FUNCTIONS_H
#define HOST_FUSED_FUNCTIONS_H

#include "cpu/rpp_cpu_simd.hpp"
#include <cpu/rpp_cpu_common.hpp>

/**************** color_twist ***************/

template <typename T>
RppStatus color_twist_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                         Rpp32f *batch_alpha, Rpp32f *batch_beta, 
                         Rpp32f *batch_hueShift, Rpp32f *batch_saturationFactor, 
                         RppiROI *roiPoints, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp64u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f hueShift = batch_hueShift[batchCount];
            Rpp32f saturationFactor = batch_saturationFactor[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];
            
            color_twist_host(srcPtr, batch_srcSizeMax[batchCount], dstPtr, alpha, beta, hueShift, saturationFactor, chnFormat, channel);

        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp64u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f hueShift = batch_hueShift[batchCount];
            Rpp32f saturationFactor = batch_saturationFactor[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            color_twist_host(srcPtrImage, batch_srcSizeMax[batchCount], dstPtrImage, alpha, beta, hueShift, saturationFactor, chnFormat, channel);
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus color_twist_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f alpha, Rpp32f beta, 
                    Rpp32f hueShift, Rpp32f saturationFactor,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f hueShiftAngle = hueShift;
    while (hueShiftAngle > 360)
    {
        hueShiftAngle = hueShiftAngle - 360;
    }
    while (hueShiftAngle < 0)
    {
        hueShiftAngle = 360 + hueShiftAngle;
    }

    hueShift = hueShiftAngle / 360;
    
    Rpp64u totalImageDim = channel * srcSize.height * srcSize.width;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
        T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

        Rpp64u imageDim = srcSize.height * srcSize.width;
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + (imageDim);
        srcPtrTempB = srcPtr + (2 * imageDim);
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + (imageDim);
        dstPtrTempB = dstPtr + (2 * imageDim);

        Rpp64u bufferLength = srcSize.height * srcSize.width;
        Rpp64u alignedLength = bufferLength & ~3;

        __m128i const zero = _mm_setzero_si128();
        __m128 pZeros = _mm_set1_ps(0.0);
        __m128 pOnes = _mm_set1_ps(1.0);
        __m128 pFactor = _mm_set1_ps(255.0);
        __m128 pHueShift = _mm_set1_ps(hueShift);
        __m128 pSaturationFactor = _mm_set1_ps(saturationFactor);
        __m128i px0, px1, px2;
        __m128 xR, xG, xB;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;
        __m128 h0, h1, h2, h3;
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        h0 = _mm_set1_ps(1.0);

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp8u arrayR[4];
        Rpp8u arrayG[4];
        Rpp8u arrayB[4];

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
        {
            px0 =  _mm_loadu_si128((__m128i *)srcPtrTempR);
            px1 =  _mm_loadu_si128((__m128i *)srcPtrTempG);
            px2 =  _mm_loadu_si128((__m128i *)srcPtrTempB);

            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
            px1 = _mm_unpacklo_epi8(px1, zero);    // pixels 0-7
            px2 = _mm_unpacklo_epi8(px2, zero);    // pixels 0-7

            xR = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
            xG = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 0-3
            xB = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px2, zero));    // pixels 0-3

            xR = _mm_div_ps(xR, pFactor);
            xG = _mm_div_ps(xG, pFactor);
            xB = _mm_div_ps(xB, pFactor);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]
                
            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Modify Hue Values and re-normalize H to fraction
            xG = _mm_add_ps(xG, pHueShift);
            
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);
            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));

            xG = _mm_sub_ps(xG, xH);

            // Modify Saturation Values
            xS = _mm_mul_ps(xS, pSaturationFactor);
            xS = _mm_min_ps(xS, pOnes);
            xS = _mm_max_ps(xS, pZeros);

            _MM_TRANSPOSE4_PS (h0, xG, xS, xV);

            __m128 h1, h2, h3;

            h1 = xG;
            h2 = xS;
            h3 = xV;

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            _MM_TRANSPOSE4_PS (x0, x1, x2, x3);

            x1 = _mm_mul_ps(x1, pFactor);
            x2 = _mm_mul_ps(x2, pFactor);
            x3 = _mm_mul_ps(x3, pFactor);

            x1 = _mm_mul_ps(x1, pMul);
            x2 = _mm_mul_ps(x2, pMul);
            x3 = _mm_mul_ps(x3, pMul);

            x1 = _mm_add_ps(x1, pAdd);
            x2 = _mm_add_ps(x2, pAdd);
            x3 = _mm_add_ps(x3, pAdd);

            px0 = _mm_cvtps_epi32(x1);
            px1 = _mm_cvtps_epi32(x2);
            px2 = _mm_cvtps_epi32(x3);

            px0 = _mm_packs_epi32(px0, px0);
            px0 = _mm_packus_epi16(px0, px0);
            *((int*)arrayR) = _mm_cvtsi128_si32(px0);

            px1 = _mm_packs_epi32(px1, px1);
            px1 = _mm_packus_epi16(px1, px1);
            *((int*)arrayG) = _mm_cvtsi128_si32(px1);

            px2 = _mm_packs_epi32(px2, px2);
            px2 = _mm_packus_epi16(px2, px2);
            *((int*)arrayB) = _mm_cvtsi128_si32(px2);

            memcpy(dstPtrTempR, arrayR, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempG, arrayG, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempB, arrayB, 4 * sizeof(Rpp8u));

            srcPtrTempR += 4;
            srcPtrTempG += 4;
            srcPtrTempB += 4;
            dstPtrTempR += 4;
            dstPtrTempG += 4;
            dstPtrTempB += 4;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            // RGB to HSV

            Rpp32f hue, sat, val;
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                hue = 0;
            }
            else if (cmax == rf)
            {
                hue = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                hue = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                hue = round(60 * (((rf - gf) / delta) + 4));
            }

            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            if (cmax == 0)
            {
                sat = 0;
            }
            else
            {
                sat = delta / cmax;
            }

            val = cmax;

            // Modify Hue and Saturation

            hue = hue + hueShiftAngle;
            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            sat *= saturationFactor;
            sat = (sat < (Rpp32f) 1) ? sat : ((Rpp32f) 1);
            sat = (sat > (Rpp32f) 0) ? sat : ((Rpp32f) 0);

            // HSV to RGB

            Rpp32f c, x, m;
            c = val * sat;
            x = c * (1 - abs(int(fmod((hue / 60), 2)) - 1));
            m = val - c;

            if ((0 <= hue) && (hue < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= hue) && (hue < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= hue) && (hue < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= hue) && (hue < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= hue) && (hue < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= hue) && (hue < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempR++;
            srcPtrTempG++;
            srcPtrTempB++;
            dstPtrTempR++;
            dstPtrTempG++;
            dstPtrTempB++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        T *srcPtrTempPx0, *srcPtrTempPx1, *srcPtrTempPx2, *srcPtrTempPx3;
        T *dstPtrTempPx0, *dstPtrTempPx1, *dstPtrTempPx2, *dstPtrTempPx3;

        srcPtrTempPx0 = srcPtr;
        srcPtrTempPx1 = srcPtr + 3;
        srcPtrTempPx2 = srcPtr + 6;
        srcPtrTempPx3 = srcPtr + 9;
        dstPtrTempPx0 = dstPtr;
        dstPtrTempPx1 = dstPtr + 3;
        dstPtrTempPx2 = dstPtr + 6;
        dstPtrTempPx3 = dstPtr + 9;

        Rpp64u bufferLength = totalImageDim;
        Rpp64u alignedLength = (bufferLength / 12) * 12;

        __m128i const zero = _mm_setzero_si128();
        __m128 pZeros = _mm_set1_ps(0.0);
        __m128 pOnes = _mm_set1_ps(1.0);
        __m128 pFactor = _mm_set1_ps(255.0);
        __m128 pHueShift = _mm_set1_ps(hueShift);
        __m128 pSaturationFactor = _mm_set1_ps(saturationFactor);
        __m128i px0, px1, px2, px3;
        __m128 xR, xG, xB, xA;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;
        __m128 h0, h1, h2, h3;
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        h0 = _mm_set1_ps(1.0);

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp8u arrayPx0[4];
        Rpp8u arrayPx1[4];
        Rpp8u arrayPx2[4];
        Rpp8u arrayPx3[4];

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
        {
            px0 = _mm_loadu_si128((__m128i *)srcPtrTempPx0);
            px1 = _mm_loadu_si128((__m128i *)srcPtrTempPx1);
            px2 = _mm_loadu_si128((__m128i *)srcPtrTempPx2);
            px3 = _mm_loadu_si128((__m128i *)srcPtrTempPx3);

            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
            px1 = _mm_unpacklo_epi8(px1, zero);    // pixels 0-7
            px2 = _mm_unpacklo_epi8(px2, zero);    // pixels 0-7
            px3 = _mm_unpacklo_epi8(px3, zero);    // pixels 0-7

            xR = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
            xG = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 0-3
            xB = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px2, zero));    // pixels 0-3
            xA = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px3, zero));    // pixels 0-3

            _MM_TRANSPOSE4_PS (xR, xG, xB, xA);

            xR = _mm_div_ps(xR, pFactor);
            xG = _mm_div_ps(xG, pFactor);
            xB = _mm_div_ps(xB, pFactor);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]
                
            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Modify Hue Values and re-normalize H to fraction
            xG = _mm_add_ps(xG, pHueShift);
            
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);
            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));

            xG = _mm_sub_ps(xG, xH);

            // Modify Saturation Values
            xS = _mm_mul_ps(xS, pSaturationFactor);
            xS = _mm_min_ps(xS, pOnes);
            xS = _mm_max_ps(xS, pZeros);

            _MM_TRANSPOSE4_PS (h0, xG, xS, xV);

            __m128 h1, h2, h3;

            h1 = xG;
            h2 = xS;
            h3 = xV;

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            x0 = _mm_shuffle_ps(x0,x0, _MM_SHUFFLE(0,3,2,1));
            x1 = _mm_shuffle_ps(x1,x1, _MM_SHUFFLE(0,3,2,1));
            x2 = _mm_shuffle_ps(x2,x2, _MM_SHUFFLE(0,3,2,1));
            x3 = _mm_shuffle_ps(x3,x3, _MM_SHUFFLE(0,3,2,1));

            x0 = _mm_mul_ps(x0, pFactor);
            x1 = _mm_mul_ps(x1, pFactor);
            x2 = _mm_mul_ps(x2, pFactor);
            x3 = _mm_mul_ps(x3, pFactor);

            x0 = _mm_mul_ps(x0, pMul);
            x1 = _mm_mul_ps(x1, pMul);
            x2 = _mm_mul_ps(x2, pMul);
            x3 = _mm_mul_ps(x3, pMul);

            x0 = _mm_add_ps(x0, pAdd);
            x1 = _mm_add_ps(x1, pAdd);
            x2 = _mm_add_ps(x2, pAdd);
            x3 = _mm_add_ps(x3, pAdd);

            px0 = _mm_cvtps_epi32(x0);
            px1 = _mm_cvtps_epi32(x1);
            px2 = _mm_cvtps_epi32(x2);
            px3 = _mm_cvtps_epi32(x3);

            px0 = _mm_packs_epi32(px0, px0);
            px0 = _mm_packus_epi16(px0, px0);
            *((int*)arrayPx0) = _mm_cvtsi128_si32(px0);

            px1 = _mm_packs_epi32(px1, px1);
            px1 = _mm_packus_epi16(px1, px1);
            *((int*)arrayPx1) = _mm_cvtsi128_si32(px1);

            px2 = _mm_packs_epi32(px2, px2);
            px2 = _mm_packus_epi16(px2, px2);
            *((int*)arrayPx2) = _mm_cvtsi128_si32(px2);

            px3 = _mm_packs_epi32(px3, px3);
            px3 = _mm_packus_epi16(px3, px3);
            *((int*)arrayPx3) = _mm_cvtsi128_si32(px3);

            memcpy(dstPtrTempPx0, arrayPx0, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempPx1, arrayPx1, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempPx2, arrayPx2, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempPx3, arrayPx3, 4 * sizeof(Rpp8u));

            srcPtrTempPx0 += 12;
            srcPtrTempPx1 += 12;
            srcPtrTempPx2 += 12;
            srcPtrTempPx3 += 12;
            dstPtrTempPx0 += 12;
            dstPtrTempPx1 += 12;
            dstPtrTempPx2 += 12;
            dstPtrTempPx3 += 12;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

            srcPtrTempR = srcPtrTempPx0;
            srcPtrTempG = srcPtrTempPx0 + 1;
            srcPtrTempB = srcPtrTempPx0 + 2;
            dstPtrTempR = dstPtrTempPx0;
            dstPtrTempG = dstPtrTempPx0 + 1;
            dstPtrTempB = dstPtrTempPx0 + 2;

            // RGB to HSV

            Rpp32f hue, sat, val;
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                hue = 0;
            }
            else if (cmax == rf)
            {
                hue = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                hue = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                hue = round(60 * (((rf - gf) / delta) + 4));
            }

            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            if (cmax == 0)
            {
                sat = 0;
            }
            else
            {
                sat = delta / cmax;
            }

            val = cmax;

            // Modify Hue and Saturation

            hue = hue + hueShiftAngle;
            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            sat *= saturationFactor;
            sat = (sat < (Rpp32f) 1) ? sat : ((Rpp32f) 1);
            sat = (sat > (Rpp32f) 0) ? sat : ((Rpp32f) 0);

            // HSV to RGB

            Rpp32f c, x, m;
            c = val * sat;
            x = c * (1 - abs(int(fmod((hue / 60), 2)) - 1));
            m = val - c;

            if ((0 <= hue) && (hue < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= hue) && (hue < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= hue) && (hue < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= hue) && (hue < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= hue) && (hue < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= hue) && (hue < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempPx0 += 3;
            dstPtrTempPx0 += 3;
        }
    }

    return RPP_SUCCESS;
}

/**************** crop_mirror ***************/

template <typename T, typename U>
RppStatus crop_mirror_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, 
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32u *batch_mirrorFlag, Rpp32u outputFormatToggle,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            if (mirrorFlag == 0)
            {
                for(int c = 0; c < channel; c++)
                {
                    T *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x1;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        memcpy(dstPtrChannel, srcPtrChannelROI, batch_dstSize[batchCount].width * sizeof(T));
                        dstPtrChannel += batch_dstSizeMax[batchCount].width;
                        srcPtrChannelROI += batch_srcSizeMax[batchCount].width;
                    }
                }
            }
            else if (mirrorFlag == 1)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width + batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    T *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x2;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        T *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);
                        
                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 16) * 16;

                        __m128i px0;
                        __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            srcPtrChannelROI -= 15;
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                            px0 = _mm_shuffle_epi8(px0, vMask);
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            srcPtrChannelROI -= 1;
                            dstPtrTemp += 16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = (T) *srcPtrChannelROI--;
                        }
                        
                        srcPtrChannelROI += srcROIIncrement;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);
                
                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);

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
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;
            
            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * batch_dstSize[batchCount].width;
            
            if (mirrorFlag == 0)
            {
                T *srcPtrImageROI, *dstPtrImageTemp;
                srcPtrImageROI = srcPtrImage + (y1 * srcElementsInRowMax) + (x1 * channel);
                dstPtrImageTemp = dstPtrImage;


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    memcpy(dstPtrImageTemp, srcPtrImageROI, elementsInRowROI * sizeof(T));
                    dstPtrImageTemp += dstElementsInRowMax;
                    srcPtrImageROI += srcElementsInRowMax;
                }
            }
            else if (mirrorFlag == 1)
            {
                T  *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + ((x2 - 1) * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax + elementsInRowROI;
                

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    T *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;
                    Rpp32u alignedLength = ((bufferLength / 15) * 15) - 1;

                    __m128i px0;
                    __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                    {
                        srcPtrROI -= 13;
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);
                        px0 = _mm_shuffle_epi8(px0, vMask);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrROI -= 2;
                        dstPtrTemp += 15;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                    {
                        memcpy(dstPtrTemp, srcPtrROI, channel * sizeof(T));
                        dstPtrTemp += channel;
                        srcPtrROI -= channel;
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);
                
                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus crop_mirror_host(T* srcPtr, RppiSize srcSize, U* dstPtr, RppiSize dstSize,
                                     Rpp32u crop_pos_x, Rpp32u crop_pos_y,
                                     Rpp32u mirrorFlag, Rpp32u outputFormatToggle,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    U *dstPtrTemp;
    dstPtrTemp = dstPtr;

    Rpp32u srcImageDim = srcSize.height * srcSize.width;

    if(chnFormat == RPPI_CHN_PLANAR)
    {
        if (mirrorFlag == 0)
        {
            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannelROI;
                srcPtrChannelROI = srcPtr + (c * srcImageDim) + (crop_pos_y * srcSize.width) + crop_pos_x;

                for(int i = 0; i < dstSize.height; i++)
                {
                    memcpy(dstPtrTemp, srcPtrChannelROI, dstSize.width * sizeof(T));
                    dstPtrTemp += dstSize.width;
                    srcPtrChannelROI += srcSize.width;
                }
            }
        }
        else if (mirrorFlag == 1)
        {
            Rpp32u srcROIIncrement = srcSize.width + dstSize.width;
            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannelROI;
                srcPtrChannelROI = srcPtr + (c * srcImageDim) + (crop_pos_y * srcSize.width) + crop_pos_x + dstSize.width - 1;
                
                for(int i = 0; i < dstSize.height; i++)
                {
                    Rpp32u bufferLength = dstSize.width;
                    Rpp32u alignedLength = (bufferLength / 16) * 16;

                    __m128i px0;
                    __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        srcPtrChannelROI -= 15;
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                        px0 = _mm_shuffle_epi8(px0, vMask);

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrChannelROI -= 1;
                        dstPtrTemp += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = *srcPtrChannelROI--;
                    }

                    srcPtrChannelROI += srcROIIncrement;
                }
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u srcElementsInRow = channel * srcSize.width;
        Rpp32u dstElementsInRow = channel * dstSize.width;

        if (mirrorFlag == 0)
        {
            T *srcPtrROI;
            srcPtrROI = srcPtr + (crop_pos_y * srcElementsInRow) + (crop_pos_x * channel);

            for(int i = 0; i < dstSize.height; i++)
            {
                memcpy(dstPtrTemp, srcPtrROI, dstElementsInRow * sizeof(T));
                dstPtrTemp += dstElementsInRow;
                srcPtrROI += srcElementsInRow;
            }
        }
        else if (mirrorFlag == 1)
        {
            T  *srcPtrROI;
            srcPtrROI = srcPtr + (crop_pos_y * srcElementsInRow) + ((crop_pos_x + dstSize.width - 1) * channel);

            Rpp32u srcROIIncrement = srcElementsInRow + dstElementsInRow;
            
            for(int i = 0; i < dstSize.height; i++)
            {
                Rpp32u bufferLength = dstElementsInRow;
                Rpp32u alignedLength = ((bufferLength / 15) * 15) - 1;

                __m128i px0;
                __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                {
                    srcPtrROI -= 13;
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);
                    px0 = _mm_shuffle_epi8(px0, vMask);

                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrROI -= 2;
                    dstPtrTemp += 15;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                {
                    memcpy(dstPtrTemp, srcPtrROI, channel * sizeof(T));
                    dstPtrTemp += channel;
                    srcPtrROI -= channel;
                }

                srcPtrROI += srcROIIncrement;
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** crop_mirror_normalize ***************/

template <typename T, typename U>
RppStatus crop_mirror_normalize_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, 
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32f *batch_mean, Rpp32f *batch_stdDev,
                                     Rpp32u *batch_mirrorFlag, Rpp32u outputFormatToggle,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = batch_mean[batchCount];
            Rpp32f stdDev = batch_stdDev[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            if (mirrorFlag == 0)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width - batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    T *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x1;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        T *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);
                        
                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 16) * 16;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i px0, px1, px2, px3;
                        __m128 p0, p1, p2, p3;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            p0 = _mm_sub_ps(p0, vMean);
                            p1 = _mm_sub_ps(p1, vMean);
                            p2 = _mm_sub_ps(p2, vMean);
                            p3 = _mm_sub_ps(p3, vMean);
                            px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                            px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                            px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                            px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));
                            
                            px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                            px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            srcPtrChannelROI += 16;
                            dstPtrTemp += 16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrChannelROI) - mean) / stdDev);
                            dstPtrTemp++;
                            srcPtrChannelROI++;
                        }
                        
                        srcPtrChannelROI += srcROIIncrement;

                    }
                }
            }
            else if (mirrorFlag == 1)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width + batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    T *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x2;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        T *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);
                        
                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 16) * 16;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i px0, px1, px2, px3;
                        __m128 p0, p1, p2, p3;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);
                        __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            srcPtrChannelROI -= 15;
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                            px0 = _mm_shuffle_epi8(px0, vMask);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            p0 = _mm_sub_ps(p0, vMean);
                            p1 = _mm_sub_ps(p1, vMean);
                            p2 = _mm_sub_ps(p2, vMean);
                            p3 = _mm_sub_ps(p3, vMean);
                            px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                            px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                            px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                            px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));
                            
                            px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                            px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            srcPtrChannelROI -= 1;
                            dstPtrTemp += 16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrChannelROI) - mean) / stdDev);
                            dstPtrTemp++;
                            srcPtrChannelROI--;
                        }
                        
                        srcPtrChannelROI += srcROIIncrement;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);
                
                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);

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
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;
            
            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = 0;//batch_mean[batchCount];
            Rpp32f stdDev = 1; //batch_stdDev[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * batch_dstSize[batchCount].width;
            
            if (mirrorFlag == 0)
            {
                T  *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + (x1 * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax - elementsInRowROI;
                

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    T *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;
                    Rpp32u alignedLength = (bufferLength / 16) * 16;

                    __m128i const zero = _mm_setzero_si128();
                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1, p2, p3;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        p0 = _mm_sub_ps(p0, vMean);
                        p1 = _mm_sub_ps(p1, vMean);
                        p2 = _mm_sub_ps(p2, vMean);
                        p3 = _mm_sub_ps(p3, vMean);
                        px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                        px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                        px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                        px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));
                        
                        px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                        px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrROI += 16;
                        dstPtrTemp += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrROI) - mean) / stdDev);
                        dstPtrTemp++;
                        srcPtrROI++;
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            else if (mirrorFlag == 1)
            {
                T  *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + ((x2 - 1) * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax + elementsInRowROI;
                

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    T *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;
                    Rpp32u alignedLength = ((bufferLength / 15) * 15) - 1;

                    __m128i const zero = _mm_setzero_si128();
                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1, p2, p3;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);
                    __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                    {
                        srcPtrROI -= 13;
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);
                        px0 = _mm_shuffle_epi8(px0, vMask);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        p0 = _mm_sub_ps(p0, vMean);
                        p1 = _mm_sub_ps(p1, vMean);
                        p2 = _mm_sub_ps(p2, vMean);
                        p3 = _mm_sub_ps(p3, vMean);
                        px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                        px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                        px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                        px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));
                        
                        px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                        px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrROI -= 2;
                        dstPtrTemp += 15;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrROI) - mean) / stdDev);
                            dstPtrTemp++;
                            srcPtrROI++;
                        }
                        srcPtrROI -= (2 * channel);
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);
                
                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus crop_mirror_normalize_host(T* srcPtr, RppiSize srcSize, U* dstPtr, RppiSize dstSize,
                                     Rpp32u crop_pos_x, Rpp32u crop_pos_y,
                                     Rpp32f mean, Rpp32f stdDev,
                                     Rpp32u mirrorFlag, Rpp32u outputFormatToggle,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    U *dstPtrTemp;
    dstPtrTemp = dstPtr;

    Rpp32u srcImageDim = srcSize.height * srcSize.width;

    if(chnFormat == RPPI_CHN_PLANAR)
    {
        if (mirrorFlag == 0)
        {
            Rpp32u srcROIIncrement = srcSize.width - dstSize.width;
            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannelROI;
                srcPtrChannelROI = srcPtr + (c * srcImageDim) + (crop_pos_y * srcSize.width) + crop_pos_x;
                
                for(int i = 0; i < dstSize.height; i++)
                {
                    Rpp32u bufferLength = dstSize.width;
                    Rpp32u alignedLength = (bufferLength / 16) * 16;

                    __m128i const zero = _mm_setzero_si128();
                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1, p2, p3;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        p0 = _mm_sub_ps(p0, vMean);
                        p1 = _mm_sub_ps(p1, vMean);
                        p2 = _mm_sub_ps(p2, vMean);
                        p3 = _mm_sub_ps(p3, vMean);
                        px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                        px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                        px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                        px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));
                        
                        px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                        px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrChannelROI += 16;
                        dstPtrTemp += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrChannelROI) - mean) / stdDev);
                        dstPtrTemp++;
                        srcPtrChannelROI++;
                    }

                    srcPtrChannelROI += srcROIIncrement;
                }
            }
        }
        else if (mirrorFlag == 1)
        {
            Rpp32u srcROIIncrement = srcSize.width + dstSize.width;
            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannelROI;
                srcPtrChannelROI = srcPtr + (c * srcImageDim) + (crop_pos_y * srcSize.width) + crop_pos_x + dstSize.width - 1;
                
                for(int i = 0; i < dstSize.height; i++)
                {
                    Rpp32u bufferLength = dstSize.width;
                    Rpp32u alignedLength = (bufferLength / 16) * 16;

                    __m128i const zero = _mm_setzero_si128();
                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1, p2, p3;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);
                    __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        srcPtrChannelROI -= 15;
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                        px0 = _mm_shuffle_epi8(px0, vMask);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        p0 = _mm_sub_ps(p0, vMean);
                        p1 = _mm_sub_ps(p1, vMean);
                        p2 = _mm_sub_ps(p2, vMean);
                        p3 = _mm_sub_ps(p3, vMean);
                        px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                        px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                        px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                        px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));
                        
                        px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                        px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrChannelROI -= 1;
                        dstPtrTemp += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrChannelROI) - mean) / stdDev);
                        dstPtrTemp++;
                        srcPtrChannelROI--;
                    }

                    srcPtrChannelROI += srcROIIncrement;
                }
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u srcElementsInRow = channel * srcSize.width;
        Rpp32u dstElementsInRow = channel * dstSize.width;

        if (mirrorFlag == 0)
        {
            T  *srcPtrROI;
            srcPtrROI = srcPtr + (crop_pos_y * srcElementsInRow) + (crop_pos_x * channel);

            Rpp32u srcROIIncrement = srcElementsInRow - dstElementsInRow;
            
            for(int i = 0; i < dstSize.height; i++)
            {
                Rpp32u bufferLength = dstElementsInRow;
                Rpp32u alignedLength = (bufferLength / 16) * 16;

                __m128i const zero = _mm_setzero_si128();
                __m128i px0, px1, px2, px3;
                __m128 p0, p1, p2, p3;
                __m128 vMean = _mm_set1_ps(mean);
                __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);

                    px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                    px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                    p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                    p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                    p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                    p0 = _mm_sub_ps(p0, vMean);
                    p1 = _mm_sub_ps(p1, vMean);
                    p2 = _mm_sub_ps(p2, vMean);
                    p3 = _mm_sub_ps(p3, vMean);
                    px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                    px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                    px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                    px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));
                    
                    px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                    px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                    px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrROI += 16;
                    dstPtrTemp += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrROI) - mean) / stdDev);
                    dstPtrTemp++;
                    srcPtrROI++;
                }

                srcPtrROI += srcROIIncrement;
            }
        }
        else if (mirrorFlag == 1)
        {
            T  *srcPtrROI;
            srcPtrROI = srcPtr + (crop_pos_y * srcElementsInRow) + ((crop_pos_x + dstSize.width - 1) * channel);

            Rpp32u srcROIIncrement = srcElementsInRow + dstElementsInRow;
            
            for(int i = 0; i < dstSize.height; i++)
            {
                Rpp32u bufferLength = dstElementsInRow;
                Rpp32u alignedLength = ((bufferLength / 15) * 15) - 1;

                __m128i const zero = _mm_setzero_si128();
                __m128i px0, px1, px2, px3;
                __m128 p0, p1, p2, p3;
                __m128 vMean = _mm_set1_ps(mean);
                __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);
                __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                {
                    srcPtrROI -= 13;
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);
                    px0 = _mm_shuffle_epi8(px0, vMask);

                    px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                    px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                    p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                    p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                    p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                    p0 = _mm_sub_ps(p0, vMean);
                    p1 = _mm_sub_ps(p1, vMean);
                    p2 = _mm_sub_ps(p2, vMean);
                    p3 = _mm_sub_ps(p3, vMean);
                    px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                    px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                    px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                    px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));
                    
                    px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                    px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                    px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrROI -= 2;
                    dstPtrTemp += 15;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                {
                    for(int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrROI) - mean) / stdDev);
                        dstPtrTemp++;
                        srcPtrROI++;
                    }
                    srcPtrROI -= (2 * channel);
                }

                srcPtrROI += srcROIIncrement;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_f32_host_batch(Rpp32f* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp32f* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, 
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32f *batch_mean, Rpp32f *batch_stdDev,
                                     Rpp32u *batch_mirrorFlag, Rpp32u outputFormatToggle,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = 0; //batch_mean[batchCount];      // Currently outputs without normalization
            Rpp32f stdDev = 1; //batch_stdDev[batchCount];  // Currently outputs without normalization

            Rpp32f *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            if (mirrorFlag == 0)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width - batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    Rpp32f *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x1;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        Rpp32f *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);
                        
                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 4) * 4;

                        __m128 p0;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                        {
                            p0 = _mm_loadu_ps(srcPtrChannelROI);
                            p0 = _mm_sub_ps(p0, vMean);
                            p0 = _mm_mul_ps(p0, vInvStdDev);
                            _mm_storeu_ps(dstPtrTemp, p0);

                            srcPtrChannelROI += 4;
                            dstPtrTemp += 4;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (*srcPtrChannelROI - mean) / stdDev;
                            dstPtrTemp++;
                            srcPtrChannelROI++;
                        }
                        
                        srcPtrChannelROI += srcROIIncrement;

                    }
                }
            }
            else if (mirrorFlag == 1)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width + batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    Rpp32f *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x2;

                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        Rpp32f *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);
                        
                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 4) * 4;

                        __m128 p0;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                        {
                            srcPtrChannelROI -= 3;
                            p0 = _mm_loadu_ps(srcPtrChannelROI);
                            p0 = _mm_shuffle_ps(p0, p0, _MM_SHUFFLE(0,1,2,3));
                            p0 = _mm_sub_ps(p0, vMean);
                            p0 = _mm_mul_ps(p0, vInvStdDev);
                            _mm_storeu_ps(dstPtrTemp, p0);

                            srcPtrChannelROI -= 1;
                            dstPtrTemp += 4;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (*srcPtrChannelROI - mean) / stdDev;
                            dstPtrTemp++;
                            srcPtrChannelROI--;
                        }
                        
                        srcPtrChannelROI += srcROIIncrement;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                // Add
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;
            
            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = 0; //batch_mean[batchCount];      // Currently outputs without normalization
            Rpp32f stdDev = 1; //batch_stdDev[batchCount];  // Currently outputs without normalization
            
            Rpp32f *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * batch_dstSize[batchCount].width;
            
            if (mirrorFlag == 0)
            {
                Rpp32f *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + (x1 * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax - elementsInRowROI;
                

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp32f *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;
                    Rpp32u alignedLength = (bufferLength / 4) * 4;

                    __m128 p0;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        p0 = _mm_loadu_ps(srcPtrROI);
                        p0 = _mm_sub_ps(p0, vMean);
                        p0 = _mm_mul_ps(p0, vInvStdDev);
                        _mm_storeu_ps(dstPtrTemp, p0);

                        srcPtrROI += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (*srcPtrROI - mean) / stdDev;
                        dstPtrTemp++;
                        srcPtrROI++;
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            else if (mirrorFlag == 1)
            {
                Rpp32f *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + ((x2 - 1) * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax + elementsInRowROI;
                

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp32f *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;

                    __m128 p0;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < 3; vectorLoopCount+=3)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp = (*srcPtrROI - mean) / stdDev;
                            dstPtrTemp++;
                            srcPtrROI++;
                        }
                        srcPtrROI -= (2 * channel);
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        p0 = _mm_loadu_ps(srcPtrROI);
                        p0 = _mm_sub_ps(p0, vMean);
                        p0 = _mm_mul_ps(p0, vInvStdDev);
                        _mm_storeu_ps(dstPtrTemp, p0);

                        srcPtrROI -= 3;
                        dstPtrTemp += 3;
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            if (outputFormatToggle == 1)
            {
                // Add
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_f32_host(Rpp32f* srcPtr, RppiSize srcSize, Rpp32f* dstPtr, RppiSize dstSize,
                                         Rpp32u crop_pos_x, Rpp32u crop_pos_y,
                                         Rpp32f mean, Rpp32f stdDev,
                                         Rpp32u mirrorFlag, Rpp32u outputFormatToggle,
                                         RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f *dstPtrTemp;
    dstPtrTemp = dstPtr;

    Rpp32u srcImageDim = srcSize.height * srcSize.width;

    if(chnFormat == RPPI_CHN_PLANAR)
    {
        if (mirrorFlag == 0)
        {
            Rpp32u srcROIIncrement = srcSize.width - dstSize.width;

            Rpp32u bufferLength = dstSize.width;
            Rpp32u alignedLength = (bufferLength / 4) * 4;

            __m128 p0;
            __m128 vMean = _mm_set1_ps(mean);
            __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);

            for(int c = 0; c < channel; c++)
            {
                Rpp32f *srcPtrChannelROI;
                srcPtrChannelROI = srcPtr + (c * srcImageDim) + (crop_pos_y * srcSize.width) + crop_pos_x;
                
                for(int i = 0; i < dstSize.height; i++)
                {
                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        p0 = _mm_loadu_ps(srcPtrChannelROI);
                        p0 = _mm_sub_ps(p0, vMean);
                        p0 = _mm_mul_ps(p0, vInvStdDev);
                        _mm_storeu_ps(dstPtrTemp, p0);

                        srcPtrChannelROI += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (*srcPtrChannelROI - mean) / stdDev;
                        dstPtrTemp++;
                        srcPtrChannelROI++;
                    }

                    srcPtrChannelROI += srcROIIncrement;
                }
            }
        }
        else if (mirrorFlag == 1)
        {
            Rpp32u srcROIIncrement = srcSize.width + dstSize.width;
            for(int c = 0; c < channel; c++)
            {
                Rpp32f *srcPtrChannelROI;
                srcPtrChannelROI = srcPtr + (c * srcImageDim) + (crop_pos_y * srcSize.width) + crop_pos_x + dstSize.width - 1;
                
                for(int i = 0; i < dstSize.height; i++)
                {
                    Rpp32u bufferLength = dstSize.width;
                    Rpp32u alignedLength = (bufferLength / 4) * 4;

                    __m128 p0;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        srcPtrChannelROI -= 3;
                        p0 = _mm_loadu_ps(srcPtrChannelROI);
                        p0 = _mm_shuffle_ps(p0, p0, _MM_SHUFFLE(0,1,2,3));
                        p0 = _mm_sub_ps(p0, vMean);
                        p0 = _mm_mul_ps(p0, vInvStdDev);
                        _mm_storeu_ps(dstPtrTemp, p0);

                        srcPtrChannelROI -= 1;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (*srcPtrChannelROI - mean) / stdDev;
                        dstPtrTemp++;
                        srcPtrChannelROI--;
                    }

                    srcPtrChannelROI += srcROIIncrement;
                }
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u srcElementsInRow = channel * srcSize.width;
        Rpp32u dstElementsInRow = channel * dstSize.width;

        if (mirrorFlag == 0)
        {
            Rpp32f *srcPtrROI;
            srcPtrROI = srcPtr + (crop_pos_y * srcElementsInRow) + (crop_pos_x * channel);

            Rpp32u srcROIIncrement = srcElementsInRow - dstElementsInRow;
            
            for(int i = 0; i < dstSize.height; i++)
            {
                Rpp32u bufferLength = dstElementsInRow;
                Rpp32u alignedLength = (bufferLength / 4) * 4;

                __m128 p0;
                __m128 vMean = _mm_set1_ps(mean);
                __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    p0 = _mm_loadu_ps(srcPtrROI);
                    p0 = _mm_sub_ps(p0, vMean);
                    p0 = _mm_mul_ps(p0, vInvStdDev);
                    _mm_storeu_ps(dstPtrTemp, p0);

                    srcPtrROI += 4;
                    dstPtrTemp += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp = (*srcPtrROI - mean) / stdDev;
                    dstPtrTemp++;
                    srcPtrROI++;
                }

                srcPtrROI += srcROIIncrement;
            }
        }
        else if (mirrorFlag == 1)
        {
            Rpp32f *srcPtrROI;
            srcPtrROI = srcPtr + (crop_pos_y * srcElementsInRow) + ((crop_pos_x + dstSize.width - 1) * channel);

            Rpp32u srcROIIncrement = srcElementsInRow + dstElementsInRow;
            
            for(int i = 0; i < dstSize.height; i++)
            {
                Rpp32u bufferLength = dstElementsInRow;
                __m128 p0;
                __m128 vMean = _mm_set1_ps(mean);
                __m128 vInvStdDev = _mm_set1_ps(1.0 / stdDev);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < 3; vectorLoopCount+=3)
                {
                    for(int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = (*srcPtrROI - mean) / stdDev;
                        dstPtrTemp++;
                        srcPtrROI++;
                    }
                    srcPtrROI -= (2 * channel);
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    p0 = _mm_loadu_ps(srcPtrROI);
                    p0 = _mm_sub_ps(p0, vMean);
                    p0 = _mm_mul_ps(p0, vInvStdDev);
                    _mm_storeu_ps(dstPtrTemp, p0);

                    srcPtrROI -= 3;
                    dstPtrTemp += 3;
                }

                srcPtrROI += srcROIIncrement;
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** crop ***************/

template <typename T, typename U>
RppStatus crop_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, 
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            for(int c = 0; c < channel; c++)
            {
                T *dstPtrChannel, *srcPtrChannelROI;
                dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x1;

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    memcpy(dstPtrChannel, srcPtrChannelROI, batch_dstSize[batchCount].width * sizeof(T));
                    dstPtrChannel += batch_dstSizeMax[batchCount].width;
                    srcPtrChannelROI += batch_srcSizeMax[batchCount].width;
                }
            }           
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;
            
            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * batch_dstSize[batchCount].width;
                
            T *srcPtrImageROI, *dstPtrImageTemp;
            srcPtrImageROI = srcPtrImage + (y1 * srcElementsInRowMax) + (x1 * channel);
            dstPtrImageTemp = dstPtrImage;

            for(int i = 0; i < batch_dstSize[batchCount].height; i++)
            {
                memcpy(dstPtrImageTemp, srcPtrImageROI, elementsInRowROI * sizeof(T));
                dstPtrImageTemp += dstElementsInRowMax;
                srcPtrImageROI += srcElementsInRowMax;
            }
        }
    }
    return RPP_SUCCESS;
}

// /**************** resize_crop_mirror ***************/

template <typename T>
RppStatus resize_crop_mirror_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, 
                                        Rpp32u *batch_x1, Rpp32u *batch_x2, Rpp32u *batch_y1, Rpp32u *batch_y2, Rpp32u *batch_mirrorFlag, 
                                        Rpp32u nbatchSize,
                                        RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u y2 = batch_y2[batchCount];
            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];

            Rpp64u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp64u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;
            
            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = RPPABS(y2 - y1) + 1;
            srcSizeROI.width = RPPABS(x2 - x1) + 1;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;
            
            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            T *dstPtrROI = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
            T *srcPtrImageTemp, *srcPtrROITemp;
            
            srcPtrROITemp = srcPtrROI;
            for (int c = 0; c < channel; c++)
            {
                srcPtrImageTemp = srcPtrImage + (c * srcImageDimMax) + ((Rpp32u) y1 * batch_srcSizeMax[batchCount].width) + (Rpp32u) x1;
                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    memcpy(srcPtrROITemp, srcPtrImageTemp, srcSizeROI.width * sizeof(T));
                    srcPtrImageTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrROITemp += srcSizeROI.width;
                }
            }

            if (mirrorFlag == 0)
            {
                resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);
            }
            else if (mirrorFlag == 1)
            {
                T *srcPtrROIMirrorred = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
                T *srcPtrROITemp2, *srcPtrROIMirrorredTemp;
                srcPtrROIMirrorredTemp = srcPtrROIMirrorred;
                Rpp32u bufferLength = srcSizeROI.width;
                Rpp32u alignedLength = (bufferLength / 16) * 16;

                for (int c = 0; c < channel; c++)
                {
                    srcPtrROITemp = srcPtrROI + (c * srcSizeROI.height * srcSizeROI.width) + srcSizeROI.width - 1;
                    for (int i = 0; i < srcSizeROI.height; i++)
                    {
                        srcPtrROITemp2 = srcPtrROITemp;

                        __m128i px0;
                        __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            srcPtrROITemp2 -= 15;
                            px0 = _mm_loadu_si128((__m128i *)srcPtrROITemp2);
                            px0 = _mm_shuffle_epi8(px0, vMask);
                            _mm_storeu_si128((__m128i *)srcPtrROIMirrorredTemp, px0);
                            srcPtrROITemp2 -= 1;
                            srcPtrROIMirrorredTemp += 16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *srcPtrROIMirrorredTemp++ = (T) *srcPtrROITemp2--;
                        }
                        srcPtrROITemp = srcPtrROITemp + srcSizeROI.width;
                    }
                }

                resize_kernel_host(srcPtrROIMirrorred, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

                free(srcPtrROIMirrorred);
            }

            compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u y2 = batch_y2[batchCount];
            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = RPPABS(y2 - y1) + 1;
            srcSizeROI.width = RPPABS(x2 - x1) + 1;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * srcSizeROI.width;
            
            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            T *dstPtrROI = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
            T *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;

            srcPtrImageTemp = srcPtrImage + ((Rpp32u) y1 * elementsInRowMax) + (channel * (Rpp32u) x1);
            for (int i = 0; i < srcSizeROI.height; i++)
            {
                memcpy(srcPtrROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                srcPtrImageTemp += elementsInRowMax;
                srcPtrROITemp += elementsInRowROI;
            }

            if (mirrorFlag == 0)
            {
                resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);
            }
            else if (mirrorFlag == 1)
            {
                T *srcPtrROIMirrorred = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
                T *srcPtrROIMirrorredTemp;
                srcPtrROIMirrorredTemp = srcPtrROIMirrorred;
                Rpp32u bufferLength = channel * srcSizeROI.width;
                Rpp32u alignedLength = (bufferLength / 15) * 15;

                srcPtrROITemp = srcPtrROI + (channel * (srcSizeROI.width - 1));

                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    __m128i px0;
                    __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                    {
                        srcPtrROITemp -= 13;
                        px0 = _mm_loadu_si128((__m128i *)srcPtrROITemp);
                        px0 = _mm_shuffle_epi8(px0, vMask);
                        _mm_storeu_si128((__m128i *)srcPtrROIMirrorredTemp, px0);
                        srcPtrROITemp -= 2;
                        srcPtrROIMirrorredTemp += 15;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                    {
                        memcpy(srcPtrROIMirrorredTemp, srcPtrROITemp, channel * sizeof(T));
                        srcPtrROIMirrorredTemp += channel;
                        srcPtrROITemp -= channel;
                    }

                    srcPtrROITemp = srcPtrROITemp + (channel * (2 * srcSizeROI.width));
                }

                resize_kernel_host(srcPtrROIMirrorred, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

                free(srcPtrROIMirrorred);
            }

            compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }
    
    return RPP_SUCCESS;
}
















































// Processing all vs unpadded pixels backup

// template <typename T>
// RppStatus color_twist_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
//                          Rpp32f *batch_alpha, Rpp32f *batch_beta, 
//                          Rpp32f *batch_hueShift, Rpp32f *batch_saturationFactor, 
//                          RppiROI *roiPoints, Rpp32u nbatchSize,
//                          RppiChnFormat chnFormat, Rpp32u channel)
// {
//     if(chnFormat == RPPI_CHN_PLANAR)
//     {

//         for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
//         {
//             Rpp64u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

//             // Rpp32f x1 = roiPoints[batchCount].x;
//             // Rpp32f y1 = roiPoints[batchCount].y;
//             // Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
//             // Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
//             // if (x2 == -1)
//             // {
//             //     x2 = batch_srcSize[batchCount].width - 1;
//             //     roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
//             // }
//             // if (y2 == -1)
//             // {
//             //     y2 = batch_srcSize[batchCount].height - 1;
//             //     roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
//             // }

//             Rpp32f hueShift = batch_hueShift[batchCount];
//             Rpp32f saturationFactor = batch_saturationFactor[batchCount];
//             Rpp32f alpha = batch_alpha[batchCount];
//             Rpp32f beta = batch_beta[batchCount];
            
//             // T *srcPtrImage, *dstPtrImage;
//             // Rpp32u loc = 0;
//             // compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
//             // srcPtrImage = srcPtr + loc;
//             // dstPtrImage = dstPtr + loc;

//             // RppiSize srcSizeROI;
//             // srcSizeROI.height = roiPoints[batchCount].roiHeight;
//             // srcSizeROI.width = roiPoints[batchCount].roiWidth;
            
//             // T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
//             // T *dstPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
//             // T *srcPtrImageTemp, *srcPtrROITemp, *dstPtrROITemp;
//             // srcPtrROITemp = srcPtrROI;
//             // dstPtrROITemp = dstPtrROI;
//             // for (int c = 0; c < channel; c++)
//             // {
//             //     srcPtrImageTemp = srcPtrImage + (c * imageDimMax) + ((Rpp32u) y1 * batch_srcSizeMax[batchCount].width) + (Rpp32u) x1;
//             //     for (int i = 0; i < srcSizeROI.height; i++)
//             //     {
//             //         memcpy(srcPtrROITemp, srcPtrImageTemp, srcSizeROI.width * sizeof(T));
//             //         srcPtrImageTemp += batch_srcSizeMax[batchCount].width;
//             //         srcPtrROITemp += srcSizeROI.width;
//             //     }
//             // }

//             // color_twist_host(srcPtrROI, srcSizeROI, dstPtrROI, alpha, beta, hueShift, saturationFactor, chnFormat, channel);
//             color_twist_host(srcPtr, batch_srcSizeMax[batchCount], dstPtr, alpha, beta, hueShift, saturationFactor, chnFormat, channel);

//             // for(int c = 0; c < channel; c++)
//             // {
//             //     T *srcPtrChannel, *dstPtrChannel;
//             //     srcPtrChannel = srcPtrImage + (c * imageDimMax);
//             //     dstPtrChannel = dstPtrImage + (c * imageDimMax);

//             //     Rpp32u roiRowCount = 0;

//             //     for(int i = 0; i < batch_srcSize[batchCount].height; i++)
//             //     {
//             //         Rpp32f pixel;

//             //         T *srcPtrTemp, *dstPtrTemp, *dstPtrROITemp;
//             //         srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
//             //         dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    
//             //         if (!((y1 <= i) && (i <= y2)))
//             //         {
//             //             memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

//             //             dstPtrTemp += batch_srcSizeMax[batchCount].width;
//             //             srcPtrTemp += batch_srcSizeMax[batchCount].width;
//             //         }
//             //         else
//             //         {
//             //             dstPtrROITemp = dstPtrROI + (roiRowCount * srcSizeROI.width);
//             //             for(int j = 0; j < batch_srcSize[batchCount].width; j++)
//             //             {
//             //                 if((x1 <= j) && (j <= x2 ))
//             //                 {
//             //                     *dstPtrTemp = *dstPtrROITemp;

//             //                     srcPtrTemp++;
//             //                     dstPtrTemp++;
//             //                     dstPtrROITemp++;
//             //                 }
//             //                 else
//             //                 {
//             //                     *dstPtrTemp = *srcPtrTemp;

//             //                     srcPtrTemp++;
//             //                     dstPtrTemp++;
//             //                 }
//             //             }
//             //             roiRowCount++;
//             //         }
//             //     }
//             // }

//             // free(srcPtrROI);
//             // free(dstPtrROI);
//         }
//     }
//     else if(chnFormat == RPPI_CHN_PACKED)
//     {

//         for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
//         {
//             Rpp64u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

//             // Rpp32f x1 = roiPoints[batchCount].x;
//             // Rpp32f y1 = roiPoints[batchCount].y;
//             // Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
//             // Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
//             // if (x2 == -1)
//             // {
//             //     x2 = batch_srcSize[batchCount].width - 1;
//             //     roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
//             // }
//             // if (y2 == -1)
//             // {
//             //     y2 = batch_srcSize[batchCount].height - 1;
//             //     roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
//             // }

//             Rpp32f hueShift = batch_hueShift[batchCount];
//             Rpp32f saturationFactor = batch_saturationFactor[batchCount];
//             Rpp32f alpha = batch_alpha[batchCount];
//             Rpp32f beta = batch_beta[batchCount];
            
//             T *srcPtrImage, *dstPtrImage;
//             Rpp32u loc = 0;
//             compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
//             srcPtrImage = srcPtr + loc;
//             dstPtrImage = dstPtr + loc;

//             // RppiSize srcSizeROI;
//             // srcSizeROI.height = roiPoints[batchCount].roiHeight;
//             // srcSizeROI.width = roiPoints[batchCount].roiWidth;

//             // Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
//             // Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
//             // Rpp32u elementsInRowROI = channel * srcSizeROI.width;
            
//             // T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
//             // T *dstPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
//             // T *srcPtrImageTemp, *srcPtrROITemp, *dstPtrROITemp;
//             // srcPtrROITemp = srcPtrROI;

//             // srcPtrImageTemp = srcPtrImage + ((Rpp32u) y1 * elementsInRowMax) + (channel * (Rpp32u) x1);
//             // for (int i = 0; i < srcSizeROI.height; i++)
//             // {
//             //     memcpy(srcPtrROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
//             //     srcPtrImageTemp += elementsInRowMax;
//             //     srcPtrROITemp += elementsInRowROI;
//             // }

//             // color_twist_host(srcPtrROI, srcSizeROI, dstPtrROI, alpha, beta, hueShift, saturationFactor, chnFormat, channel);
//             color_twist_host(srcPtrImage, batch_srcSizeMax[batchCount], dstPtrImage, alpha, beta, hueShift, saturationFactor, chnFormat, channel);

//             // Rpp32u roiRowCount = 0;

//             // for(int i = 0; i < batch_srcSize[batchCount].height; i++)
//             // {
//             //     Rpp32f pixel;

//             //     T *srcPtrTemp, *dstPtrTemp, *dstPtrROITemp;
//             //     srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
//             //     dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

//             //     if (!((y1 <= i) && (i <= y2)))
//             //     {
//             //         memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

//             //         dstPtrTemp += elementsInRowMax;
//             //         srcPtrTemp += elementsInRowMax;
//             //     }
//             //     else
//             //     {
//             //         dstPtrROITemp = dstPtrROI + (roiRowCount * elementsInRowROI);
//             //         for(int j = 0; j < batch_srcSize[batchCount].width; j++)
//             //         {
//             //             if (!((x1 <= j) && (j <= x2 )))
//             //             {
//             //                 memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));

//             //                 dstPtrTemp += channel;
//             //                 srcPtrTemp += channel;
//             //             }
//             //             else
//             //             {
//             //                 memcpy(dstPtrTemp, dstPtrROITemp, channel * sizeof(T));
//             //                 srcPtrTemp += channel;
//             //                 dstPtrTemp += channel;
//             //                 dstPtrROITemp += channel;
//             //             }
//             //         }
//             //         roiRowCount++;
//             //     }
//             // }

//             // free(srcPtrROI);
//             // free(dstPtrROI);
//         }
//     }

//     return RPP_SUCCESS;
// }

#endif
