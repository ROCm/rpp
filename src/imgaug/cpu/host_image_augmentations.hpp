#include <cpu/rpp_cpu_common.hpp>
#include <stdlib.h>
#include <time.h>

/************ Blur************/

template <typename T>
RppStatus blur_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f stdDev, Rpp32u kernelSize,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }
    Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    int bound = ((kernelSize - 1) / 2);

    generate_gaussian_kernel_host(stdDev, kernel, kernelSize);

    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + (2 * bound);
    srcSizeMod.height = srcSize.height + (2 * bound);
    T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(T));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    RppiSize rppiKernelSize;
    rppiKernelSize.height = kernelSize;
    rppiKernelSize.width = kernelSize;
    convolve_image_host(srcPtrMod, srcSizeMod, dstPtr, srcSize, kernel, rppiKernelSize, chnFormat, channel);
    free(kernel);
    free(srcPtrMod);
    return RPP_SUCCESS;
}

/**************** Contrast ***************/

template <typename T, typename U>
RppStatus contrast_host(T* srcPtr, RppiSize srcSize, U* dstPtr, 
                        Rpp32f new_min, Rpp32f new_max,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp;
    U *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f pixel, min, max;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
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
                srcPtrTemp++;
            }

            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                *dstPtrTemp = (U) (((((Rpp32f) (*srcPtrTemp)) - min) * ((new_max - new_min) / (max - min))) + new_min);
                srcPtrTemp++;
                dstPtrTemp++;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
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
                *dstPtrTemp = (U) (((((Rpp32f) (*srcPtrTemp)) - min) * ((new_max - new_min) / (max - min))) + new_min);
                srcPtrTemp = srcPtrTemp + channel;
                dstPtrTemp = dstPtrTemp + channel;
            }
        }
    }

    return RPP_SUCCESS;
}

/************ Brightness ************/

#include <omp.h>
#if ENABLE_SSE_INTRINSICS

// TODO:: add AVX if supported
template <typename T>
RppStatus brightness_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                          Rpp32f alpha, Rpp32f beta,
                          RppiChnFormat chnFormat, Rpp32u channel)
{
    if (std::is_same<T, Rpp8u>::value) {
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

    } else {
        T *srcPtrTemp, *dstPtrTemp;
        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;
        //Rpp32f pixel;
    #pragma omp parallel for simd
        for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
        {
            Rpp32f pixel = ((Rpp32f) (srcPtrTemp[i])) * alpha + beta;
            pixel = RPPPIXELCHECK(pixel);
            dstPtrTemp[i] = (T) round(pixel);
        }

    }

    return RPP_SUCCESS;

}

#else

template <typename T>
RppStatus brightness_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                          Rpp32f alpha, Rpp32f beta, 
                          RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    //Rpp32f pixel;
#pragma omp parallel for simd
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        Rpp32f pixel = ((Rpp32f) (srcPtrTemp[i])) * alpha + beta;
        pixel = RPPPIXELCHECK(pixel);
        dstPtrTemp[i] = (T) round(pixel);
    }

    return RPP_SUCCESS;

}

#endif

/**************** Gamma Correction ***************/

template <typename T>
RppStatus gamma_correction_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32f gamma,
                                RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    //Rpp32f pixel;
#pragma omp parallel for simd
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        Rpp32f pixel = ((Rpp32f) (srcPtrTemp[i])) / 255.0;
        pixel = pow(pixel, gamma);
        pixel = pixel * 255.0;
        pixel = RPPPIXELCHECK(pixel);
        dstPtrTemp[i] = (T) pixel;
    }

    return RPP_SUCCESS;

}


/**************** Pixelate ***************/

template <typename T>
RppStatus pixelate_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp8u *dstTemp,*srcTemp;
    dstTemp = dstPtr;
    srcTemp = srcPtr;

    if(chnFormat == RPPI_CHN_PACKED)
    {
#pragma omp parallel for
        for(int y = 0 ; y < srcSize.height ;y += 7)
        {
#pragma omp parallel for
            for(int x = 0 ; x < srcSize.width ;x +=7)
            {
                int sum = 0;
                for(int c = 0 ; c < channel ; c++)    
                {    
                    for(int i = 0 ; i < 7 ; i++)
                    {
                        for(int j = 0 ; j < 7 ; j++)
                        {
                            if(y + i < srcSize.height && x + j < srcSize.width && y + i >= 0 && x + j >= 0)
                            {    
                                sum += srcPtr [ ((y + i) * srcSize.width * channel + (x + j) * channel + c)];
                            }
                        }
                    }
                    sum /= 49;

                    for(int i = 0 ; i < 7 ; i++)
                    {
                        for(int j = 0 ; j < 7 ; j++)
                        {
                            if(y + i < srcSize.height && x + j < srcSize.width)
                            {    
                                dstTemp [ ((y + i) * srcSize.width * channel + (x + j) * channel + c)] = RPPPIXELCHECK(sum);
                            }
                        }
                    }
                }
            }
        }
    }    
    else
    {
        for(int c = 0 ; c < channel ; c++)
        {
#pragma omp parallel for
            for(int y = 0 ; y < srcSize.height ;y += 7)
            {
#pragma omp parallel for
                for(int x = 0 ; x < srcSize.width ;x +=7)
                {
                    int sum = 0;
                    for(int i = 0 ; i < 7 ; i++)
                    {
                        for(int j = 0 ; j < 7 ; j++)
                        {
                            if(y + i < srcSize.height && x + j < srcSize.width && y + i >= 0 && x + j >= 0)
                            {    
                                sum += srcPtr [(y + i) * srcSize.width + (x + j) + c * srcSize.height * srcSize.width];
                            }
                        }
                    }
                    sum /= 49;

                    for(int i = 0 ; i < 7 ; i++)
                    {
                        for(int j = 0 ; j < 7 ; j++)
                        {
                            if(y + i < srcSize.height && x + j < srcSize.width)
                            {    
                                dstTemp [ (y + i) * srcSize.width + (x + j) + c * srcSize.height * srcSize.width] = RPPPIXELCHECK(sum);
                            }
                        }
                    }

                }
            }
        }
    }
    return RPP_SUCCESS;
}

/**************** Jitter ***************/

template <typename T>
RppStatus jitter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u kernelSize, 
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp8u *dstTemp,*srcTemp;
    dstTemp = dstPtr;
    srcTemp = srcPtr;
    const int bound = (kernelSize - 1) / 2;
    const unsigned int width = srcSize.width;
    const unsigned int height = srcSize.height;
    int anhx[srcSize.width];
    int anhy[srcSize.width];
    for(int i = 0; i<  srcSize.width; i++)
    {
        anhx[i] = rand() % kernelSize;
        anhy[i] = rand() % kernelSize;
    }
    if(chnFormat == RPPI_CHN_PACKED)
    {
#pragma omp parallel for
        for(int id_y = 0 ; id_y < srcSize.height ; id_y++)
        {
#pragma omp parallel for simd
            for(int id_x = 0 ; id_x < srcSize.width ; id_x++)
            {
                int pixIdx = id_y * channel * width + id_x * channel;
                int nhx = anhx[id_x];
                int nhy = anhy[id_x];
                if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width - 1)
                {
                    int index = ((id_y - bound) * channel * width) + ((id_x - bound) * channel) + (nhy * channel * width) + (nhx * channel);
                    for(int i = 0 ; i < channel ; i++)
                    {
                        dstTemp [pixIdx + i] = srcTemp[ index + i];
                    }
                }
                else 
                {
                    for(int i = 0 ; i < channel ; i++)
                    {
                        dstTemp[pixIdx + i]= srcTemp[ pixIdx + i];
                    }
                }
            }
        }
    }
    else
    {
#pragma omp parallel for
        for(int id_y = 0 ; id_y < srcSize.height ; id_y++)
        {
#pragma omp parallel for simd
            for(int id_x = 0 ; id_x < srcSize.width ; id_x++)
            {
                int pixIdx = id_y * width + id_x;
                int nhx = anhx[id_x];
                int nhy = anhy[id_x];
                int bound = (kernelSize - 1) / 2;
                if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width - 1)
                {
                    int index = ((id_y - bound) * width) + (id_x - bound) + (nhy * width) + (nhx);
                    for(int i = 0 ; i < channel ; i++)
                    {
                        dstPtr [ pixIdx + (height * width * i)] = srcPtr [ index + (height * width * i)];
                    }
                }
                else 
                {
                    for(int i = 0 ; i < channel ; i++)
                    {
                        dstPtr [pixIdx + (height * width * i)] = srcPtr [ pixIdx + (height * width * i)];
                    }
                }
            }
        }
        
    }
    
    
    return RPP_SUCCESS;
}

/**************** Snow ***************/

template <typename T, typename U>
RppStatus snow_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    Rpp32f strength,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    strength = strength/100;
    int snow_mat[5][5] = {{0,50,75,50,0}, {40,80,120,80,40}, {75,120,255,120,75}, {40,80,120,80,40}, {0,50,75,50,0}};

    Rpp32u snowDrops = (Rpp32u)(strength * srcSize.width * srcSize.height * channel );

    U *dstptrtemp;
    dstptrtemp=dstPtr;
    for(int k=0;k<srcSize.height*srcSize.width*channel;k++)
    {
        *dstptrtemp = 0;
        dstptrtemp++;
    }
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int i = 0 ; i < snowDrops ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp32f pixel;
            for(int k = 0;k < channel;k++)
            {
                dstPtr[(row * srcSize.width) + (column) + (k * srcSize.height * srcSize.width)] = snow_mat[0][0] ;
            }
            for(int j = 0;j < 5;j++)
            {
                if(row + 5 < srcSize.height && row + 5 > 0 )
                for(int k = 0;k < channel;k++)
                {
                    for(int m = 0;m < 5;m++)
                    {
                        if (column + 5 < srcSize.width && column + 5 > 0)
                        {
                            dstPtr[(row * srcSize.width) + (column) + (k * srcSize.height * srcSize.width) + (srcSize.width * j) + m] = snow_mat[j][m] ;
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(int i = 0 ; i < snowDrops ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp32f pixel;
            for(int k = 0;k < channel;k++)
            {
                dstPtr[(channel * row * srcSize.width) + (column * channel) + k] = snow_mat[0][0] ;
            }
            for(int j = 0;j < 5;j++)
            {
                if(row + 5 < srcSize.height && row + 5 > 0 )
                for(int k = 0;k < channel;k++)
                {
                    for(int m = 0;m < 5;m++)
                    {
                        if (column + 5 < srcSize.width && column + 5 > 0)
                        {
                            dstPtr[(channel * row * srcSize.width) + (column * channel) + k + (channel * srcSize.width * j) + (channel * m)] = snow_mat[j][m];
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32u pixel = ((Rpp32u) srcPtr[i]) + (Rpp32u)dstPtr[i];
        dstPtr[i] = RPPPIXELCHECK(pixel);
    }
    return RPP_SUCCESS;
}

/**************** Blend ***************/

template <typename T>
RppStatus blend_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr, 
                        Rpp32f alpha, RppiChnFormat chnFormat, 
                        unsigned int channel)
{
#pragma omp parallel for
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        dstPtr[i] = ((1 - alpha) * (srcPtr1[i])) + (alpha * (srcPtr2[i]));
    }  

    return RPP_SUCCESS;  
}

/**************** Add Noise ***************/

//Salt and Pepper Host function

template <typename T>
RppStatus noise_snp_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                         Rpp32f noiseProbability,
                         RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp8u *cpdst,*cpsrc;
    cpdst = dstPtr;
    cpsrc = srcPtr;
    for (int i = 0; i < ( channel * srcSize.width * srcSize.height ); i++ )
    {
        *cpdst = *cpsrc;
        cpdst++;
        cpsrc++;
    }
    if(noiseProbability != 0)
    {
        srand(time(0));
        Rpp32u noisePixel = (Rpp32u)(noiseProbability * srcSize.width * srcSize.height );
        Rpp32u pixelDistance = (srcSize.width * srcSize.height) / noisePixel;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            for(int i = 0 ; i < srcSize.width * srcSize.height * channel ; i += channel*pixelDistance)
            {
                Rpp32u initialPixel = rand() % pixelDistance;
                dstPtr += initialPixel*channel;
                Rpp8u newPixel = rand()%2 ? 0 : 255;
                for(int j = 0 ; j < channel ; j++)
                {
                    *dstPtr = newPixel;
                    dstPtr++;
                }
                dstPtr += ((pixelDistance - initialPixel - 1) * channel);
            }
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            if(channel == 3)
            {
                Rpp8u *dstPtrTemp1,*dstPtrTemp2;
                dstPtrTemp1 = dstPtr + (srcSize.height * srcSize.width);
                dstPtrTemp2 = dstPtr + (2 * srcSize.height * srcSize.width);
                for(int i = 0 ; i < srcSize.width * srcSize.height * channel ; i += pixelDistance)
                {
                    Rpp32u initialPixel = rand() % pixelDistance;
                    dstPtr += initialPixel;
                    Rpp8u newPixel = (rand() % 2) ? 255 : 1;
                    *dstPtr = newPixel;
                    dstPtr += ((pixelDistance - initialPixel - 1));

                    dstPtrTemp1 += initialPixel;
                    *dstPtrTemp1 = newPixel;
                    dstPtrTemp1 += ((pixelDistance - initialPixel - 1));

                    dstPtrTemp2 += initialPixel;
                    *dstPtrTemp2 = newPixel;
                    dstPtrTemp2 += ((pixelDistance - initialPixel - 1));

                }
            }
            else
            {
                for(int i = 0 ; i < srcSize.width * srcSize.height ; i += pixelDistance)
                {
                    Rpp32u initialPixel = rand() % pixelDistance;
                    dstPtr += initialPixel;
                    Rpp8u newPixel = rand()%2 ? 255 : 1;
                    *dstPtr = newPixel;
                    dstPtr += ((pixelDistance - initialPixel - 1));
                }
            }

        }
    }
    return RPP_SUCCESS;
}
/**************** Fog ***************/
template <typename T>
RppStatus fog_host(T* srcPtr, RppiSize srcSize,
                   Rpp32f fogValue,
                   RppiChnFormat chnFormat,   unsigned int channel, T* temp)
{
    if(fogValue <= 0)
    {
#pragma omp parallel for
        for(int i = 0;i < srcSize.height * srcSize.width * channel;i++)
        {
            srcPtr[i] = temp[i];
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
                *srcPtr = fogGenerator(*srcPtr, fogValue, 1, check);
                srcPtr++;
                if(channel > 1)
                {
                    *srcPtr1 = fogGenerator(*srcPtr1, fogValue, 2, check);
                    *srcPtr2 = fogGenerator(*srcPtr2, fogValue, 3, check);
                    srcPtr1++;
                    srcPtr2++;
                }
            }
        }
        else
        {
            Rpp8u * srcPtr1 = srcPtr + 1;
            Rpp8u * srcPtr2 = srcPtr + 2;
#pragma omp parallel for
            for (int i = 0; i < (srcSize.width * srcSize.height * channel); i += 3)
            {
                Rpp32f check = 0;//(srcPtr[i] + srcPtr1[i] + srcPtr2[i]) / 3;
                srcPtr[i] = 0;//fogGenerator(srcPtr[i], fogValue, 1, check);
                srcPtr1[i] = 1;//fogGenerator(srcPtr1[i], fogValue, 2, check);
                srcPtr2[i] = 2;//fogGenerator(srcPtr2[i], fogValue, 3, check);
            }
        }
    }
    return RPP_SUCCESS;

}

/**************** Rain ***************/
template <typename T>
RppStatus rain_host(T* srcPtr, RppiSize srcSize,T* dstPtr,
                    Rpp32f rainPercentage, Rpp32f rainWidth, Rpp32f rainHeight, Rpp32f transparency,
                    RppiChnFormat chnFormat,   unsigned int channel)
{ 
    rainPercentage = rainPercentage / 250;
    transparency /= 5;

    const Rpp32u rainDrops = (Rpp32u)(rainPercentage * srcSize.width * srcSize.height * channel );
    const unsigned rand_len = srcSize.width;
    unsigned int col_rand[rand_len];
    unsigned int row_rand[rand_len];
    for(int i = 0; i<  rand_len; i++)
    {
        col_rand[i] = rand() % srcSize.width;
        row_rand[i] = rand() % srcSize.height;
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
            for(int k = 0;k < channel;k++)
            {
                //pixel=(Rpp32f)dstPtr[(row * srcSize.width) + (column) + (k*srcSize.height*srcSize.width)] + 5;
                dstPtr[(row * srcSize.width) + (column) + (k * srcSize.height * srcSize.width)] = (k == 0) ? 196 : (k == 1) ? 226 : 255 ;
            }
            for(int j = 0;j < rainHeight;j++)
            {
                if(row + rainHeight < srcSize.height && row + rainHeight > 0 )
                for(int k = 0;k < channel;k++)
                {
                    for(int m = 0;m < rainWidth;m++)
                    {
                        if (column + rainWidth < srcSize.width && column + rainWidth > 0)
                        {
                            //pixel=(Rpp32f)dstPtr[(row * srcSize.width) + (column) + (k*srcSize.height*srcSize.width) + (srcSize.width*j)+m]+5;
                            dstPtr[(row * srcSize.width) + (column) + (k * srcSize.height * srcSize.width) + (srcSize.width * j) + m] = (k == 0) ? 196 : (k == 1) ? 226 : 255 ;
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
            for(int k = 0;k < channel;k++)
            {
                //pixel=(Rpp32f)dstPtr[(channel * row * srcSize.width) + (column * channel) + k] + 5;
                dstPtr[(channel * row * srcSize.width) + (column * channel) + k] = (k == 0) ? 196 : (k == 1) ? 226 : 255 ;
            }
            for(int j = 0;j < rainHeight;j++)
            {
                if(row + rainHeight < srcSize.height && row + rainHeight > 0 )
                for(int k = 0;k < channel;k++)
                {
                    for(int m = 0;m < rainWidth;m++)
                    {
                        if (column + rainWidth < srcSize.width && column + rainWidth > 0)
                        {
                            //pixel=(Rpp32f)dstPtr[(channel * row * srcSize.width) + (column * channel) + k + (channel * srcSize.width * j)+(channel*m)]+5;
                            dstPtr[(channel * row * srcSize.width) + (column * channel) + k + (channel * srcSize.width * j) + (channel * m)] = (k == 0) ? 196 : (k == 1) ? 226 : 255 ;
                        }
                    } 
                }            
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]) + transparency * dstPtr[i];
        dstPtr[i] = RPPPIXELCHECK(pixel);
    }
    return RPP_SUCCESS;
}

/**************** Exposure Modification ***************/

template <typename T, typename U>
RppStatus exposure_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    Rpp32f exposureFactor,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;
#pragma omp parallel for
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32f) (srcPtrTemp[i])) * (pow(2, exposureFactor));
        pixel = RPPPIXELCHECK(pixel);
        dstPtrTemp[i] = (T) round(pixel);
    }

    return RPP_SUCCESS;
}
