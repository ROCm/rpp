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


template <typename T>
RppStatus brightness_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                          Rpp32f alpha, Rpp32f beta, 
                          RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32f) (*srcPtrTemp)) * alpha + beta;
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp = (T) round(pixel);
        srcPtrTemp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}


/**************** Gamma Correction ***************/

template <typename T>
RppStatus gamma_correction_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32f gamma,
                                RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32f) (*srcPtrTemp)) / 255.0;
        pixel = pow(pixel, gamma);
        pixel = pixel * 255.0;
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp = (T) pixel;
        srcPtrTemp++;
        dstPtrTemp++;
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
    int sum = 0;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        for(int y = 0 ; y < srcSize.height ;)
        {
            for(int x = 0 ; x < srcSize.width ;)
            {
                for(int c = 0 ; c < channel ; c++)    
                {    
                    for(int i = 0 ; i < 7 ; i++)
                    {
                        for(int j = 0 ; j < 7 ; j++)
                        {
                            if(y + i < srcSize.height && x + j < srcSize.width && y + i >= 0 && x + j >= 0)
                            {    
                                sum += *(srcPtr + ((y + i) * srcSize.width * channel + (x + j) * channel + c));
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
                                *(dstTemp + ((y + i) * srcSize.width * channel + (x + j) * channel + c)) = RPPPIXELCHECK(sum);
                            }
                        }
                    }
                    sum = 0;
                }
                x +=7;
            }
            y += 7;
        }
    }    
    else
    {
        for(int c = 0 ; c < channel ; c++)
        {
            for(int y = 0 ; y < srcSize.height ;)
            {
                for(int x = 0 ; x < srcSize.width ;)    
                {    
                    for(int i = 0 ; i < 7 ; i++)
                    {
                        for(int j = 0 ; j < 7 ; j++)
                        {
                            if(y + i < srcSize.height && x + j < srcSize.width && y + i >= 0 && x + j >= 0)
                            {    
                                sum += *(srcPtr + (y + i) * srcSize.width + (x + j) + c * srcSize.height * srcSize.width);
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
                                *(dstTemp + (y + i) * srcSize.width + (x + j) + c * srcSize.height * srcSize.width) = RPPPIXELCHECK(sum);
                            }
                        }
                    }
                    sum = 0;
                    x +=7;
                }
                y += 7;
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
    int bound = (kernelSize - 1) / 2;
    srand(time(0)); 
    unsigned int width = srcSize.width;
    unsigned int height = srcSize.height;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        for(int id_y = 0 ; id_y < srcSize.height ; id_y++)
        {
            for(int id_x = 0 ; id_x < srcSize.width ; id_x++)
            {
                int pixIdx = id_y * channel * width + id_x * channel;
                int nhx = rand() % (kernelSize);
                int nhy = rand() % (kernelSize);
                if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width - 1)
                {
                    int index = ((id_y - bound) * channel * width) + ((id_x - bound) * channel) + (nhy * channel * width) + (nhx * channel);
                    for(int i = 0 ; i < channel ; i++)
                    {
                        *(dstPtr + pixIdx + i) = *(srcPtr + index + i);  
                    }
                }
                else 
                {
                    for(int i = 0 ; i < channel ; i++)
                    {
                        *(dstPtr + pixIdx + i) = *(srcPtr + pixIdx + i);  
                    }
                }
            }
        }
    }
    else
    {
        for(int id_y = 0 ; id_y < srcSize.height ; id_y++)
        {
            for(int id_x = 0 ; id_x < srcSize.width ; id_x++)
            {
                int pixIdx = id_y * width + id_x;
                int channelPixel = height * width;
                int nhx = rand() % (kernelSize);
                int nhy = rand() % (kernelSize);
                int bound = (kernelSize - 1) / 2;
                if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width - 1)
                {
                    int index = ((id_y - bound) * width) + (id_x - bound) + (nhy * width) + (nhx);
                    for(int i = 0 ; i < channel ; i++)
                    {
                        *(dstPtr + pixIdx + (height * width * i)) = *(srcPtr + index + (height * width * i));  
                    }
                }
                else 
                {
                    for(int i = 0 ; i < channel ; i++)
                    {
                        *(dstPtr + pixIdx + (height * width * i)) = *(srcPtr + pixIdx + (height * width * i));  
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
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        *dstPtr = ((1 - alpha) * (*srcPtr1)) + (alpha * (*srcPtr2));
        srcPtr1++;
        srcPtr2++;
        dstPtr++;
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
        for(int i = 0;i < srcSize.height * srcSize.width * channel;i++)
        {
            *srcPtr = *temp;
            srcPtr++;
            temp++;
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
            Rpp8u *srcPtr1, *srcPtr2;
            srcPtr1 = srcPtr + 1;
            srcPtr2 = srcPtr + 2;
            for (int i = 0; i < (srcSize.width * srcSize.height * channel); i += 3)
            {
                Rpp32f check = (*srcPtr + *srcPtr1 + *srcPtr2) / 3;
                *srcPtr = fogGenerator(*srcPtr, fogValue, 1, check);
                *srcPtr1 = fogGenerator(*srcPtr1, fogValue, 2, check);
                *srcPtr2 = fogGenerator(*srcPtr2, fogValue, 3, check);
                srcPtr += 3;
                srcPtr1 += 3;
                srcPtr2 += 3;
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

    Rpp32u rainDrops = (Rpp32u)(rainPercentage * srcSize.width * srcSize.height * channel );
    
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int i = 0 ; i < rainDrops ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
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
        for(int i = 0 ; i < rainDrops ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
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

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32f) (*srcPtrTemp)) * (pow(2, exposureFactor));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp = (T) round(pixel);
        dstPtrTemp++;
        srcPtrTemp++;
    }

    return RPP_SUCCESS;
}