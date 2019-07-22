#include <cpu/rpp_cpu_common.hpp>
#include <stdlib.h>
#include <time.h>

/************ Blur************/

template <typename T>
RppStatus blur_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f stdDev, unsigned int kernelSize,
                    RppiChnFormat chnFormat, unsigned int channel)
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
    Rpp8u *srcPtrMod = (Rpp8u *)calloc(srcSizeMod.width * srcSizeMod.height * channel, sizeof(Rpp8u));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);
    
    convolve_image_host(srcPtrMod, srcSizeMod, dstPtr, srcSize, kernel, kernelSize, chnFormat, channel);
    
    return RPP_SUCCESS;
}

/**************** Contrast ***************/

template <typename T>
RppStatus contrast_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                        Rpp32u new_min, Rpp32u new_max,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            Rpp32f Min, Max;
            Min = srcPtr[c * srcSize.height * srcSize.width];
            Max = srcPtr[c * srcSize.height * srcSize.width];
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                if (srcPtr[i + (c * srcSize.height * srcSize.width)] < Min)
                {
                    Min = srcPtr[i + (c * srcSize.height * srcSize.width)];
                }
                if (srcPtr[i + (c * srcSize.height * srcSize.width)] > Max)
                {
                    Max = srcPtr[i + (c * srcSize.height * srcSize.width)];
                }
            }
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                Rpp32f pixel = (Rpp32f) srcPtr[i + (c * srcSize.height * srcSize.width)];
                pixel = ((pixel - Min) * ((new_max - new_min) / (Max - Min))) + new_min;
                pixel = (pixel < (Rpp32f)new_max) ? pixel : ((Rpp32f)new_max);
                pixel = (pixel > (Rpp32f)new_min) ? pixel : ((Rpp32f)new_min);
                dstPtr[i + (c * srcSize.height * srcSize.width)] = (Rpp8u) pixel;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(int c = 0; c < channel; c++)
        {
            Rpp32f Min, Max;
            Min = srcPtr[c];
            Max = srcPtr[c];
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                if (srcPtr[(channel * i) + c] < Min)
                {
                    Min = srcPtr[(channel * i) + c];
                }
                if (srcPtr[(channel * i) + c] > Max)
                {
                    Max = srcPtr[(channel * i) + c];
                }
            }
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                Rpp32f pixel = (Rpp32f) srcPtr[(channel * i) + c];
                pixel = ((pixel - Min) * ((new_max - new_min) / (Max - Min))) + new_min;
                pixel = (pixel < (Rpp32f)new_max) ? pixel : ((Rpp32f)new_max);
                pixel = (pixel > (Rpp32f)new_min) ? pixel : ((Rpp32f)new_min);
                dstPtr[(channel * i) + c] = (Rpp8u) pixel;
            }
        }
    }

    return RPP_SUCCESS;
}

/************ Brightness ************/

template <typename T>
RppStatus brightness_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                          Rpp32f alpha, Rpp32f beta, 
                          RppiChnFormat chnFormat, unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]) * alpha + beta;
        pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
        pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
        dstPtr[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}


/**************** Gamma Correction ***************/

template <typename T>
RppStatus gamma_correction_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32f gamma,
                                RppiChnFormat chnFormat,   unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]) / 255;
        pixel = pow(pixel, gamma);
        pixel *= 255;
        dstPtr[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}


/**************** Pixelate ***************/

template <typename T>
RppStatus pixelate_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    unsigned int kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2, 
                    RppiChnFormat chnFormat, unsigned int channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }

    unsigned int bound = ((kernelSize - 1) / 2);

    if ((RPPINRANGE(x1, bound, srcSize.width - 1 - bound) == 0) 
        || (RPPINRANGE(x2, bound, srcSize.width - 1 - bound) == 0) 
        || (RPPINRANGE(y1, bound, srcSize.height - 1 - bound) == 0) 
        || (RPPINRANGE(y2, bound, srcSize.height - 1 - bound) == 0))
    {
        return RPP_ERROR;
    }



    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = *srcPtrTemp;
        srcPtrTemp++;
        dstPtrTemp++;
    }



    Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));

    generate_box_kernel_host(kernel, kernelSize);

    RppiSize srcSizeMod, srcSizeSubImage;
    T *srcPtrMod, *srcPtrSubImage, *dstPtrSubImage;

    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);
    compute_subimage_location_host(dstPtr, &dstPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    srcSizeMod.height = srcSizeSubImage.height + (2 * bound);
    srcSizeMod.width = srcSizeSubImage.width + (2* bound);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        srcPtrMod = srcPtrSubImage - (bound * srcSize.width) - bound;
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrMod = srcPtrSubImage - (bound * srcSize.width * channel) - (bound * channel);
    }

    convolve_subimage_host(srcPtrMod, srcSizeMod, dstPtrSubImage, srcSizeSubImage, srcSize, kernel, kernelSize, chnFormat, channel);

    return RPP_SUCCESS;
}

/**************** Jitter Add ***************/

template <typename T>
RppStatus jitter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    unsigned int maxJitterX, unsigned int maxJitterY, 
                    RppiChnFormat chnFormat, unsigned int channel)
{
    if ((RPPINRANGE(maxJitterX, 0, srcSize.width - 1) == 0) 
        || (RPPINRANGE(maxJitterY, 0, srcSize.height - 1) == 0))
    {
        return RPP_ERROR;
    }

    Rpp8u *dstPtrForJitter = (Rpp8u *)calloc(channel * srcSize.height * srcSize.width, sizeof(Rpp8u));

    T *srcPtrTemp, *dstPtrTemp;
    T *srcPtrBeginJitter, *dstPtrBeginJitter;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtrForJitter;
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = *srcPtrTemp;
        srcPtrTemp++;
        dstPtrTemp++;
    }

    srand (time(NULL));
    int jitteredPixelLocDiffX, jitteredPixelLocDiffY;
    int jitterRangeX = 2 * maxJitterX;
    int jitterRangeY = 2 * maxJitterY;

    if (chnFormat == RPPI_CHN_PLANAR)
    {      
        srcPtrBeginJitter = srcPtr + (maxJitterY * srcSize.width) + maxJitterX;
        dstPtrBeginJitter = dstPtrForJitter + (maxJitterY * srcSize.width) + maxJitterX;
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtrBeginJitter + (c * srcSize.height * srcSize.width);
            dstPtrTemp = dstPtrBeginJitter + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSize.height - jitterRangeY; i++)
            {
                for (int j = 0; j < srcSize.width - jitterRangeX; j++)
                {
                    jitteredPixelLocDiffX = (rand() % (jitterRangeX + 1));
                    jitteredPixelLocDiffY = (rand() % (jitterRangeY + 1));
                    jitteredPixelLocDiffX -= maxJitterX;
                    jitteredPixelLocDiffY -= maxJitterY;
                    *dstPtrTemp = *(srcPtrTemp + (jitteredPixelLocDiffY * (int) srcSize.width) + jitteredPixelLocDiffX);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtrTemp += jitterRangeX;
                dstPtrTemp += jitterRangeX;
            }
        }

        
        resize_crop_kernel_host<Rpp8u>(static_cast<Rpp8u*>(dstPtrForJitter), srcSize, static_cast<Rpp8u*>(dstPtr), srcSize,
                            maxJitterX, maxJitterY, srcSize.width - maxJitterX - 1, srcSize.height - maxJitterY - 1,
                            RPPI_CHN_PLANAR, channel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        int elementsInRow = (int)(srcSize.width * channel);
        int channeledJitterRangeX = jitterRangeX * channel;
        int channeledJitterRangeY = jitterRangeY * channel;
        srcPtrBeginJitter = srcPtr + (maxJitterY * elementsInRow) + (maxJitterX * channel);
        dstPtrBeginJitter = dstPtrForJitter + (maxJitterY * elementsInRow) + (maxJitterX * channel);
        srcPtrTemp = srcPtrBeginJitter;
        dstPtrTemp = dstPtrBeginJitter;
        for (int i = 0; i < srcSize.height - jitterRangeY; i++)
        {
            for (int j = 0; j < srcSize.width - jitterRangeX; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    jitteredPixelLocDiffX = rand() % (jitterRangeX + 1);
                    jitteredPixelLocDiffY = rand() % (jitterRangeY + 1);
                    jitteredPixelLocDiffX -= maxJitterX;
                    jitteredPixelLocDiffY -= maxJitterY;
                    *dstPtrTemp = *(srcPtrTemp + (jitteredPixelLocDiffY * elementsInRow) + (jitteredPixelLocDiffX * (int) channel));
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
            }
            srcPtrTemp += channeledJitterRangeX;
            dstPtrTemp += channeledJitterRangeX;
        }
        resize_crop_kernel_host<Rpp8u>(static_cast<Rpp8u*>(dstPtrForJitter), srcSize, static_cast<Rpp8u*>(dstPtr), srcSize,
                            maxJitterX, maxJitterY, srcSize.width - maxJitterX - 1, srcSize.height - maxJitterY - 1,
                            RPPI_CHN_PACKED, channel);
    }
    
    return RPP_SUCCESS;
}

/**************** Occlusion Add ***************/

template <typename T>
RppStatus occlusion_host(T* srcPtr1,RppiSize srcSize1,  T* srcPtr2, RppiSize srcSize2, T* dstPtr, 
                            Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                            Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2, 
                            RppiChnFormat chnFormat, unsigned int channel)
{
    RppiSize srcSize1SubImage, srcSize2SubImage, dstSizeSubImage;
    T *srcPtr1SubImage, *srcPtr2SubImage, *dstPtrSubImage;

    compute_subimage_location_host(srcPtr2, &srcPtr2SubImage, srcSize2, &srcSize2SubImage, src2x1, src2y1, src2x2, src2y2, chnFormat, channel);

    T *srcPtrResize = (T*) calloc(channel * srcSize2SubImage.height * srcSize2SubImage.width, sizeof(T));

    generate_crop_host(srcPtr2, srcSize2, srcPtr2SubImage, srcSize2SubImage, srcPtrResize, chnFormat, channel);

    compute_subimage_location_host(srcPtr1, &srcPtr1SubImage, srcSize1, &srcSize1SubImage, src1x1, src1y1, src1x2, src1y2, chnFormat, channel);

    T *dstPtrResize = (T*) calloc(channel * srcSize1SubImage.height * srcSize1SubImage.width, sizeof(T));
    
    resize_kernel_host(srcPtrResize, srcSize2SubImage, dstPtrResize, srcSize1SubImage, chnFormat, channel);

    compute_subimage_location_host(dstPtr, &dstPtrSubImage, srcSize1, &dstSizeSubImage, src1x1, src1y1, src1x2, src1y2, chnFormat, channel);

    T *srcPtr1Temp, *dstPtrTemp;
    srcPtr1Temp = srcPtr1;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize1.height * srcSize1.width); i++)
    {
        *dstPtrTemp = *srcPtr1Temp;
        srcPtr1Temp++;
        dstPtrTemp++;
    }

    T *dstPtrResizeTemp, *dstPtrSubImageTemp;
    dstPtrResizeTemp = dstPtrResize;
    dstPtrSubImageTemp = dstPtrSubImage;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSize1.width - dstSizeSubImage.width;
        for (int c = 0;  c < channel; c++)
        {
            dstPtrSubImageTemp = dstPtrSubImage + (c * srcSize1.height * srcSize1.width);
            for (int i = 0; i < srcSize1SubImage.height; i++)
            {
                for (int j = 0; j < srcSize1SubImage.width; j++)
                {
                    *dstPtrSubImageTemp = *dstPtrResizeTemp;
                    dstPtrResizeTemp++;
                    dstPtrSubImageTemp++;
                }
                dstPtrSubImageTemp += remainingElementsInRow;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = channel * (srcSize1.width - dstSizeSubImage.width);
        for (int i = 0; i < srcSize1SubImage.height; i++)
        {
            for (int j = 0; j < srcSize1SubImage.width; j++)
            {
                for (int c = 0;  c < channel; c++)
                {
                    *dstPtrSubImageTemp = *dstPtrResizeTemp;
                    dstPtrResizeTemp++;
                    dstPtrSubImageTemp++;
                }
            }
            dstPtrSubImageTemp += remainingElementsInRow;
        }
    }
  
    return RPP_SUCCESS;
}

/**************** Snowy ***************/

template <typename T, typename U>
RppStatus snow_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    Rpp32f strength,
                    RppiChnFormat chnFormat, unsigned channel, RppiFormat imageFormat)
{
    if (strength < 0 || strength > 1)
    {
        return RPP_ERROR;
    }

    if (imageFormat == RGB)
    {
        Rpp32f *srcPtrHSL = (Rpp32f *)calloc(channel * srcSize.height * srcSize.width, sizeof(Rpp32f));
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            compute_rgb_to_hsl_host(srcPtr, srcSize, srcPtrHSL, RPPI_CHN_PLANAR, 3);

            Rpp32f *srcPtrHSLTemp;
            srcPtrHSLTemp = srcPtrHSL + (2 * srcSize.height * srcSize.width);

            for (int i = 0; i < srcSize.height * srcSize.width; i++)
            {
                if (*srcPtrHSLTemp < strength)
                {
                    *srcPtrHSLTemp *= 4;
                }
                if (*srcPtrHSLTemp > 1)
                {
                    *srcPtrHSLTemp = 1;
                }
                srcPtrHSLTemp++;
            }

            compute_hsl_to_rgb_host(srcPtrHSL, srcSize, dstPtr, RPPI_CHN_PLANAR, 3);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            compute_rgb_to_hsl_host(srcPtr, srcSize, srcPtrHSL, RPPI_CHN_PACKED, 3);

            Rpp32f *srcPtrHSLTemp;
            srcPtrHSLTemp = srcPtrHSL + 2;

            for (int i = 0; i < srcSize.height * srcSize.width; i++)
            {
                if (*srcPtrHSLTemp < strength)
                {
                    *srcPtrHSLTemp *= 4;
                }
                if (*srcPtrHSLTemp > 1)
                {
                    *srcPtrHSLTemp = 1;
                }
                srcPtrHSLTemp = srcPtrHSLTemp + channel;
            }

            compute_hsl_to_rgb_host(srcPtrHSL, srcSize, dstPtr, RPPI_CHN_PACKED, 3);
        }
    }
    else
    {
        return RPP_ERROR;
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

//Gaussian host function

template <typename T>
RppStatus noise_gaussian_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                        Rpp32f mean, Rpp32f sigma, 
                        RppiChnFormat chnFormat, unsigned int channel)
{
    std::default_random_engine generator;
    std::normal_distribution<>  distribution{mean, sigma}; 
    for(int i = 0; i < (srcSize.height * srcSize.width * channel) ; i++)
    {
        Rpp32f pixel = ((Rpp32f) *srcPtr) + ((Rpp32f) distribution(generator));
		*dstPtr = RPPPIXELCHECK(pixel); 
        dstPtr++;
        srcPtr++;       
    }
    return RPP_SUCCESS;
}

//Salt and Pepper Host function

template <typename T>
RppStatus noise_snp_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                        Rpp32f noiseProbability, 
                        RppiChnFormat chnFormat, unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]);
        dstPtr[i] = RPPPIXELCHECK(pixel);
    }
    srand(time(0)); 
    Rpp32u noisePixel= (Rpp32u)(noiseProbability * srcSize.width * srcSize.height );
    Rpp32u pixelDistance= (srcSize.width * srcSize.height) / noisePixel;
    if(chnFormat==RPPI_CHN_PACKED)
    {
        for(int i=0 ; i<srcSize.width * srcSize.height *channel ; i+=channel*pixelDistance)
        {
            Rpp32u initialPixel= rand() % pixelDistance;
            dstPtr += initialPixel*channel;
            Rpp8u newPixel=rand()%2 ? 0 : 255;
            std::cout<<(int)newPixel<<" ";
            for(int j=0 ; j<channel ; j++)
            {
                *dstPtr=newPixel;
                dstPtr++;
            }
            dstPtr+= ((pixelDistance - initialPixel - 1) * channel);
        }
    }
    else if(chnFormat==RPPI_CHN_PLANAR)
    {
        if(channel==3)
        {
            Rpp8u *dstPtrTemp1,*dstPtrTemp2;
            dstPtrTemp1 = dstPtr + (srcSize.height * srcSize.width);
            dstPtrTemp2 = dstPtr + (2 * srcSize.height * srcSize.width);   
            for(int i=0 ; i<srcSize.width * srcSize.height * channel ; i+=pixelDistance)
            {
                Rpp32u initialPixel= rand() % pixelDistance;
                dstPtr += initialPixel;
                Rpp8u newPixel=rand()%2 ? 255 : 1;
                *dstPtr=newPixel;
                dstPtr+= ((pixelDistance - initialPixel - 1));

                dstPtrTemp1 += initialPixel;
                *dstPtrTemp1=newPixel;
                dstPtrTemp1+= ((pixelDistance - initialPixel - 1));

                dstPtrTemp2 += initialPixel;
                *dstPtrTemp2=newPixel;
                dstPtrTemp2+= ((pixelDistance - initialPixel - 1));
                
            }
        }
        else
        {
            for(int i=0 ; i<srcSize.width * srcSize.height ; i+=pixelDistance)
            {
                Rpp32u initialPixel= rand() % pixelDistance;
                dstPtr += initialPixel;
                Rpp8u newPixel=rand()%2 ? 255 : 1;
                *dstPtr=newPixel;
                dstPtr+= ((pixelDistance - initialPixel - 1));
            }   
        }
        
    }
    return RPP_SUCCESS;
}

/**************** Random Shadow ***************/

template <typename T>
RppStatus random_shadow_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                             Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                             Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY, 
                             RppiChnFormat chnFormat, unsigned int channel)
{
    srand (time(NULL));
    RppiSize srcSizeSubImage, dstSizeSubImage, shadowSize;
    T *srcPtrSubImage, *dstPtrSubImage;
    
    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    compute_subimage_location_host(dstPtr, &dstPtrSubImage, srcSize, &dstSizeSubImage, x1, y1, x2, y2, chnFormat, channel);
    
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = *srcPtrTemp;
        srcPtrTemp++;
        dstPtrTemp++;
    }

    for (int shadow = 0; shadow < numberOfShadows; shadow++)
    {
        shadowSize.height = rand() % (srcSizeSubImage.height / 2);
        shadowSize.width = rand() % (srcSizeSubImage.width / 2);
        Rpp32u shadowPosI = rand() % (srcSizeSubImage.height - shadowSize.height);
        Rpp32u shadowPosJ = rand() % (srcSizeSubImage.width - shadowSize.width);
        
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            Rpp32u remainingElementsInRow = srcSize.width - shadowSize.width;
            for (int c = 0; c < channel; c++)
            {
                dstPtrTemp = dstPtrSubImage + (c * srcSize.height * srcSize.width) + (shadowPosI * srcSize.width) + shadowPosJ;
                srcPtrTemp = srcPtrSubImage + (c * srcSize.height * srcSize.width) + (shadowPosI * srcSize.width) + shadowPosJ;

                for (int i = 0; i < shadowSize.height; i++)
                {
                    for (int j = 0; j < shadowSize.width; j++)
                    {
                        *dstPtrTemp = *srcPtrTemp / 2;
                        dstPtrTemp++;
                        srcPtrTemp++;
                    }
                    dstPtrTemp += remainingElementsInRow;
                    srcPtrTemp += remainingElementsInRow;
                }
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            dstPtrTemp = dstPtrSubImage + (channel * ((shadowPosI * srcSize.width) + shadowPosJ));
            srcPtrTemp = srcPtrSubImage + (channel * ((shadowPosI * srcSize.width) + shadowPosJ));
            Rpp32u remainingElementsInRow = channel * (srcSize.width - shadowSize.width);
            for (int i = 0; i < shadowSize.height; i++)
            {
                for (int j = 0; j < shadowSize.width; j++)
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = *srcPtrTemp / 2;
                        dstPtrTemp++;
                        srcPtrTemp++;
                    }
                }
                dstPtrTemp += remainingElementsInRow;
                srcPtrTemp += remainingElementsInRow;
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
    if(fogValue<=0)
    {
        for(int i=0;i<srcSize.height*srcSize.width*channel;i++)
        {
            *srcPtr=*temp;
            srcPtr++;
            temp++;
        }
    }
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp8u *srcPtr1,*srcPtr2;
        if(channel>1)
        {
            srcPtr1=srcPtr + (srcSize.width * srcSize.height);
            srcPtr2=srcPtr + (srcSize.width * srcSize.height*2);
        }
        for (int i = 0; i < (srcSize.width * srcSize.height); i++)
        {
            Rpp32f check= *srcPtr + *srcPtr1 + *srcPtr2;
            if(check >= (240*3) && fogValue!=0)
            {            }
            else if(check>=(170*3))
            {
                Rpp32f pixel = ((Rpp32f) *srcPtr)  * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
                *srcPtr = (Rpp8u)RPPPIXELCHECK(pixel);
                srcPtr++;
                if(channel>1)
                {
                    pixel = ((Rpp32f) *srcPtr1) * (1.5 + fogValue) + (7*fogValue);
                    *srcPtr1 = (Rpp8u)RPPPIXELCHECK(pixel);
                    pixel = ((Rpp32f) *srcPtr2) * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
                    *srcPtr2 = (Rpp8u)RPPPIXELCHECK(pixel);
				    srcPtr1++;srcPtr2++;
                }
            }
            else if(check<=(85*3))
            {
                Rpp32f pixel = ((Rpp32f) *srcPtr) * (1.5 + pow(fogValue,2)) - (fogValue*4) + (130*fogValue);
                *srcPtr = (Rpp8u)RPPPIXELCHECK(pixel);
                srcPtr++;
                if(channel>1)
                {
                    pixel = ((Rpp32f) *srcPtr1) * (1.5 + pow(fogValue,2)) + (130*fogValue);
                    *srcPtr1 = (Rpp8u)RPPPIXELCHECK(pixel);
                    pixel = ((Rpp32f) *srcPtr2) * (1.5 + pow(fogValue,2)) + (fogValue*4) + 130*fogValue;
                    *srcPtr2 = (Rpp8u)RPPPIXELCHECK(pixel);
                    srcPtr1++;srcPtr2++;
                }
            }
            else
            {
                Rpp32f pixel = ((Rpp32f) *srcPtr) * (1.5 + pow(fogValue,1.5)) - (fogValue*4) + 20 + (100*fogValue);
                *srcPtr = (Rpp8u)RPPPIXELCHECK(pixel);
                srcPtr++;
                if(channel>1)
                {
                    pixel = ((Rpp32f) *srcPtr1) * (1.5 + pow(fogValue,1.5)) + 20 + (100*fogValue);
                    *srcPtr1 = (Rpp8u)RPPPIXELCHECK(pixel);
                    pixel = ((Rpp32f) *srcPtr2) * (1.5 + pow(fogValue,1.5)) + (fogValue*4) + (100*fogValue);
                    *srcPtr2 = (Rpp8u)RPPPIXELCHECK(pixel);
                    srcPtr1++;srcPtr2++;
                }
            }
        }
    }
    else
    {
        Rpp8u *srcPtr1,*srcPtr2;
        srcPtr1=srcPtr+1;
        srcPtr2=srcPtr+2;
        for (int i = 0; i < (srcSize.width * srcSize.height * channel); i+=3)
        {
            Rpp32f check=*srcPtr + *srcPtr1 + *srcPtr2;
            if(check >= (240*3) && fogValue!=0)
            {            }
            else if(check>=(170*3) && fogValue!=0)
            {
                Rpp32f pixel = ((Rpp32f) *srcPtr) * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
                *srcPtr = (Rpp8u)RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) *srcPtr1) * (1.5 + fogValue) + (7*fogValue);
                *srcPtr1 =(Rpp8u) RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) *srcPtr2) * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
                *srcPtr2 = (Rpp8u)RPPPIXELCHECK(pixel);
            }
            else if(check<=(85*3) && fogValue!=0)
            {
                Rpp32f pixel = ((Rpp32f) *srcPtr) * (1.5 + pow(fogValue,2)) - (fogValue*4) + (130*fogValue);
                *srcPtr = (Rpp8u)RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) *srcPtr1) * (1.5 + pow(fogValue,2)) + (130*fogValue);
                *srcPtr1 = (Rpp8u)RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) *srcPtr2) * (1.5 + pow(fogValue,2)) + (fogValue*4) + 130*fogValue;
                *srcPtr2 = (Rpp8u)RPPPIXELCHECK(pixel);
            }
            else if(fogValue!=0)
            {
                Rpp32f pixel = ((Rpp32f) *srcPtr) * (1.5 + pow(fogValue,1.5)) - (fogValue*4) + 20 + (100*fogValue);
                *srcPtr = (Rpp8u)RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) *srcPtr1) * (1.5 + pow(fogValue,1.5)) + 20 + (100*fogValue);
                *srcPtr1 = (Rpp8u)RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) *srcPtr2) * (1.5 + pow(fogValue,1.5)) + (fogValue*4) + (100*fogValue);
                *srcPtr2 = (Rpp8u)RPPPIXELCHECK(pixel);
            }
			srcPtr+=3;
			srcPtr1+=3;
			srcPtr2+=3;
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
    rainPercentage=rainPercentage/250;
    transparency/=5;

    Rpp32u rainDrops= (Rpp32u)(rainPercentage * srcSize.width * srcSize.height * channel );
    
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int i = 0 ; i < rainDrops ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp32f pixel;
            for(int k=0;k<channel;k++)
            {
                //pixel=(Rpp32f)dstPtr[(row * srcSize.width) + (column) + (k*srcSize.height*srcSize.width)] + 5;
                dstPtr[(row * srcSize.width) + (column) + (k*srcSize.height*srcSize.width)] = (k==0)?196:(k==1)?226:255 ;
            }
            if (row+rainHeight < srcSize.height && column+rainWidth< srcSize.width)
            {
                for(int j=1;j<rainHeight;j++)
                {
                    for(int k=0;k<channel;k++)
                    {
                        for(int m=0;m<rainWidth;m++)
                        {
                            //pixel=(Rpp32f)dstPtr[(row * srcSize.width) + (column) + (k*srcSize.height*srcSize.width) + (srcSize.width*j)+m]+5;
                            dstPtr[(row * srcSize.width) + (column) + (k*srcSize.height*srcSize.width) + (srcSize.width*j)+m] = (k==0)?196:(k==1)?226:255 ;
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
            for(int k=0;k<channel;k++)
            {
                //pixel=(Rpp32f)dstPtr[(channel * row * srcSize.width) + (column * channel) + k] + 5;
                dstPtr[(channel * row * srcSize.width) + (column * channel) + k] = (k==0)?196:(k==1)?226:255 ;
            }
            if (row+rainHeight < srcSize.height && column+rainWidth< srcSize.width)
            {
                for(int j=1;j<rainHeight;j++)
                {
                    for(int k=0;k<channel;k++)
                    {
                        for(int m=0;m<rainWidth;m++)
                        {
                            //pixel=(Rpp32f)dstPtr[(channel * row * srcSize.width) + (column * channel) + k + (channel * srcSize.width * j)+(channel*m)]+5;
                            dstPtr[(channel * row * srcSize.width) + (column * channel) + k + (channel * srcSize.width * j)+(channel*m)] = (k==0)?196:(k==1)?226:255 ;
                        } 
                    }            
                }
            }
        }
    }

        for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i])+ transparency * dstPtr[i];
        dstPtr[i] = RPPPIXELCHECK(pixel);
    }

}





/**************** Random Crop Letterbox ***************/

template <typename T>
RppStatus random_crop_letterbox_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                                     Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                                     RppiChnFormat chnFormat, unsigned int channel)
{
    if ((RPPINRANGE(x1, 0, srcSize.width - 1) == 0) 
        || (RPPINRANGE(x2, 0, srcSize.width - 1) == 0) 
        || (RPPINRANGE(y1, 0, srcSize.height - 1) == 0) 
        || (RPPINRANGE(y2, 0, srcSize.height - 1) == 0))
    {
        return RPP_ERROR;
    }

    RppiSize srcNewSize;
    int xDiff = (int) x2 - (int) x1;
    int yDiff = (int) y2 - (int) y1;
    srcNewSize.width = (Rpp32u) RPPABS(xDiff);
    srcNewSize.height = (Rpp32u) RPPABS(yDiff);
    
    T *srcNewPtr = (T *)calloc(channel * srcNewSize.height * srcNewSize.width, sizeof(T));
    T *srcPtrTemp, *srcNewPtrTemp;
    srcPtrTemp = srcPtr;
    srcNewPtrTemp = srcNewPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp += (c * srcSize.height * srcSize.width);
            srcPtrTemp += ((RPPMIN2(y1, y2) * srcSize.width) + RPPMIN2(x1, x2));
            for (int i = RPPMIN2(y1, y2); i < RPPMAX2(y1, y2); i++)
            {
                for (int j = RPPMIN2(x1, x2); j < RPPMAX2(x1, x2); j++)
                {
                    *srcNewPtrTemp = *srcPtrTemp;
                    srcNewPtrTemp++;
                    srcPtrTemp++;
                }
                srcPtrTemp += (srcSize.width - srcNewSize.width);
            }
            srcPtrTemp = srcPtr;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        srcPtrTemp += (RPPMIN2(y1, y2) * elementsInRow) + (RPPMIN2(x1, x2) * channel);
        for (int i = RPPMIN2(y1, y2); i < RPPMAX2(y1, y2); i++)
        {
            for (int j = RPPMIN2(x1, x2); j < RPPMAX2(x1, x2); j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *srcNewPtrTemp = *srcPtrTemp;
                    srcNewPtrTemp++;
                    srcPtrTemp++;
                }
            }
            srcPtrTemp += ((srcSize.width - srcNewSize.width) * channel);
        }
    }

    generate_evenly_padded_image_host(srcNewPtr, srcNewSize, dstPtr, dstSize, chnFormat, channel);

    return RPP_SUCCESS;
    
}

/**************** Exposure Modification ***************/

template <typename T, typename U>
RppStatus exposure_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    Rpp32f exposureFactor,
                    RppiChnFormat chnFormat, unsigned channel, RppiFormat imageFormat)
{
    Rpp32f pixel;
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (imageFormat == RGB)
    {
        for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
        {
            pixel = *srcPtrTemp * (pow(2, exposureFactor));
            pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
            pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
            *dstPtrTemp = (T) round(pixel);
            dstPtrTemp++;
            srcPtrTemp++;
        }
    }
    else if (imageFormat == HSV)
    {
        exposureFactor = RPPABS(exposureFactor);
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            for (int i = 0; i < ((channel - 1) * (srcSize.width * srcSize.height)); i++)
            {
                *dstPtrTemp = *srcPtrTemp;
                srcPtrTemp++;
                dstPtrTemp++;
            }
            for (int i = 0; i < (srcSize.width * srcSize.height); i++)
            {
                pixel = *srcPtrTemp * exposureFactor;
                pixel = (pixel < (Rpp32f) 1) ? pixel : ((Rpp32f) 1);
                pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
                *dstPtrTemp = pixel;
                dstPtrTemp++;
                srcPtrTemp++;
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            int count = 0;
            for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
            {
                if (count == 2)
                {
                    pixel = *srcPtrTemp * exposureFactor;
                    pixel = (pixel < (Rpp32f) 1) ? pixel : ((Rpp32f) 1);
                    pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
                    *dstPtrTemp = pixel;
                    dstPtrTemp++;
                    srcPtrTemp++;
                    count = 0;
                }
                else
                {
                    *dstPtrTemp = *srcPtrTemp;
                    dstPtrTemp++;
                    srcPtrTemp++;
                    count++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}



