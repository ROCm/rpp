#include <cpu/rpp_cpu_common.hpp>

/**************** Bilateral Filter ***************/

template <typename T>
RppStatus bilateral_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32u kernelSize, Rpp32f sigmaI, Rpp32f sigmaS,
                                RppiChnFormat chnFormat, Rpp32u channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }

    Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    int bound = ((kernelSize - 1) / 2);

    RppiSize srcSizeMod;
    srcSizeMod.height = srcSize.height + (2 * bound);
    srcSizeMod.width = srcSize.width + (2 * bound);
    T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(T));
    
    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    Rpp32u remainingElementsInRowPlanar = srcSizeMod.width - kernelSize;
    Rpp32u remainingElementsInRowPacked = (srcSizeMod.width - kernelSize) * channel;
    Rpp32u incrementToWindowCenterPlanar = (bound * srcSizeMod.width) + bound;
    Rpp32u incrementToWindowCenterPacked = ((bound * srcSizeMod.width) + bound) * channel;
    Rpp32f multiplierI, multiplierS, multiplier;
    multiplierI = -1 / (2 * sigmaI * sigmaI);
    multiplierS = -1 / (2 * sigmaS * sigmaS);
    multiplier = 1 / (4 * M_PI * M_PI * sigmaI * sigmaI * sigmaS * sigmaS);
    
    T *srcPtrWindow, *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;

    T maxVal = (T)(std::numeric_limits<T>::max());
    T minVal = (T)(std::numeric_limits<T>::min());

    RppiSize rppiKernelSize;
    rppiKernelSize.height = kernelSize;
    rppiKernelSize.width = kernelSize;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    generate_bilateral_kernel_host<T>(multiplierI, multiplierS, multiplier, kernel, kernelSize, bound, 
                                                      srcPtrWindow, srcSizeMod, remainingElementsInRowPlanar, incrementToWindowCenterPlanar, 
                                                      chnFormat, channel);
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                            kernel, rppiKernelSize, remainingElementsInRowPlanar, maxVal, minVal, 
                                            chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += (kernelSize - 1);
            }
            srcPtrWindow += ((kernelSize - 1) * srcSizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {   
                    generate_bilateral_kernel_host<T>(multiplierI, multiplierS, multiplier, kernel, kernelSize, bound, 
                                                      srcPtrWindow, srcSizeMod, remainingElementsInRowPacked, incrementToWindowCenterPacked, 
                                                      chnFormat, channel);
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                            kernel, rppiKernelSize, remainingElementsInRowPacked, maxVal, minVal, 
                                            chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += ((kernelSize - 1) * channel);
        }
    }
    
    return RPP_SUCCESS;

}


/**************** Box Filter ***************/

template <typename T>
RppStatus box_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u kernelSize,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }
    Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    int bound = ((kernelSize - 1) / 2);

    generate_box_kernel_host(kernel, kernelSize);

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


/**************** Sobel Filter ***************/

template <typename T>
RppStatus sobel_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                            Rpp32u sobelType, 
                            RppiChnFormat chnFormat, Rpp32u channel)
{
    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + 2;
    srcSizeMod.height = srcSize.height + 2;

    T *srcPtrMod = (T *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(T));
    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    Rpp32s *dstPtrIntermediate = (Rpp32s *)calloc(srcSize.height * srcSize.width * channel, sizeof(Rpp32s));

    RppiSize rppiKernelSize;
    rppiKernelSize.height = 3;
    rppiKernelSize.width = 3;

    if (sobelType == 0)
    {
        Rpp32f *kernelX = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
        generate_sobel_kernel_host(kernelX, 1);
        convolve_image_host(srcPtrMod, srcSizeMod, dstPtrIntermediate, srcSize, kernelX, rppiKernelSize, chnFormat, channel);
    }
    else if (sobelType == 1)
    {
        Rpp32f *kernelY = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
        generate_sobel_kernel_host(kernelY, 2);
        convolve_image_host(srcPtrMod, srcSizeMod, dstPtrIntermediate, srcSize, kernelY, rppiKernelSize, chnFormat, channel);
    }
    else if (sobelType == 2)
    {
        Rpp32f *kernelX = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
        generate_sobel_kernel_host(kernelX, 1);
        Rpp32s *dstPtrIntermediateX = (Rpp32s *)calloc(srcSize.height * srcSize.width * channel, sizeof(Rpp32s));
        convolve_image_host(srcPtrMod, srcSizeMod, dstPtrIntermediateX, srcSize, kernelX, rppiKernelSize, chnFormat, channel);

        Rpp32f *kernelY = (Rpp32f *)calloc(3 * 3, sizeof(Rpp32f));
        generate_sobel_kernel_host(kernelY, 2);
        Rpp32s *dstPtrIntermediateY = (Rpp32s *)calloc(srcSize.height * srcSize.width * channel, sizeof(Rpp32s));
        convolve_image_host(srcPtrMod, srcSizeMod, dstPtrIntermediateY, srcSize, kernelY, rppiKernelSize, chnFormat, channel);

        compute_magnitude_host(dstPtrIntermediateX, dstPtrIntermediateY, srcSize, dstPtrIntermediate, chnFormat, channel);
    }

    compute_threshold_host(dstPtrIntermediate, srcSize, dstPtr, (T) 10, (T) 255, 2, chnFormat, channel);

    return RPP_SUCCESS;
}


/**************** Median Filter ***************/

template <typename T>
RppStatus median_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u kernelSize,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }
    int bound = ((kernelSize - 1) / 2);

    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + (2 * bound);
    srcSizeMod.height = srcSize.height + (2 * bound);
    Rpp8u *srcPtrMod = (Rpp8u *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(Rpp8u));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);
    
    Rpp32u remainingElementsInRowPlanar = srcSizeMod.width - kernelSize;
    Rpp32u remainingElementsInRowPacked = (srcSizeMod.width - kernelSize) * channel;
    
    T *srcPtrWindow, *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    median_filter_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                      kernelSize, remainingElementsInRowPlanar, 
                                      chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += (kernelSize - 1);
            }
            srcPtrWindow += ((kernelSize - 1) * srcSizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {   
                    median_filter_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                      kernelSize, remainingElementsInRowPacked, 
                                      chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += ((kernelSize - 1) * channel);
        }
    }
    
    return RPP_SUCCESS;
}


/**************** Custom Convolution ***************/

template <typename T>
RppStatus custom_convolution_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                  Rpp32f *kernel, RppiSize rppiKernelSize, Rpp32f scale, 
                                  RppiChnFormat chnFormat, Rpp32u channel)
{
    if (rppiKernelSize.height % 2 == 0 || rppiKernelSize.width % 2 == 0)
    {
        return RPP_ERROR;
    }

    int boundY = ((rppiKernelSize.height - 1) / 2);
    int boundX = ((rppiKernelSize.width - 1) / 2);

    RppiSize srcSizeMod1, srcSizeMod2;

    srcSizeMod1.height = srcSize.height + boundY;
    srcSizeMod1.width = srcSize.width + boundX;
    T *srcPtrMod1 = (T *)calloc(srcSizeMod1.height * srcSizeMod1.width * channel, sizeof(T));
    generate_corner_padded_image_host(srcPtr, srcSize, srcPtrMod1, srcSizeMod1, 1, chnFormat, channel);

    srcSizeMod2.height = srcSizeMod1.height + boundY;
    srcSizeMod2.width = srcSizeMod1.width + boundX;
    T *srcPtrMod2 = (T *)calloc(srcSizeMod2.height * srcSizeMod2.width * channel, sizeof(T));
    generate_corner_padded_image_host(srcPtrMod1, srcSizeMod1, srcPtrMod2, srcSizeMod2, 4, chnFormat, channel);

    Rpp32f *kernelTemp;
    kernelTemp = kernel;
    for (int i = 0; i < (rppiKernelSize.height * rppiKernelSize.width); i++)
    {
        *kernelTemp = *kernelTemp / scale;
        kernelTemp++;
    }
    
    convolve_image_host(srcPtrMod2, srcSizeMod2, dstPtr, srcSize, kernel, rppiKernelSize, chnFormat, channel);
    
    return RPP_SUCCESS;
}