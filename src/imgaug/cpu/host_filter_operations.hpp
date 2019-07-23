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
    T *srcPtrMod = (T *)calloc(srcSizeMod.width * srcSizeMod.height * channel, sizeof(T));
    
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
                                            kernel, kernelSize, remainingElementsInRowPlanar, 
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
                                            kernel, kernelSize, remainingElementsInRowPacked, 
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
    T *srcPtrMod = (T *)calloc(srcSizeMod.width * srcSizeMod.height * channel, sizeof(T));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);
    
    convolve_image_host(srcPtrMod, srcSizeMod, dstPtr, srcSize, kernel, kernelSize, chnFormat, channel);
    
    return RPP_SUCCESS;
}