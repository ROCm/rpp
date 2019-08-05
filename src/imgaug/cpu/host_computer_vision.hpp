#include <cpu/rpp_cpu_common.hpp>

/**************** Local Binary Pattern ***************/

template <typename T>
RppStatus local_binary_pattern_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u kernelSize = 3;
    Rpp32u bound = ((kernelSize - 1) / 2);

    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + (2 * bound);
    srcSizeMod.height = srcSize.height + (2 * bound);
    Rpp8u *srcPtrMod = (Rpp8u *)calloc(srcSizeMod.height * srcSizeMod.width * channel, sizeof(Rpp8u));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    T *srcPtrWindow, *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u rowIncrementForWindow = kernelSize - 1;
        Rpp32u channelIncrementForWindow = (kernelSize - 1) * srcSizeMod.width;
        Rpp32u remainingElementsInRow = srcSizeMod.width - kernelSize;
        Rpp32u centerPixelIncrement = srcSize.width + 1;

        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    local_binary_pattern_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                    remainingElementsInRow, srcPtrWindow + centerPixelIncrement, 
                                    chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += (rowIncrementForWindow);
            }
            srcPtrWindow += (channelIncrementForWindow);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u rowIncrementForWindow = (kernelSize - 1) * channel;
        Rpp32u remainingElementsInRow = (srcSizeMod.width - kernelSize) * channel;
        Rpp32u centerPixelIncrement = channel * (srcSize.width + 1);

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {   
                    local_binary_pattern_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                    remainingElementsInRow, srcPtrWindow + centerPixelIncrement, 
                                    chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += (rowIncrementForWindow);
        }
    }

    return RPP_SUCCESS;
}

/**************** Data Object Copy ***************/

template <typename T>
RppStatus data_object_copy_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    memcpy(dstPtr, srcPtr, srcSize.height * srcSize.width * channel * sizeof(T));
    
    return RPP_SUCCESS;
}

/**************** Gaussian Image Pyramid ***************/

template <typename T>
RppStatus gaussian_image_pyramid_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
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
    T *srcPtrConvolved = (T *)calloc(channel * srcSize.height * srcSize.width, sizeof(T));
    convolve_image_host(srcPtrMod, srcSizeMod, srcPtrConvolved, srcSize, kernel, rppiKernelSize, chnFormat, channel);

    RppiSize dstSize;
    dstSize.height = (srcSize.height + 1) / 2;
    dstSize.width = (srcSize.width + 1) / 2;

    compute_downsampled_image_host(srcPtrConvolved, srcSize, dstPtr, dstSize, chnFormat, channel);
    
    return RPP_SUCCESS;
}