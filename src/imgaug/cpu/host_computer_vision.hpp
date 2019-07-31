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