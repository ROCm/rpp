#include <cpu/rpp_cpu_common.hpp>

/**************** Bilateral Filter ***************/

template <typename T>
RppStatus bilateral_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32u kernelSize, Rpp64f sigmaI, Rpp64f sigmaS,
                                RppiChnFormat chnFormat, unsigned int channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }

    Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    int bound = ((kernelSize - 1) / 2);

    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + (2 * bound);
    srcSizeMod.height = srcSize.height + (2 * bound);
    Rpp8u *srcPtrMod = (Rpp8u *)calloc(srcSizeMod.width * srcSizeMod.height * channel, sizeof(Rpp8u));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);

    int remainingElementsInRowPlanar = srcSizeMod.width - kernelSize;
    int remainingElementsInRowPacked = (srcSizeMod.width - kernelSize) * channel;
    
    T *srcPtrWindow, *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u rowEndIncrement = srcSizeMod.width - kernelSize;
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    generate_bilateral_kernel_host<T>(sigmaI, sigmaS, kernel, kernelSize, srcPtrWindow, srcSizeMod, rowEndIncrement, chnFormat, channel);
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                                 kernel, kernelSize, remainingElementsInRowPlanar, remainingElementsInRowPacked, 
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
        Rpp32u rowEndIncrement = channel * (srcSizeMod.width - kernelSize);
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {   
                    generate_bilateral_kernel_host<T>(sigmaI, sigmaS, kernel, kernelSize, srcPtrWindow, srcSizeMod, rowEndIncrement, chnFormat, channel);
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                                 kernel, kernelSize, remainingElementsInRowPlanar, remainingElementsInRowPacked, 
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += ((kernelSize - 1) * channel);
        }
    }