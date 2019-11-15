#include <cpu/rpp_cpu_common.hpp>


/**************** Color Temperature ***************/

template <typename T>
RppStatus color_temperature_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp8s adjustmentValue,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (channel != 1 && channel !=  3)
    {
        return RPP_ERROR;
    }
    if (adjustmentValue < -100 || adjustmentValue > 100)
    {
        return RPP_ERROR;
    }   

    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;


    if (channel == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32s pixel = ((Rpp32s)srcPtrTemp[i]) + (Rpp32s) adjustmentValue;
            pixel = RPPPIXELCHECK(pixel);
            dstPtrTemp[i] = (T) pixel;
        }
    }
    else if (channel == 3)
    {   
        if (chnFormat == RPPI_CHN_PLANAR)
        {
#pragma omp parallel for
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                Rpp32s pixel = ((Rpp32s)srcPtrTemp[i]) + (Rpp32s) adjustmentValue;
                pixel = RPPPIXELCHECK(pixel);
                dstPtrTemp[i] = (T) pixel;
            }
            dstPtrTemp += (srcSize.height * srcSize.width);
            srcPtrTemp += (srcSize.height * srcSize.width);
#pragma omp parallel for
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                dstPtrTemp[i] = srcPtrTemp[i];
            }
            dstPtrTemp += (srcSize.height * srcSize.width);
            srcPtrTemp += (srcSize.height * srcSize.width);
#pragma omp parallel for
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                Rpp32s pixel = ((Rpp32s)srcPtrTemp[i]) - (Rpp32s) adjustmentValue;
                pixel = RPPPIXELCHECK(pixel);
                dstPtrTemp[i] = (T) pixel;

            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
#pragma omp parallel for
            for (int i = 0; i < (srcSize.height * srcSize.width*3); i +=3 )
            {
                Rpp32s pixel = ((Rpp32s)srcPtrTemp[i])+ (Rpp32s) adjustmentValue;
                dstPtrTemp[i] = (T) RPPPIXELCHECK(pixel);
                dstPtrTemp[i+1] = srcPtrTemp[i+1];
                pixel = ((Rpp32s)srcPtrTemp[i+2])- (Rpp32s) adjustmentValue;
                dstPtrTemp[i+2] = (T) RPPPIXELCHECK(pixel);
            }
        }
    }
    
    return RPP_SUCCESS;
}

/**************** Vignette ***************/

template <typename T>
RppStatus vignette_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f stdDev,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f *mask = (Rpp32f *)calloc(srcSize.height * srcSize.width, sizeof(Rpp32f));
    Rpp32f *maskTemp;
    maskTemp = mask;

    RppiSize kernelRowsSize, kernelColumnsSize;
    kernelRowsSize.height = srcSize.height;
    kernelRowsSize.width = 1;
    kernelColumnsSize.height = srcSize.width;
    kernelColumnsSize.width = 1;

    Rpp32f *kernelRows = (Rpp32f *)calloc(kernelRowsSize.height * kernelRowsSize.width, sizeof(Rpp32f));
    Rpp32f *kernelColumns = (Rpp32f *)calloc(kernelColumnsSize.height * kernelColumnsSize.width, sizeof(Rpp32f));

    if (kernelRowsSize.height % 2 == 0)
    {
        generate_gaussian_kernel_asymmetric_host(stdDev, kernelRows, kernelRowsSize.height - 1, kernelRowsSize.width);
        kernelRows[kernelRowsSize.height - 1] = kernelRows[kernelRowsSize.height - 2];
    }
    else
    {
        generate_gaussian_kernel_asymmetric_host(stdDev, kernelRows, kernelRowsSize.height, kernelRowsSize.width);
    }
    
    if (kernelColumnsSize.height % 2 == 0)
    {
        generate_gaussian_kernel_asymmetric_host(stdDev, kernelColumns, kernelColumnsSize.height - 1, kernelColumnsSize.width);
        kernelColumns[kernelColumnsSize.height - 1] = kernelColumns[kernelColumnsSize.height - 2];
    }
    else
    {
        generate_gaussian_kernel_asymmetric_host(stdDev, kernelColumns, kernelColumnsSize.height, kernelColumnsSize.width);
    }


#pragma omp parallel for
    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            maskTemp[i*srcSize.width+ j] = kernelRows[i] * kernelColumns[j];
        }
    }

    Rpp32f max = 0;
    maskTemp = mask;
    for (int i = 0; i < (srcSize.height * srcSize.width); i++)
    {
        if (*maskTemp> max)
        {
            max = *maskTemp;
        }
    }

    maskTemp = mask;
#pragma omp parallel for
    for (int i = 0; i < (srcSize.height * srcSize.width); i++)
    {
        maskTemp[i] /= max;
    }

    Rpp32f *maskFinal = (Rpp32f *)calloc(channel * srcSize.height * srcSize.width, sizeof(Rpp32f));
    Rpp32f *maskFinalTemp;
    maskFinalTemp = maskFinal;
    maskTemp = mask;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            maskTemp = mask;
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    *maskFinalTemp = *maskTemp;
                    maskFinalTemp++;
                    maskTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
#pragma omp parallel for
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    maskFinalTemp[i*srcSize.width*channel + j*channel+c] = maskTemp[i*srcSize.width+j];
                }
            }
        }
    }

    compute_multiply_host(srcPtr, maskFinal, srcSize, dstPtr, channel);
    free(maskFinal);
    free(kernelRows);
    free(kernelColumns);
    free(mask);

    return RPP_SUCCESS;
}