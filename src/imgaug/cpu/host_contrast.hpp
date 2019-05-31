template <typename T>
RppStatus host_contrast(T* srcPtr, RppiSize srcSize, T* dstPtr, 
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