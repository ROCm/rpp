
template <typename T>
RppStatus host_brightness_contrast(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32f alpha, Rpp32f beta, int channel, RppiChnFormat chnFormat)
{
    //logic is planar/packed independent
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        dstPtr[i] = srcPtr[i] * alpha + beta;
    }

    return RPP_SUCCESS;

}
