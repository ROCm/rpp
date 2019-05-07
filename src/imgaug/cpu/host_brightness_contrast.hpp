
template <typename T>
RppStatus host_brightness_contrast(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32f alpha, Rpp32f beta, RppiChnFormat chnFormat)
{
    //logic is planar/packed independent
    for (int i = 0; i < (srcSize.width * srcSize.height); i++)
    {
        dstPtr[i] = srcPtr[i] * alpha + beta;
    }

    return RPP_SUCCESS;

}