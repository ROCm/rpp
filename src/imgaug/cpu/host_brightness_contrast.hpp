
template <typename T>
RppStatus host_brightness_contrast(T* srcPtr, RppiSize size, T* dstPtr,
                                Rpp32f alpha, Rpp32f beta, RppiChnFormat chnFormat)
{
    //logic is planar/packed independent
    for (int i = 0; i < (size.width * size.height); i++)
    {
        dstPtr[i] = srcPtr[i] * alpha + beta;
    }

    return RPP_SUCCESS;

}