
template <typename T>
RppStatus host_brightness_contrast(T* pSrc, RppiSize size, T* pDst,
                                Rpp32f alpha, Rpp32f beta, RppiChnFormat chnFormat)
{
    //logic is planar/packed independent
    for (int i = 0; i < (size.width * size.height); i++)
    {
        pDst[i] = pSrc[i] * alpha + beta;
    }

    return RPP_SUCCESS;

}