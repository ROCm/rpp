#include <algorithm>

template <typename T>
RppStatus host_brightness_contrast(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32f alpha, Rpp32f beta, unsigned int channel, RppiChnFormat chnFormat)
{
    //logic is planar/packed independent
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]) * alpha + beta;
        pixel = std::min(pixel, (Rpp32f) 255);
        pixel = std::max(pixel, (Rpp32f) 0);
        dstPtr[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}
