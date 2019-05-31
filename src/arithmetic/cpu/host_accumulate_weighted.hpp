#include "host_declarations.hpp"

template <typename T>
RppStatus host_accumulate_weighted(T* srcPtr1, T* srcPtr2, RppiSize srcSize,
                                   Rpp32f alpha,
                                   unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((1 - alpha) * ((Rpp32f) srcPtr1[i])) + (alpha * ((Rpp32s) srcPtr2[i]));
        pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
        pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
        srcPtr1[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}