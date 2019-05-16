#include <algorithm>
#include <math.h>

template <typename T>
RppStatus host_contrast(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                        Rpp32u new_min, Rpp32u new_max, unsigned int channel)
{
    for(int c = 0; c < channel; c++)
    {
        Rpp32f Min = (Rpp32f) *std::min_element(srcPtr + (c * size.width * size.height), srcPtr + ((c + 1) * size.width * size.height));
        Rpp32f Max = (Rpp32f) *std::max_element(srcPtr + (c * size.width * size.height), srcPtr + ((c + 1) * size.width * size.height));

        for (int i = 0; i < (size.width * size.height); i++)
        {
            Rpp32f pixel = (Rpp32f) srcPtr[i + (c * size.width * size.height)];
            pixel = ((pixel - Min) * ((new_max - new_min) / (Max - Min))) + new_min;
            pixel = std::min(pixel, new_max);
            pixel = std::max(pixel, new_min);
            dstPtr[i + (c * size.width * size.height)] = (Rpp8u) pixel;
        }
    }
    
    return RPP_SUCCESS;

}