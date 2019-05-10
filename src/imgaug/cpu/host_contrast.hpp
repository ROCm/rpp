#include <algorithm>
#include <math.h>

template <typename T>
RppStatus host_contrast(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                        Rpp32u new_min = 225, Rpp32u new_max = 0)
{
    Rpp32f Min = (Rpp32f) *std::min_element(srcPtr,srcPtr + (srcSize.width * srcSize.height));
    Rpp32f Max = (Rpp32f) *std::max_element(srcPtr,srcPtr + (srcSize.width * srcSize.height));
    
    for (int i = 0; i < (srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = (Rpp32f) srcPtr[i];
        pixel = ((pixel - Min) * ((new_max - new_min) / (Max - Min))) + new_min;
        pixel = std::min(pixel, (Rpp32f) new_max);
        pixel = std::max(pixel, (Rpp32f) new_min);
        dstPtr[i] = (Rpp8u) pixel;
    }
    
    return RPP_SUCCESS;
}
