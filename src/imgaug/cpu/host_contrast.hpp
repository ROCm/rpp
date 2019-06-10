#include <algorithm>
#include <math.h>
using namespace std;

template <typename T>
RppStatus contrast_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                        Rpp32u new_min, Rpp32u new_max, unsigned int channel)
{
    for(int c = 0; c < channel; c++)
    {
        Rpp32f Min = (Rpp32f) *std::min_element(srcPtr + (c * srcSize.width * srcSize.height), srcPtr + ((c + 1) * srcSize.width * srcSize.height));
        Rpp32f Max = (Rpp32f) *std::max_element(srcPtr + (c * srcSize.width * srcSize.height), srcPtr + ((c + 1) * srcSize.width * srcSize.height));

        for (int i = 0; i < (srcSize.width * srcSize.height); i++)
        {
            Rpp32f pixel = (Rpp32f) srcPtr[i + (c * srcSize.width * srcSize.height)];
            pixel = ((pixel - Min) * ((new_max - new_min) / (Max - Min))) + new_min;
            pixel = std::min(pixel, (Rpp32f)new_max);
            pixel = std::max(pixel, (Rpp32f)new_min);
            dstPtr[i + (c * srcSize.width * srcSize.height)] = (Rpp8u) pixel;
        }
    }
    
    return RPP_SUCCESS;

}