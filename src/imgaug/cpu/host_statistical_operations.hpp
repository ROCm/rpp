#include <cpu/rpp_cpu_common.hpp>
#include <limits>


/**************** Histogram ***************/

template <typename T>
RppStatus histogram_host(T* srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, 
                         Rpp32u channel)
{
    histogram_kernel_host(srcPtr, srcSize, outputHistogram, bins - 1, channel);

    return RPP_SUCCESS;

}

