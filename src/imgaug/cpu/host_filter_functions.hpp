#include <cpu/rpp_cpu_common.hpp>

template <typename T>
RppStatus bilateral_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32s diameter, Rpp64f sigmaI, Rpp64f sigmaS,
                                RppiChnFormat chnFormat, unsigned int channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for(int i = 2; i < srcSize.height - 2; i++) {
                for(int j = 2; j < srcSize.width - 2; j++) {
                    int x = i, y = j;
                    double iFiltered = 0;
                    double wP = 0;
                    int neighbor_x = 0;
                    int neighbor_y = 0;
                    int half = diameter / 2;

                    for(int m = 0; m < diameter; m++) {
                        for(int n = 0; n < diameter; n++) {
                            neighbor_x = x - (half - m);
                            neighbor_y = y - (half - n);
                            double gi = RPPGAUSSIAN(srcPtr[(c * srcSize.height * srcSize.width) + (neighbor_x * srcSize.width) + neighbor_y] - srcPtr[(c * srcSize.height * srcSize.width) + (x * srcSize.width) + y], sigmaI);
                            double gs = RPPGAUSSIAN(RPPDISTANCE(x, y, neighbor_x, neighbor_y), sigmaS);
                            double w = gi * gs;
                            iFiltered = iFiltered + srcPtr[(c * srcSize.height * srcSize.width) + (neighbor_x * srcSize.width) + neighbor_y] * w;
                            wP = wP + w;
                        }
                    }
                    iFiltered = iFiltered / wP;
                    dstPtr[(c * srcSize.height * srcSize.width) + (x * srcSize.width) + y] = (Rpp8u) iFiltered;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int c = 0; c < channel; c++)
        {
            for(int i = 2; i < srcSize.height - 2; i++) {
                for(int j = 2; j < srcSize.width - 2; j++) {
                    int x = i, y = j;
                    double iFiltered = 0;
                    double wP = 0;
                    int neighbor_x = 0;
                    int neighbor_y = 0;
                    int half = diameter / 2;

                    for(int m = 0; m < diameter; m++) {
                        for(int n = 0; n < diameter; n++) {
                            neighbor_x = x - (half - m);
                            neighbor_y = y - (half - n);
                            double gi = RPPGAUSSIAN(srcPtr[c + (channel * neighbor_x * srcSize.width) + (channel * neighbor_y)] - srcPtr[c + (channel * x * srcSize.width) + (channel * y)], sigmaI);
                            double gs = RPPGAUSSIAN(RPPDISTANCE(x, y, neighbor_x, neighbor_y), sigmaS);
                            double w = gi * gs;
                            iFiltered = iFiltered + srcPtr[c + (channel * neighbor_x * srcSize.width) + (channel * neighbor_y)] * w;
                            wP = wP + w;
                        }
                    }
                    iFiltered = iFiltered / wP;
                    dstPtr[c + (channel * x * srcSize.width) + (channel * y)] = (Rpp8u) iFiltered;
                }
            }
        }
    }
    return RPP_SUCCESS;

}