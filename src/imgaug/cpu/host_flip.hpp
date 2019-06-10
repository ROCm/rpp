#include <algorithm>
#include <math.h>
using namespace std;

template <typename T>
RppStatus flip_pln_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                        RppiAxis flipAxis, unsigned channel)
{
    if (flipAxis == RPPI_HORIZONTAL_AXIS)
    {
        int srcLoc = 0, dstLoc = 0;
        for (int i = (srcSize.height - 1); i >= 0; i--)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                srcLoc = (i * srcSize.width) + j;
                dstPtr[dstLoc] = srcPtr[srcLoc];
                if( channel == 3) {
                    dstPtr[dstLoc + (srcSize.width * srcSize.height)] = srcPtr[srcLoc + (srcSize.width * srcSize.height)];
                    dstPtr[dstLoc + (2 * srcSize.width * srcSize.height)] = srcPtr[srcLoc + (2 * srcSize.width * srcSize.height)];
                }
                dstLoc += 1;
            }
        }
    }
    else if (flipAxis == RPPI_VERTICAL_AXIS)
    {
        int srcLoc = 0, dstLoc = 0;
        for (int i = (srcSize.width - 1); i >= 0; i--)
        {
            dstLoc = srcSize.width - 1 - i;
            for (int j = 0; j < srcSize.height; j++)
            {
                srcLoc = (j * srcSize.width) + i;
                dstPtr[dstLoc] = srcPtr[srcLoc];
                if( channel == 3) {
                    dstPtr[dstLoc + (srcSize.width * srcSize.height)] = srcPtr[srcLoc + (srcSize.width * srcSize.height)];
                    dstPtr[dstLoc + (2 * srcSize.width * srcSize.height)] = srcPtr[srcLoc + (2 * srcSize.width * srcSize.height)];
                }
                dstLoc += srcSize.width;
            }
        }
    }
    else if (flipAxis == RPPI_BOTH_AXIS)
    {
        Rpp8u *pInter = (Rpp8u *)malloc(channel * srcSize.width * srcSize.height * sizeof(Rpp8u));
        int srcLoc = 0, interLoc = 0;
        for (int i = (srcSize.height - 1); i >= 0; i--)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                srcLoc = (i * srcSize.width) + j;
                pInter[interLoc] = srcPtr[srcLoc];
                if( channel == 3) {
                    pInter[interLoc + (srcSize.width * srcSize.height)] = srcPtr[srcLoc + (srcSize.width * srcSize.height)];
                    pInter[interLoc + (2 * srcSize.width * srcSize.height)] = srcPtr[srcLoc + (2 * srcSize.width * srcSize.height)];
                }
                interLoc += 1;
            }
        }
        int dstLoc = 0;
        interLoc = 0;
        for (int i = (srcSize.width - 1); i >= 0; i--)
        {
            dstLoc = srcSize.width - 1 - i;
            for (int j = 0; j < srcSize.height; j++)
            {
                interLoc = (j * srcSize.width) + i;
                dstPtr[dstLoc] = pInter[interLoc];
                if( channel == 3) {
                    dstPtr[dstLoc + (srcSize.width * srcSize.height)] = pInter[interLoc + (srcSize.width * srcSize.height)];
                    dstPtr[dstLoc + (2 * srcSize.width * srcSize.height)] = pInter[interLoc + (2 * srcSize.width * srcSize.height)];
                }
                dstLoc += srcSize.width;
            }
        }
    }

    return RPP_SUCCESS;

}

template <typename T>
RppStatus flip_pkd_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiAxis flipAxis, unsigned channel)
{
    if (flipAxis == RPPI_HORIZONTAL_AXIS)
    {
        int srcLoc = 0, dstLoc = 0;
        for (int i = (srcSize.height - 1); i >= 0; i--)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                srcLoc = (i * 3 * srcSize.width) + (3 * j);
                dstPtr[dstLoc] = srcPtr[srcLoc];
                dstPtr[dstLoc + 1] = srcPtr[srcLoc + 1];
                dstPtr[dstLoc + 2] = srcPtr[srcLoc + 2];
                srcLoc += 3;
                dstLoc += 3;
            }
        }
    }
    else if (flipAxis == RPPI_VERTICAL_AXIS)
    {
        int srcLoc = 0, dstLoc = 0;
        for (int i = (srcSize.width - 1); i >= 0; i--)
        {
            dstLoc = 3 * (srcSize.width - 1 - i);
            for (int j = 0; j < srcSize.height; j++)
            {
                srcLoc = (j * 3 * srcSize.width) + (i * 3);
                dstPtr[dstLoc] = srcPtr[srcLoc];
                dstPtr[dstLoc + 1] = srcPtr[srcLoc + 1];
                dstPtr[dstLoc + 2] = srcPtr[srcLoc + 2];
                dstLoc += (srcSize.width * 3);
            }
        }
    }
    else if (flipAxis == RPPI_BOTH_AXIS)
    {
        Rpp8u *pInter = (Rpp8u *)malloc(channel * srcSize.width * srcSize.height * sizeof(Rpp8u));
        int srcLoc = 0, interLoc = 0;

        for (int i = (srcSize.height - 1); i >= 0; i--)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                srcLoc = (i * 3 * srcSize.width) + (3 * j);
                pInter[interLoc] = srcPtr[srcLoc];
                pInter[interLoc + 1] = srcPtr[srcLoc + 1];
                pInter[interLoc + 2] = srcPtr[srcLoc + 2];
                srcLoc += 3;
                interLoc += 3;
            }
        }

        int dstLoc = 0;
        interLoc = 0;


        for (int i = (srcSize.width - 1); i >= 0; i--)
        {
            dstLoc = 3 * (srcSize.width - 1 - i);
            for (int j = 0; j < srcSize.height; j++)
            {
                interLoc = (j * 3 * srcSize.width) + (i * 3);
                dstPtr[dstLoc] = pInter[interLoc];
                dstPtr[dstLoc + 1] = pInter[interLoc + 1];
                dstPtr[dstLoc + 2] = pInter[interLoc + 2];
                dstLoc += (srcSize.width * 3);
            }
        }
    }

    return RPP_SUCCESS;

}