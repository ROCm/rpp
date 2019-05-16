#include <algorithm>
#include <math.h>
using namespace std;

template <typename T>
RppStatus host_flip(T* srcPtr, RppiSize srcSize, T* dstPtr, 
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