#include <algorithm>
#include <math.h>

template <typename T>
RppStatus host_flip(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                        RppiAxis flipAxis, unsigned channel)
{
    if (flipAxis == RPPI_HORIZONTAL_AXIS)
    {
        int srcLoc = 0, dstLoc = 0;
        for (int i = (size.height - 1); i >= 0; i--)
        {
            for (int j = 0; j < size.width; j++)
            {
                srcLoc = (i * size.width) + j;
                pDst[dstLoc] = pSrc[srcLoc];
                if( channel == 3) {
                    pDst[dstLoc + (size.width * size.height)] = pSrc[srcLoc + (size.width * size.height)];
                    pDst[dstLoc + (2 * size.width * size.height)] = pSrc[srcLoc + (2 * size.width * size.height)];
                }
                dstLoc += 1;
            }
        }
    }
    else if (flipAxis == RPPI_VERTICAL_AXIS)
    {
        int srcLoc = 0, dstLoc = 0;
        for (int i = (size.width - 1); i >= 0; i--)
        {
            dstLoc = size.width - 1 - i;
            for (int j = 0; j < size.height; j++)
            {
                srcLoc = (j * size.width) + i;
                pDst[dstLoc] = pSrc[srcLoc];
                if( channel == 3) {
                    pDst[dstLoc + (size.width * size.height)] = pSrc[srcLoc + (size.width * size.height)];
                    pDst[dstLoc + (2 * size.width * size.height)] = pSrc[srcLoc + (2 * size.width * size.height)];
                }
                dstLoc += size.width;
            }
        }
    }
    else if (flipAxis == RPPI_BOTH_AXIS)
    {
        Rpp8u *pInter = (Rpp8u *)malloc(channel * size.width * size.height * sizeof(Rpp8u));
        int srcLoc = 0, interLoc = 0;
        for (int i = (size.height - 1); i >= 0; i--)
        {
            for (int j = 0; j < size.width; j++)
            {
                srcLoc = (i * size.width) + j;
                pInter[interLoc] = pSrc[srcLoc];
                if( channel == 3) {
                    pInter[interLoc + (size.width * size.height)] = pSrc[srcLoc + (size.width * size.height)];
                    pInter[interLoc + (2 * size.width * size.height)] = pSrc[srcLoc + (2 * size.width * size.height)];
                }
                interLoc += 1;
            }
        }
        int dstLoc = 0;
        interLoc = 0;
        for (int i = (size.width - 1); i >= 0; i--)
        {
            dstLoc = size.width - 1 - i;
            for (int j = 0; j < size.height; j++)
            {
                interLoc = (j * size.width) + i;
                pDst[dstLoc] = pInter[interLoc];
                if( channel == 3) {
                    pDst[dstLoc + (size.width * size.height)] = pInter[interLoc + (size.width * size.height)];
                    pDst[dstLoc + (2 * size.width * size.height)] = pInter[interLoc + (2 * size.width * size.height)];
                }
                dstLoc += size.width;
            }
        }
    }

    return RPP_SUCCESS;

}