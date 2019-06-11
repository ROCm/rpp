#include <algorithm>
#include <math.h>
using namespace std;

/**************** Flip ***************/
// host planar flip implementation

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

// host packed flip implementation

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

/**************** Rotate ***************/
// host Rotate implementation

template <typename T>
RppStatus host_rotate(T *srcPtr, RppiSize srcSize, T *dstPtr, RppiSize dstSize, Rpp32f angleRad, int channel)
{
    // Step 1 - Rotate
    int m, n;
    Rpp8u *pSrc = (Rpp8u *)malloc(channel * dstSize.width * dstSize.height * sizeof(Rpp8u));
    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            m = (int) ((i * cos(angleRad)) - (j * sin(angleRad)) + (srcSize.width * sin(angleRad)));
            n = (int) ((i * sin(angleRad)) + (j * cos(angleRad)));
            pSrc[(m * dstSize.width) + n] = srcPtr[(i * srcSize.width) + j];
        }
    }
    RppiSize size;
    size.width = dstSize.width;
    size.height = dstSize.height;

    // Step 2 - Blur
    float kernel_3x3[9] = {1,2,1,2,4,2,1,2,1};
    for (int i = 0; i < 9; i++)
    {
        kernel_3x3[i] *= 0.0625;
    }
    RppiSize sizeMod;
    sizeMod.width = size.width + 2;
    sizeMod.height = size.height + 2;

    Rpp8u *pSrcMod = (Rpp8u *)malloc(sizeMod.width * sizeMod.height * sizeof(Rpp8u));

    int srcLoc = 0, srcModLoc = 0, dstLoc = 0;
    for (int i = 0; i < sizeMod.width; i++)
    {
        pSrcMod[srcModLoc] = 0;
        srcModLoc += 1;
    }
    for (int i = 0; i < size.height; i++)
    {
        pSrcMod[srcModLoc] = 0;
        srcModLoc += 1;
        for (int j = 0; j < size.width; j++)
        {
            pSrcMod[srcModLoc] = pSrc[srcLoc];
            srcModLoc += 1;
            srcLoc += 1;
        }
        pSrcMod[srcModLoc] = 0;
        srcModLoc += 1;
    }
    for (int i = 0; i < sizeMod.width; i++)
    {
        pSrcMod[srcModLoc] = 0;
        srcModLoc += 1;
    }
    
    dstLoc = 0;
    srcModLoc = 0;
    int convLocs[9] = {0}, count = 0;
    float pixel = 0.0;

    for (int i = 0; i < size.height; i++)
    {
        for (int j = 0; j < size.width; j++)
        {
            count = 0;
            pixel = 0.0;
            for (int m = 0; m < 3; m++)
            {
                for (int n = 0; n < 3; n++, count++)
                {
                    convLocs[count] = srcModLoc + (m * sizeMod.width) + n;
                }
            }
            for (int k = 0; k < 9; k++)
            {
                pixel += (kernel_3x3[k] * (float)pSrcMod[convLocs[k]]);
            }
            pixel = std::min(pixel, (Rpp32f) 255);
            pixel = std::max(pixel, (Rpp32f) 0);
            dstPtr[dstLoc] = (Rpp8u) round(pixel);
            dstLoc += 1;
            srcModLoc += 1;
        }
        srcModLoc += 2;
    }

    return RPP_SUCCESS;
}
