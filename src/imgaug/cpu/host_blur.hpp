#include <algorithm>

template <typename T>
RppStatus host_blur(T* srcPtr, RppiSize srcSize, T* dstPtr)
{
 float kernel_3x3[9] = {1,2,1,2,4,2,1,2,1};
    //printf("\n\nKernel:\n");
    for (int i = 0; i < 9; i++)
    {
        kernel_3x3[i] *= 0.0625;
        //printf("%f\t", kernel_3x3[i]);
        //if (((i + 1) % 3) == 0)
        //{
        //    printf("\n");
        //}
    }
    RppiSize sizeMod;
    sizeMod.width = srcSize.width + 2;
    sizeMod.height = srcSize.height + 2;

    Rpp8u *pSrcMod = (Rpp8u *)malloc(sizeMod.width * sizeMod.height * sizeof(Rpp8u));

    int srcLoc = 0, srcModLoc = 0, dstLoc = 0;
    for (int i = 0; i < sizeMod.width; i++)
    {
        pSrcMod[srcModLoc] = 0;
        srcModLoc += 1;
    }
    //srcModLoc = sizeMod.width;
    for (int i = 0; i < srcSize.height; i++)
    {
        pSrcMod[srcModLoc] = 0;
        srcModLoc += 1;
        for (int j = 0; j < srcSize.width; j++)
        {
            pSrcMod[srcModLoc] = srcPtr[srcLoc];
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

    //printf("\n\nPaddded image:\n");
    //display(pSrcMod, sizeMod);

    dstLoc = 0;
    srcModLoc = 0;
    int convLocs[9] = {0}, count = 0;
    float pixel = 0.0;

    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
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
