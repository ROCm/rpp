#ifndef RPP_CPU_COMMON_H
#define RPP_CPU_COMMON_H

#include <math.h>
#include <algorithm>
#include <cstring>
#include <rppdefs.h>

#define PI 3.14159265
#define RAD(deg)                (deg * PI / 180)
#define RPPABS(a)               ((a < 0) ? (-a) : (a))
#define RPPMIN2(a,b)            ((a < b) ? a : b)
#define RPPMIN3(a,b,c)          ((a < b) && (a < c) ?  a : ((b < c) ? b : c))
#define RPPMAX2(a,b)            ((a > b) ? a : b)
#define RPPMAX3(a,b,c)          ((a > b) && (a > c) ?  a : ((b > c) ? b : c))
#define RPPINRANGE(a, x, y)     ((a >= x) && (a <= y) ? 1 : 0)
#define RPPFLOOR(a)             ((int) a)
#define RPPCEIL(a)              ((int) (a + 1.0))
#define RPPISEVEN(a)            ((a % 2 == 0) ? 1 : 0)
#define RPPPIXELCHECK(pixel)    (pixel < (Rpp32f) 0) ? ((Rpp32f) 0) : ((pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255))

#if ENABLE_SIMD_INTRINSICS

#define M256I(m256i_register) (*((_m256i_union*)&m256i_register))
typedef union {
    char               m256i_i8[32];
    short              m256i_i16[16];
    int                m256i_i32[8];
    long long          m256i_i64[4];
    __m128i            m256i_i128[2];
}_m256i_union;

#endif

// Generate Functions

inline RppStatus generate_gaussian_kernel_host(Rpp32f stdDev, Rpp32f* kernel, Rpp32u kernelSize)
{
    Rpp32f s, sum = 0.0, multiplier;
    int bound = ((kernelSize - 1) / 2);
    Rpp32u c = 0;
    s = 1 / (2 * stdDev * stdDev);
    multiplier = (1 / M_PI) * (s);
    for (int i = -bound; i <= bound; i++)
    {
        for (int j = -bound; j <= bound; j++)
        {
            kernel[c] = multiplier * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        kernel[i] /= sum;
    }

    return RPP_SUCCESS;
}

inline RppStatus generate_gaussian_kernel_asymmetric_host(Rpp32f stdDev, Rpp32f* kernel, Rpp32u kernelSizeX, Rpp32u kernelSizeY)
{
    Rpp32f s, sum = 0.0, multiplier;
    if (kernelSizeX % 2 == 0)
    {
        return RPP_ERROR;
    }
    if (kernelSizeY % 2 == 0)
    {
        return RPP_ERROR;
    }
    int boundX = ((kernelSizeX - 1) / 2);
    int boundY = ((kernelSizeY - 1) / 2);
    Rpp32u c = 0;
    s = 1 / (2 * stdDev * stdDev);
    multiplier = (1 / M_PI) * (s);
    for (int i = -boundY; i <= boundY; i++)
    {
        for (int j = -boundX; j <= boundX; j++)
        {
            kernel[c] = multiplier * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSizeX * kernelSizeY); i++)
    {
        kernel[i] /= sum;
    }

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus generate_bilateral_kernel_host(Rpp32f multiplierI, Rpp32f multiplierS, Rpp32f multiplier, Rpp32f* kernel, Rpp32u kernelSize, int bound, 
                                         T* srcPtrWindow, RppiSize srcSizeMod, Rpp32u remainingElementsInRow, Rpp32u incrementToWindowCenter, 
                                         RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f sum = 0.0;
    Rpp32f* kernelTemp;
    kernelTemp = kernel;
    
    T *srcPtrWindowTemp, *srcPtrWindowCenter;
    srcPtrWindowTemp = srcPtrWindow;
    srcPtrWindowCenter = srcPtrWindow + incrementToWindowCenter;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int i = -bound; i <= bound; i++)
        {
            for (int j = -bound; j <= bound; j++)
            {
                T pixel = *srcPtrWindowCenter - *srcPtrWindowTemp;
                pixel = RPPABS(pixel);
                pixel = pixel * pixel;
                *kernelTemp = multiplier * exp((multiplierS * (i*i + j*j)) + (multiplierI * pixel));
                sum = sum + *kernelTemp;
                kernelTemp++;
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = -bound; i <= bound; i++)
        {
            for (int j = -bound; j <= bound; j++)
            {
                T pixel = *srcPtrWindowCenter - *srcPtrWindowTemp;
                pixel = RPPABS(pixel);
                pixel = pixel * pixel;
                *kernelTemp = multiplier * exp((multiplierS * (i*i + j*j)) + (multiplierI * pixel));
                sum = sum + *kernelTemp;
                kernelTemp++;
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }

    kernelTemp = kernel;
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        *kernelTemp = *kernelTemp / sum;
        kernelTemp++;
    }
    
    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus generate_evenly_padded_image_host(T* srcPtr, RppiSize srcSize, T* srcPtrMod, RppiSize srcSizeMod, 
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    if (RPPISEVEN(srcSize.height) != RPPISEVEN(srcSizeMod.height) 
        || RPPISEVEN(srcSize.width) != RPPISEVEN(srcSizeMod.width)
        || srcSizeMod.height < srcSize.height
        || srcSizeMod.width < srcSize.width)
    {
        return RPP_ERROR;
    }
    T *srcPtrTemp, *srcPtrModTemp;
    srcPtrTemp = srcPtr;
    srcPtrModTemp = srcPtrMod;
    int bound = (srcSizeMod.height - srcSize.height) / 2;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            memset (srcPtrModTemp,(T) 0,bound * srcSizeMod.width * sizeof(T));
            srcPtrModTemp += (bound * srcSizeMod.width);
            for (int i = 0; i < srcSize.height; i++)
            {
                memset (srcPtrModTemp,(T) 0,bound * sizeof(T));
                srcPtrModTemp += bound;
                
                memcpy(srcPtrModTemp, srcPtrTemp, srcSize.width * sizeof(T));
                srcPtrModTemp += srcSize.width;
                srcPtrTemp += srcSize.width;
                
                memset (srcPtrModTemp,(T) 0,bound * sizeof(T));
                srcPtrModTemp += bound;
            }
            memset (srcPtrModTemp,(T) 0,bound * srcSizeMod.width * sizeof(T));
            srcPtrModTemp += (bound * srcSizeMod.width);
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u numOfPixelsVtBorder = bound * channel;
        Rpp32u numOfPixelsHrBorder = numOfPixelsVtBorder * srcSizeMod.width;

        memset (srcPtrModTemp,(T) 0,numOfPixelsHrBorder * sizeof(T));
        srcPtrModTemp += (numOfPixelsHrBorder);

        for (int i = 0; i < srcSize.height; i++)
        {
            memset (srcPtrModTemp,(T) 0,numOfPixelsVtBorder * sizeof(T));
            srcPtrModTemp += (numOfPixelsVtBorder);

            memcpy(srcPtrModTemp, srcPtrTemp, elementsInRow * sizeof(T));
            srcPtrModTemp += elementsInRow;
            srcPtrTemp += elementsInRow;
            
            memset (srcPtrModTemp,(T) 0,numOfPixelsVtBorder * sizeof(T));
            srcPtrModTemp += (numOfPixelsVtBorder);
        }

        memset (srcPtrModTemp,(T) 0,numOfPixelsHrBorder * sizeof(T));
        srcPtrModTemp += (numOfPixelsHrBorder);
    }

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus generate_corner_padded_image_host(T* srcPtr, RppiSize srcSize, T* srcPtrMod, RppiSize srcSizeMod, Rpp32u padType, 
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *srcPtrModTemp;
    srcPtrTemp = srcPtr;
    srcPtrModTemp = srcPtrMod;
    Rpp32u boundY = srcSizeMod.height - srcSize.height;
    Rpp32u boundX = srcSizeMod.width - srcSize.width;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            if (padType == 1 || padType == 2)
            {
                memset (srcPtrModTemp,(T) 0,boundY * srcSizeMod.width * sizeof(T));
                srcPtrModTemp += (boundY * srcSizeMod.width);
            }
            
            if (padType == 1 || padType == 3)
            {
                for (int i = 0; i < srcSize.height; i++)
                {
                    memset (srcPtrModTemp,(T) 0,boundX * sizeof(T));
                    srcPtrModTemp += boundX;

                    memcpy(srcPtrModTemp, srcPtrTemp, srcSize.width * sizeof(T));
                    srcPtrModTemp += srcSize.width;
                    srcPtrTemp += srcSize.width;
                }
            }
               
            if (padType == 2 || padType == 4)
            {
                for (int i = 0; i < srcSize.height; i++)
                {
                    memcpy(srcPtrModTemp, srcPtrTemp, srcSize.width * sizeof(T));
                    srcPtrModTemp += srcSize.width;
                    srcPtrTemp += srcSize.width;
                    
                    memset (srcPtrModTemp,(T) 0,boundX * sizeof(T));
                    srcPtrModTemp += boundX;
                }
            }
            
            if (padType == 3 || padType == 4)
            {
                memset (srcPtrModTemp,(T) 0,boundY * srcSizeMod.width * sizeof(T));
                srcPtrModTemp += (boundY * srcSizeMod.width);
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u numOfPixelsVtBorder = boundX * channel;
        Rpp32u numOfPixelsHrBorder = boundY * channel * srcSizeMod.width;

        if (padType == 1 || padType == 2)
        {
            memset (srcPtrModTemp,(T) 0,numOfPixelsHrBorder * sizeof(T));
            srcPtrModTemp += (numOfPixelsHrBorder);
        }

        if (padType == 1 || padType == 3)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                memset (srcPtrModTemp,(T) 0,numOfPixelsVtBorder * sizeof(T));
                srcPtrModTemp += (numOfPixelsVtBorder);

                memcpy(srcPtrModTemp, srcPtrTemp, elementsInRow * sizeof(T));
                srcPtrModTemp += elementsInRow;
                srcPtrTemp += elementsInRow;
            }
        }
        
        if (padType == 2 || padType == 4)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                memcpy(srcPtrModTemp, srcPtrTemp, elementsInRow * sizeof(T));
                srcPtrModTemp += elementsInRow;
                srcPtrTemp += elementsInRow;

                memset (srcPtrModTemp,(T) 0,numOfPixelsVtBorder * sizeof(T));
                srcPtrModTemp += (numOfPixelsVtBorder);
            }
        }

        if (padType == 3 || padType == 4)
        {
            memset (srcPtrModTemp,(T) 0,numOfPixelsHrBorder * sizeof(T));
            srcPtrModTemp += (numOfPixelsHrBorder);
        }
    }

    return RPP_SUCCESS;
}

inline RppStatus generate_box_kernel_host(Rpp32f* kernel, Rpp32u kernelSize)
{
    Rpp32f* kernelTemp;
    kernelTemp = kernel;
    Rpp32f kernelValue = 1.0 / (Rpp32f) (kernelSize * kernelSize);
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        *kernelTemp = kernelValue;
        kernelTemp++;
    }

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus generate_crop_host(T* srcPtr, RppiSize srcSize, T* srcPtrSubImage, RppiSize srcSizeSubImage, T* dstPtr, 
                             RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrSubImageTemp, *dstPtrTemp;
    srcPtrSubImageTemp = srcPtrSubImage;
    dstPtrTemp = dstPtr;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = (srcSize.width - srcSizeSubImage.width);
        for (int c = 0; c < channel; c++)
        {
            srcPtrSubImageTemp = srcPtrSubImage + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSizeSubImage.height; i++)
            {
                for (int j = 0; j < srcSizeSubImage.width; j++)
                {
                    *dstPtrTemp = *srcPtrSubImageTemp;
                    srcPtrSubImageTemp++;
                    dstPtrTemp++;
                }
                srcPtrSubImageTemp += remainingElementsInRow;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = channel * (srcSize.width - srcSizeSubImage.width);
        for (int i = 0; i < srcSizeSubImage.height; i++)
        {
            for (int j = 0; j < srcSizeSubImage.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *dstPtrTemp = *srcPtrSubImageTemp;
                    srcPtrSubImageTemp++;
                    dstPtrTemp++;
                }
            }
            srcPtrSubImageTemp += remainingElementsInRow;
        }
    }

    return RPP_SUCCESS;
}

inline RppStatus generate_sobel_kernel_host(Rpp32f* kernel, Rpp32u type)
{
    Rpp32f* kernelTemp;
    kernelTemp = kernel;

    if (type == 1)
    {
        Rpp32f kernelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        Rpp32f* kernelXTemp;
        kernelXTemp = kernelX;

        for (int i = 0; i < 9; i++)
        {
            *kernelTemp = *kernelXTemp;
            kernelTemp++;
            kernelXTemp++;
        }
    }
    else if (type == 2)
    {
        Rpp32f kernelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
        Rpp32f* kernelYTemp;
        kernelYTemp = kernelY;

        for (int i = 0; i < 9; i++)
        {
            *kernelTemp = *kernelYTemp;
            kernelTemp++;
            kernelYTemp++;
        }
    }
    else
    {
        return RPP_ERROR;
    }

    return RPP_SUCCESS;
}

// Kernels for functions

template<typename T, typename U>
inline RppStatus convolution_kernel_host(T* srcPtrWindow, U* dstPtrPixel, RppiSize srcSize, 
                                       Rpp32f* kernel, RppiSize kernelSize, Rpp32u remainingElementsInRow, U maxVal, U minVal, 
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f pixel = 0.0;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    
    Rpp32f* kernelPtrTemp;
    kernelPtrTemp = kernel;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize.height; m++)
        {
            for (int n = 0; n < kernelSize.width; n++)
            {
                pixel += ((*kernelPtrTemp) * (Rpp32f)(*srcPtrWindowTemp));
                kernelPtrTemp++;
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize.height; m++)
        {
            for (int n = 0; n < kernelSize.width; n++)
            {
                pixel += ((*kernelPtrTemp) * (Rpp32f)(*srcPtrWindowTemp));
                kernelPtrTemp++;
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    (pixel < (Rpp32f) minVal) ? pixel = (Rpp32f) minVal : ((pixel < (Rpp32f) maxVal) ? pixel : pixel = (Rpp32f) maxVal);
    *dstPtrPixel = (U) round(pixel);

    return RPP_SUCCESS;
}


template <typename T>
inline RppStatus resize_kernel_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    if (dstSize.height < 0 || dstSize.width < 0)
    {
        return RPP_ERROR;
    }

    Rpp32f hRatio = (((Rpp32f) (dstSize.height - 1)) / ((Rpp32f) (srcSize.height - 1)));
    Rpp32f wRatio = (((Rpp32f) (dstSize.width - 1)) / ((Rpp32f) (srcSize.width - 1)));
    Rpp32f srcLocationRow, srcLocationColumn, pixel;
    Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {   
                srcLocationRow = ((Rpp32f) i) / hRatio;
                srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                if (srcLocationRowFloor > (srcSize.height - 2))
                {
                    srcLocationRowFloor = srcSize.height - 2;
                }

                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                srcPtrBottomRow  = srcPtrTopRow + srcSize.width;
                
                for (int j = 0; j < dstSize.width; j++)
                {
                    srcLocationColumn = ((Rpp32f) j) / wRatio;
                    srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                    if (srcLocationColumnFloor > (srcSize.width - 2))
                    {
                        srcLocationColumnFloor = srcSize.width - 2;
                    }
                    pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));
                    
                    *dstPtrTemp = (T) round(pixel);
                    dstPtrTemp ++;
                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRow = ((Rpp32f) i) / hRatio;
            srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
            Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

            if (srcLocationRowFloor > (srcSize.height - 2))
            {
                srcLocationRowFloor = srcSize.height - 2;
            }

            srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
            srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

            for (int j = 0; j < dstSize.width; j++)
            {   
                srcLocationColumn = ((Rpp32f) j) / wRatio;
                srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                if (srcLocationColumnFloor > (srcSize.width - 2))
                {
                    srcLocationColumnFloor = srcSize.width - 2;
                }

                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;
                
                for (int c = 0; c < channel; c++)
                {
                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));
                    
                    *dstPtrTemp = (T) round(pixel);
                    dstPtrTemp ++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

#if ENABLE_SIMD_INTRINSICS

#define FP_BITS     16
#define FP_MUL      (1<<FP_BITS)

template<>
inline RppStatus resize_kernel_host(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiSize dstSize,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
   // return RPP_SUCCESS;
    if (dstSize.height < 0 || dstSize.width < 0 )
    {
        return RPP_ERROR;
    }
    // call ref host implementation
    //if (channel > 3 )
    //    return resize_kernel_host(srcPtr, dstPtr, dstSize, chnFormat, channel);

    Rpp8u *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    Rpp32f yscale = ((Rpp32f) (dstSize.height)) / ((Rpp32f) (srcSize.height));    // yscale
    Rpp32f xscale = ((Rpp32f) (dstSize.width)) / ((Rpp32f) (srcSize.width));      //xscale
    int alignW = (dstSize.width + 15) & ~15;
    // generate maps for fixed point computations
    unsigned int *Xmap = new unsigned int[alignW*2];
    unsigned short *Xf = (unsigned short *)(Xmap + alignW);
    unsigned short *Xf1 = Xf + alignW;
    int xpos = (int)(FP_MUL * (xscale*0.5 - 0.5));
    int xinc = (int)(FP_MUL * xscale);
    int yinc = (int)(FP_MUL * yscale);      // to convert to fixed point
    unsigned int aligned_width = dstSize.width;

    // generate xmap
    for (unsigned int x = 0; x < dstSize.width; x++, xpos += xinc)
    {
        int xf;
        int xmap = (xpos >> FP_BITS);
        if (xmap >= (int)(srcSize.width - 8)){
            aligned_width = x;
        }
        if (xmap >= (int)(srcSize.width - 1)){
            Xmap[x] = (chnFormat == RPPI_CHN_PLANAR)? (srcSize.width - 1):(srcSize.width - 1)*3;
        }
        else
            Xmap[x] = (xmap<0)? 0: (chnFormat == RPPI_CHN_PLANAR)? xmap: xmap*3;
        xf = ((xpos & 0xffff) + 0x80) >> 8;
        Xf[x] = xf;
        Xf1[x] = (0x100 - xf);
    }
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;       
        for (int c = 0; c < channel; c++)
        {
            int dstride = dstSize.width;
            int sstride = srcSize.width;
            Rpp8u *pSrcBorder = srcPtrTemp + (srcSize.height*sstride);    // points to the last pixel

            int ypos = (int)(FP_MUL * (yscale*0.5 - 0.5));
            for (int y = 0; y < (int)dstSize.height; y++, ypos += yinc)
            {
                int ym, fy, fy1;
                Rpp8u *pSrc1, *pSrc2;
                Rpp8u *pdst = dstPtrTemp + y*dstride;

                ym = (ypos >> FP_BITS);
                fy = ((ypos & 0xffff) + 0x80) >> 8;
                fy1 = (0x100 - fy);
                if (ym > (int)(srcSize.height - 1)){
                    pSrc1 = pSrc2 = srcPtrTemp + (srcSize.height - 1)*sstride;
                }
                else
                {
                    pSrc1 = (ym<0)? srcPtrTemp : (srcPtr + ym*sstride);
                    pSrc2 = pSrc1 + sstride;
                }
                for (int x=0; x < dstSize.width; x++) {
                    int result;
                    const unsigned char *p0 = pSrc1 + Xmap[x];
                    const unsigned char *p01 = p0 + channel;
                    const unsigned char *p1 = pSrc2 + Xmap[x];
                    const unsigned char *p11 = p1 + channel;
                    if (p0 > pSrcBorder) p0 = pSrcBorder;
                    if (p1 > pSrcBorder) p1 = pSrcBorder;
                    if (p01 > pSrcBorder) p01 = pSrcBorder;
                    if (p11 > pSrcBorder) p11 = pSrcBorder;
                    result = ((Xf1[x] * fy1*p0[0]) + (Xf[x] * fy1*p01[0]) + (Xf1[x] * fy*p1[0]) + (Xf[x] * fy*p11[0]) + 0x8000) >> 16;
                    *pdst++ = (Rpp8u) std::max(0, std::min(result, 255));
                    result = ((Xf1[x] * fy1*p0[1]) + (Xf[x] * fy1*p01[1]) + (Xf1[x] * fy*p1[1]) + (Xf[x] * fy*p11[1]) + 0x8000) >> 16;
                    *pdst++ = (Rpp8u)std::max(0, std::min(result, 255));
                    result = ((Xf1[x] * fy1*p0[2]) + (Xf[x] * fy1*p01[2]) + (Xf1[x] * fy*p1[2]) + (Xf[x] * fy*p11[2]) + 0x8000) >> 16;
                    *pdst++ = (Rpp8u)std::max(0, std::min(result, 255));
                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
            dstPtrTemp += dstSize.width * dstSize.height; 
        }
    }
    
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        aligned_width &= ~3;
        int dstride = dstSize.width * channel;
        int sstride = srcSize.width * channel;
        Rpp8u *pSrcBorder = srcPtr + (srcSize.height*sstride) - channel;    // points to the last pixel
#if __AVX2__
        const __m256i mm_zeros = _mm256_setzero_si256();
        const __m256i mm_round = _mm256_set1_epi32((int)0x80);
        const __m256i pmask1 = _mm256_set_epi32(0, 1, 2, 4, 5, 6, 3, 7);
        const __m256i pmask2 = _mm256_set_epi32(2, 3, 4, 5, 6, 7, 0, 1);
#endif
        int ypos = (int)(FP_MUL * (yscale*0.5 - 0.5));
//#pragma omp parallel for
       for (int y = 0; y < (int)dstSize.height; y++, ypos += yinc)
        {
            int ym, fy, fy1;
            Rpp8u *pSrc1, *pSrc2;
            Rpp8u *pdst = dstPtrTemp + y*dstride;

            ym = (ypos >> FP_BITS);
            fy = ((ypos & 0xffff) + 0x80) >> 8;
            fy1 = (0x100 - fy);
            if (ym > (int)(srcSize.height - 1)){
                pSrc1 = pSrc2 = srcPtrTemp + (srcSize.height - 1)*sstride;
            }
            else
            {
                pSrc1 = (ym<0)? srcPtrTemp : (srcPtrTemp + ym*sstride);
                pSrc2 = pSrc1 + sstride;
            }
            unsigned int x = 0;
#if __AVX2__
            
            __m256i w_y = _mm256_set_epi32(fy1, fy, fy1, fy, fy1, fy, fy1, fy);
            __m256i p01, p23, ps01, ps23, px0, px1, ps2, ps3;
            for (; x < aligned_width; x += 4)
            {
                // load 2 pixels each
                M256I(p01).m256i_i128[0] = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x]]);
                M256I(p23).m256i_i128[0] = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x+1]]);
                M256I(ps01).m256i_i128[0] = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x]]);
                M256I(ps23).m256i_i128[0] = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x+1]]);

                M256I(p01).m256i_i128[1] = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x+2]]);
                M256I(p23).m256i_i128[1] = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x+3]]);
                M256I(ps01).m256i_i128[1] = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x+2]]);
                M256I(ps23).m256i_i128[1] = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x+3]]);

                // unpcklo for p01 and ps01
                p01 = _mm256_unpacklo_epi8(p01, ps01);
                p23 = _mm256_unpacklo_epi8(p23, ps23);
                p01 = _mm256_unpacklo_epi16(p01, _mm256_srli_si256(p01, 6));     //R0R1R2R3 G0G1G2G3 B0B1B2B3 XXXX for 1st and 3rd pixel
                p23 = _mm256_unpacklo_epi16(p23, _mm256_srli_si256(p23, 6));      //R0R1R2R3 G0G1G2G3 B0B1B2B3 XXXX for 2nd and 4th pixel

                // load xf and 1-xf
                ps01 = _mm256_setr_epi32(Xf1[x], Xf1[x], Xf[x], Xf[x], Xf1[x+2], Xf1[x+2], Xf[x+2], Xf[x+2]);            // xfxfxf1xf1
                ps23 = _mm256_setr_epi32(Xf1[x+1], Xf1[x+1], Xf[x+1], Xf[x+1], Xf1[x+3], Xf1[x+3], Xf[x+3], Xf[x+3]);
                ps01 = _mm256_mullo_epi32(ps01, w_y);                      // W0W1W2W3 for 1st and 3rd
                ps23 = _mm256_mullo_epi32(ps23, w_y);                      // W0W1W2W3 for 2nd and 4th
                ps01 = _mm256_srli_epi32(ps01, 8);                 // convert to 16bit
                ps23 = _mm256_srli_epi32(ps23, 8);                 // convert to 16bit
                ps01 = _mm256_packus_epi32(ps01, ps01);                 // pack to 16bit (w0w1w2w3(0), (w0w1w2w3(2), w0w1w2w3(0), (w0w1w2w3(2)))
                ps23 = _mm256_packus_epi32(ps23, ps23);                 // pack to 16bit (w0w1w2w3(1), (w0w1w2w3(3), w0w1w2w3(1), (w0w1w2w3(3))
                ps01 = _mm256_permute4x64_epi64(ps01, 0xe0);            // (w0w1w2w3(0), (w0w1w2w3(0), w0w1w2w3(0), w0w1w2w3(2)
                ps23 = _mm256_permute4x64_epi64(ps23, 0xe0);            // (w0w1w2w3(1), (w0w1w2w3(1), w0w1w2w3(1), w0w1w2w3(3)
                ps2  = _mm256_permute4x64_epi64(ps01, 0xff);            // (w0w1w2w3(2), (w0w1w2w3(2), w0w1w2w3(2), w0w1w2w3(2)
                ps3  = _mm256_permute4x64_epi64(ps23, 0xff);            // (w0w1w2w3(3), (w0w1w2w3(3), w0w1w2w3(3), w0w1w2w3(3)

                // get pixels in place for interpolation
                px0 = _mm256_unpacklo_epi8(p01, mm_zeros);        // R0R1R2R3(0), G0G1G2G3(0), B0B1B2B3(0), xxxxx 
                p01 = _mm256_unpackhi_epi8(p01, mm_zeros);        // R0R1R2R3(2), G0G1G2G3(2), B0B1B2B3(2), xxxxx 
                px1 = _mm256_unpacklo_epi8(p23, mm_zeros);        // R0R1R2R3(1), G0G1G2G3(1), B0B1B2B3(1), xxxxx 
                p23 = _mm256_unpackhi_epi8(p23, mm_zeros);        // R0R1R2R3(3), G0G1G2G3(3), B0B1B2B3(3), xxxxx 

                px0 = _mm256_madd_epi16(px0, ps01);                  // pix0: (W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3), (W0*B0+W1*B1), (W2*B2+W3*B3), xx, xx 
                px1 = _mm256_madd_epi16(px1, ps23);                  // pix1: (W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3), (W0*B0+W1*B1), (W2*B2+W3*B3), xx, xx 
                p01 = _mm256_madd_epi16(p01, ps2);                  // pix2: (W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3), (W0*B0+W1*B1), (W2*B2+W3*B3), xx, xx 
                p23 = _mm256_madd_epi16(p23, ps3);                  // pix3: (W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3), (W0*B0+W1*B1), (W2*B2+W3*B3), xx, xx 
                px0 = _mm256_hadd_epi32(px0, px1);      // R0,G0, B0, xx, R1, G1, B1, xx (32bit)
                p01 = _mm256_hadd_epi32(p01, p23);      // R2,G2, B2, xx, R3, G3, B3, xx (32bit)
                px0 = _mm256_permutevar8x32_epi32(px0, pmask1); // R0,G0, B0, R1, G1, B1, xx, xx 
                p01 = _mm256_permutevar8x32_epi32(p01, pmask1); // R2, G2, B2, R3, G3, B3, xx, xx, 
                px0 = _mm256_add_epi32(px0, mm_round);
                p01 = _mm256_add_epi32(p01, mm_round);
                px0 = _mm256_srli_epi32(px0, 8);      // /256
                p01 = _mm256_srli_epi32(px0, 8);      // /256
                px0 = _mm256_packus_epi32(px0, p01); //R0G0B0R1G1B1xx R2G2B2R3B3G3xx
                px0 = _mm256_permutevar8x32_epi32(px0, pmask1); //R0G0B0R1G1B1R2G2B2R3B3G3xxxx
                px0 = _mm256_packus_epi16(px0, mm_zeros); //R0G0B0R1G1B1R2G2B2R3G3B3xxxx ....
                _mm_storeu_si128((__m128i *)pdst, M256I(px0).m256i_i128[0]);      // write 12 bytes
                pdst += 12;
            }
#endif            
            for (; x < dstSize.width; x++) {
                int result;
                const unsigned char *p0 = pSrc1 + Xmap[x];
                const unsigned char *p01 = p0 + channel;
                const unsigned char *p1 = pSrc2 + Xmap[x];
                const unsigned char *p11 = p1 + channel;
                if (p0 > pSrcBorder) p0 = pSrcBorder;
                if (p1 > pSrcBorder) p1 = pSrcBorder;
                if (p01 > pSrcBorder) p01 = pSrcBorder;
                if (p11 > pSrcBorder) p11 = pSrcBorder;
                result = ((Xf1[x] * fy1*p0[0]) + (Xf[x] * fy1*p01[0]) + (Xf1[x] * fy*p1[0]) + (Xf[x] * fy*p11[0]) + 0x8000) >> 16;
                *pdst++ = (Rpp8u) std::max(0, std::min(result, 255));
                result = ((Xf1[x] * fy1*p0[1]) + (Xf[x] * fy1*p01[1]) + (Xf1[x] * fy*p1[1]) + (Xf[x] * fy*p11[1]) + 0x8000) >> 16;
                *pdst++ = (Rpp8u)std::max(0, std::min(result, 255));
                result = ((Xf1[x] * fy1*p0[2]) + (Xf[x] * fy1*p01[2]) + (Xf1[x] * fy*p1[2]) + (Xf[x] * fy*p11[2]) + 0x8000) >> 16;
                *pdst++ = (Rpp8u)std::max(0, std::min(result, 255));
            }
        }
    }
    if (Xmap) delete[] Xmap;

    return RPP_SUCCESS;
}
#endif

template <typename T>
inline RppStatus resize_crop_kernel_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    RppiSize srcSizeSubImage;
    T *srcPtrSubImage;

    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    T *srcPtrResize = (T*) calloc(channel * srcSizeSubImage.height * srcSizeSubImage.width, sizeof(T));

    generate_crop_host(srcPtr, srcSize, srcPtrSubImage, srcSizeSubImage, srcPtrResize, chnFormat, channel);

    resize_kernel_host(srcPtrResize, srcSizeSubImage, dstPtr, dstSize, chnFormat, channel);

    free(srcPtrResize);
    return RPP_SUCCESS;
    
}

template<typename T>
inline RppStatus erode_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize, 
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow, 
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T pixel;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    pixel = *srcPtrWindowTemp;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp < pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp < pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    *dstPtrPixel = pixel;

    return RPP_SUCCESS;
}

template<typename T>
inline RppStatus dilate_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize, 
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow, 
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T pixel;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    pixel = *srcPtrWindowTemp;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp > pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp > pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    *dstPtrPixel = pixel;

    return RPP_SUCCESS;
}

template<typename T>
inline RppStatus median_filter_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize, 
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow, 
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T *kernel = (T*)calloc(kernelSize * kernelSize, sizeof(T));
    T *kernelTemp;
    kernelTemp = kernel;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *kernelTemp = *srcPtrWindowTemp;
                srcPtrWindowTemp++;
                kernelTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *kernelTemp = *srcPtrWindowTemp;
                srcPtrWindowTemp += channel;
                kernelTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }

    std::sort(kernel, kernel + (kernelSize * kernelSize));

    *dstPtrPixel = *(kernel + (((kernelSize * kernelSize) - 1) / 2));
    free(kernel);
    return RPP_SUCCESS;
}

template<typename T>
RppStatus local_binary_pattern_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize, 
                                       Rpp32u remainingElementsInRow, T* centerPixelPtr, 
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T pixel = (T) 0;
    T *srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 0);
        }
        srcPtrWindowTemp++;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 1);
        }
        srcPtrWindowTemp++;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 2);
        }
        srcPtrWindowTemp++;
        srcPtrWindowTemp += remainingElementsInRow;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 7);
        }
        srcPtrWindowTemp += 2;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 3);
        }
        srcPtrWindowTemp++;
        srcPtrWindowTemp += remainingElementsInRow;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 6);
        }
        srcPtrWindowTemp++;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 5);
        }
        srcPtrWindowTemp++;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 4);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 0);
        }
        srcPtrWindowTemp += channel;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 1);
        }
        srcPtrWindowTemp += channel;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 2);
        }
        srcPtrWindowTemp += channel;
        srcPtrWindowTemp += remainingElementsInRow;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 7);
        }
        srcPtrWindowTemp += (2 * channel);

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 3);
        }
        srcPtrWindowTemp += channel;
        srcPtrWindowTemp += remainingElementsInRow;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 6);
        }
        srcPtrWindowTemp += channel;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 5);
        }
        srcPtrWindowTemp += channel;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 4);
        }
    }

    *dstPtrPixel = pixel;

    return RPP_SUCCESS;
}

template<typename T>
inline RppStatus non_max_suppression_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize, 
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow, T windowCenter, 
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T pixel;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    pixel = *srcPtrWindowTemp;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp > pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp > pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    if (windowCenter >= pixel)
    {
        *dstPtrPixel = windowCenter;
    }
    else
    {
        *dstPtrPixel = (T) 0;
    }

    return RPP_SUCCESS;
}

// Convolution Functions

template<typename T, typename U>
inline RppStatus convolve_image_host(T* srcPtrMod, RppiSize srcSizeMod, U* dstPtr, RppiSize srcSize, 
                        Rpp32f* kernel, RppiSize kernelSize, 
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrWindow;
    U *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;

    U maxVal = (U)(std::numeric_limits<U>::max());
    U minVal = (U)(std::numeric_limits<U>::min());

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSizeMod.width - kernelSize.width;

        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                                 kernel, kernelSize, remainingElementsInRow, maxVal, minVal, 
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += (kernelSize.width - 1);
            }
            srcPtrWindow += ((kernelSize.height - 1) * srcSizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = (srcSizeMod.width - kernelSize.width) * channel;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {   
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                                 kernel, kernelSize, remainingElementsInRow, maxVal, minVal, 
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += ((kernelSize.width - 1) * channel);
        }
    }
    
    return RPP_SUCCESS;
}

template<typename T>
inline RppStatus convolve_subimage_host(T* srcPtrMod, RppiSize srcSizeMod, T* dstPtr, RppiSize srcSizeSubImage, RppiSize srcSize, 
                        Rpp32f* kernel, RppiSize kernelSize, 
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    int widthDiffPlanar = srcSize.width - srcSizeSubImage.width;
    int widthDiffPacked = (srcSize.width - srcSizeSubImage.width) * channel;

    T *srcPtrWindow, *dstPtrTemp;

    T maxVal = (T)(std::numeric_limits<T>::max());
    T minVal = (T)(std::numeric_limits<T>::min());
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSize.width - kernelSize.width;

        for (int c = 0; c < channel; c++)
        {
            srcPtrWindow = srcPtrMod + (c * srcSize.height * srcSize.width);
            dstPtrTemp = dstPtr + (c * srcSize.height * srcSize.width);
#pragma omp parallel for
            for (int i = 0; i < srcSizeSubImage.height; i++)
            {
#pragma omp parallel for
                for (int j = 0; j < srcSizeSubImage.width; j++)
                {
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                                 kernel, kernelSize, remainingElementsInRow, maxVal, minVal, 
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += widthDiffPlanar;
                dstPtrTemp += widthDiffPlanar;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = (srcSize.width - kernelSize.width) * channel;

        srcPtrWindow = srcPtrMod;
        dstPtrTemp = dstPtr;
#pragma omp parallel for
        for (int i = 0; i < srcSizeSubImage.height; i++)
        {
#pragma omp parallel for
            for (int j = 0; j < srcSizeSubImage.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {   
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                                 kernel, kernelSize, remainingElementsInRow, maxVal, minVal, 
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += widthDiffPacked;
            dstPtrTemp += widthDiffPacked;
        }
    }
    
    return RPP_SUCCESS;
}

// Compute Functions

template<typename T>
inline RppStatus compute_subimage_location_host(T* ptr, T** ptrSubImage, 
                                         RppiSize size, RppiSize *sizeSubImage, 
                                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if ((RPPINRANGE(x1, 0, size.width - 1) == 0) 
        || (RPPINRANGE(x2, 0, size.width - 1) == 0) 
        || (RPPINRANGE(y1, 0, size.height - 1) == 0) 
        || (RPPINRANGE(y2, 0, size.height - 1) == 0))
    {
        return RPP_ERROR;
    }
    
    int yDiff = (int) y2 - (int) y1;
    int xDiff = (int) x2 - (int) x1;

    sizeSubImage->height = (Rpp32u) RPPABS(yDiff) + 1;
    sizeSubImage->width = (Rpp32u) RPPABS(xDiff) + 1;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        *ptrSubImage = ptr + (RPPMIN2(y1, y2) * size.width) + RPPMIN2(x1, x2);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        *ptrSubImage = ptr + (RPPMIN2(y1, y2) * size.width * channel) + (RPPMIN2(x1, x2) * channel);
    }

    return RPP_SUCCESS;
}

template<typename T>
inline RppStatus compute_transpose_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize, 
                                 RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
#pragma omp parallel for
            for (int i = 0; i < dstSize.height; i++)
            {
#pragma omp parallel for simd
                for (int j = 0; j < dstSize.width; j++)
                {
                    *dstPtrTemp = *(srcPtrTemp + (j * srcSize.width) + i);
                    dstPtrTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
#pragma omp parallel for
        for (int i = 0; i < dstSize.height; i++)
        {
#pragma omp parallel for simd
            for (int j = 0; j < dstSize.width; j++)
            {
                srcPtrTemp = srcPtr + (channel * ((j * srcSize.width) + i));
                for (int c = 0; c < channel; c++)
                {
                    *dstPtrTemp = *(srcPtrTemp + c);
                    dstPtrTemp++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
inline RppStatus compute_subtract_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                        Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32s pixel;
#pragma omp parallel for simd
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) - ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

template <typename T, typename U>
inline RppStatus compute_multiply_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;
#pragma omp parallel for simd
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32f) (*srcPtr1Temp)) * ((Rpp32f) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp = (T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
inline RppStatus compute_rgb_to_hsv_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
    U *dstPtrTempH, *dstPtrTempS, *dstPtrTempV;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u imageDim = srcSize.height * srcSize.width;
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + (imageDim);
        srcPtrTempB = srcPtr + (2 * imageDim);
        dstPtrTempH = dstPtr;
        dstPtrTempS = dstPtr + (imageDim);
        dstPtrTempV = dstPtr + (2 * imageDim);
#pragma omp parallel for simd
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                *dstPtrTempH = 0;
            }
            else if (cmax == rf)
            {
                *dstPtrTempH = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                *dstPtrTempH = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                *dstPtrTempH = round(60 * (((rf - gf) / delta) + 4));
            }
            
            while (*dstPtrTempH > 360)
            {
                *dstPtrTempH = *dstPtrTempH - 360;
            }
            while (*dstPtrTempH < 0)
            {
                *dstPtrTempH = 360 + *dstPtrTempH;
            }

            if (cmax == 0)
            {
                *dstPtrTempS = 0;
            }
            else
            {
                *dstPtrTempS = delta / cmax;
            }

            *dstPtrTempV = cmax;
            
            srcPtrTempR++;
            srcPtrTempG++;
            srcPtrTempB++;
            dstPtrTempH++;
            dstPtrTempS++;
            dstPtrTempV++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + 1;
        srcPtrTempB = srcPtr + 2;
        dstPtrTempH = dstPtr;
        dstPtrTempS = dstPtr + 1;
        dstPtrTempV = dstPtr + 2;
#pragma omp parallel for simd
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                *dstPtrTempH = 0;
            }
            else if (cmax == rf)
            {
                *dstPtrTempH = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                *dstPtrTempH = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                *dstPtrTempH = round(60 * (((rf - gf) / delta) + 4));
            }
            
            while (*dstPtrTempH > 360)
            {
                *dstPtrTempH = *dstPtrTempH - 360;
            }
            while (*dstPtrTempH < 0)
            {
                *dstPtrTempH = 360 + *dstPtrTempH;
            }

            if (cmax == 0)
            {
                *dstPtrTempS = 0;
            }
            else
            {
                *dstPtrTempS = delta / cmax;
            }

            *dstPtrTempV = cmax;

            srcPtrTempR += 3;
            srcPtrTempG += 3;
            srcPtrTempB += 3;
            dstPtrTempH += 3;
            dstPtrTempS += 3;
            dstPtrTempV += 3;
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T, typename U>
inline RppStatus compute_hsv_to_rgb_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTempH, *srcPtrTempS, *srcPtrTempV;
    U *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u imageDim = srcSize.height * srcSize.width;
        srcPtrTempH = srcPtr;
        srcPtrTempS = srcPtr + (imageDim);
        srcPtrTempV = srcPtr + (2 * imageDim);
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + (imageDim);
        dstPtrTempB = dstPtr + (2 * imageDim);
#pragma omp parallel for simd
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f c, x, m, rf, gf, bf;
            c = *srcPtrTempV * *srcPtrTempS;
            x = c * (1 - abs((fmod((*srcPtrTempH / 60), 2)) - 1));
            m = *srcPtrTempV - c;
            
            if ((0 <= *srcPtrTempH) && (*srcPtrTempH < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= *srcPtrTempH) && (*srcPtrTempH < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= *srcPtrTempH) && (*srcPtrTempH < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= *srcPtrTempH) && (*srcPtrTempH < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= *srcPtrTempH) && (*srcPtrTempH < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= *srcPtrTempH) && (*srcPtrTempH < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempH++;
            srcPtrTempS++;
            srcPtrTempV++;
            dstPtrTempR++;
            dstPtrTempG++;
            dstPtrTempB++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrTempH = srcPtr;
        srcPtrTempS = srcPtr + 1;
        srcPtrTempV = srcPtr + 2;
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + 1;
        dstPtrTempB = dstPtr + 2;
#pragma omp parallel for simd
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f c, x, m, rf, gf, bf;
            c = *srcPtrTempV * *srcPtrTempS;
            x = c * (1 - abs((fmod((*srcPtrTempH / 60), 2)) - 1));
            m = *srcPtrTempV - c;
            
            if ((0 <= *srcPtrTempH) && (*srcPtrTempH < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= *srcPtrTempH) && (*srcPtrTempH < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= *srcPtrTempH) && (*srcPtrTempH < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= *srcPtrTempH) && (*srcPtrTempH < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= *srcPtrTempH) && (*srcPtrTempH < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= *srcPtrTempH) && (*srcPtrTempH < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempH += 3;
            srcPtrTempS += 3;
            srcPtrTempV += 3;
            dstPtrTempR += 3;
            dstPtrTempG += 3;
            dstPtrTempB += 3;
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
inline RppStatus compute_rgb_to_hsl_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
    U *dstPtrTempH, *dstPtrTempS, *dstPtrTempL;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u imageDim = srcSize.height * srcSize.width;
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + (imageDim);
        srcPtrTempB = srcPtr + (2 * imageDim);
        dstPtrTempH = dstPtr;
        dstPtrTempS = dstPtr + (imageDim);
        dstPtrTempL = dstPtr + (2 * imageDim);
#pragma omp parallel for simd
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f rf, gf, bf, cmax, cmin, delta, divisor;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            divisor = cmax + cmin - 1;
            delta = cmax - cmin;

            if (delta == 0)
            {
                *dstPtrTempH = 0;
            }
            else if (cmax == rf)
            {
                *dstPtrTempH = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                *dstPtrTempH = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                *dstPtrTempH = round(60 * (((rf - gf) / delta) + 4));
            }
            
            while (*dstPtrTempH > 360)
            {
                *dstPtrTempH = *dstPtrTempH - 360;
            }
            while (*dstPtrTempH < 0)
            {
                *dstPtrTempH = 360 + *dstPtrTempH;
            }

            if (delta == 0)
            {
                *dstPtrTempS = 0;
            }
            else
            {
                *dstPtrTempS = delta / (1 - RPPABS(divisor));
            }

            *dstPtrTempL = (cmax + cmin) / 2;

            srcPtrTempR++;
            srcPtrTempG++;
            srcPtrTempB++;
            dstPtrTempH++;
            dstPtrTempS++;
            dstPtrTempL++;

        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + 1;
        srcPtrTempB = srcPtr + 2;
        dstPtrTempH = dstPtr;
        dstPtrTempS = dstPtr + 1;
        dstPtrTempL = dstPtr + 2;
#pragma omp parallel for simd
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f rf, gf, bf, cmax, cmin, delta, divisor;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            divisor = cmax + cmin - 1;
            delta = cmax - cmin;

            if (delta == 0)
            {
                *dstPtrTempH = 0;
            }
            else if (cmax == rf)
            {
                *dstPtrTempH = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                *dstPtrTempH = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                *dstPtrTempH = round(60 * (((rf - gf) / delta) + 4));
            }
            
            while (*dstPtrTempH > 360)
            {
                *dstPtrTempH = *dstPtrTempH - 360;
            }
            while (*dstPtrTempH < 0)
            {
                *dstPtrTempH = 360 + *dstPtrTempH;
            }

            if (delta == 0)
            {
                *dstPtrTempS = 0;
            }
            else
            {
                *dstPtrTempS = delta / (1 - RPPABS(divisor));
            }

            *dstPtrTempL = (cmax + cmin) / 2;

            srcPtrTempR += 3;
            srcPtrTempG += 3;
            srcPtrTempB += 3;
            dstPtrTempH += 3;
            dstPtrTempS += 3;
            dstPtrTempL += 3;

        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
inline RppStatus compute_hsl_to_rgb_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTempH, *srcPtrTempS, *srcPtrTempL;
    U *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u imageDim = srcSize.height * srcSize.width;
        srcPtrTempH = srcPtr;
        srcPtrTempS = srcPtr + (imageDim);
        srcPtrTempL = srcPtr + (2 * imageDim);
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + (imageDim);
        dstPtrTempB = dstPtr + (2 * imageDim);
#pragma omp parallel for simd
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f c, x, m, rf, gf, bf;
            c = (2 * *srcPtrTempL) - 1;
            c = (1 - RPPABS(c)) * *srcPtrTempS;
            x = c * (1 - abs((fmod((*srcPtrTempH / 60), 2)) - 1));
            m = *srcPtrTempL - c / 2;
            
            if ((0 <= *srcPtrTempH) && (*srcPtrTempH < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= *srcPtrTempH) && (*srcPtrTempH < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= *srcPtrTempH) && (*srcPtrTempH < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= *srcPtrTempH) && (*srcPtrTempH < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= *srcPtrTempH) && (*srcPtrTempH < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= *srcPtrTempH) && (*srcPtrTempH < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempH++;
            srcPtrTempS++;
            srcPtrTempL++;
            dstPtrTempR++;
            dstPtrTempG++;
            dstPtrTempB++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrTempH = srcPtr;
        srcPtrTempS = srcPtr + 1;
        srcPtrTempL = srcPtr + 2;
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + 1;
        dstPtrTempB = dstPtr + 2;
#pragma omp parallel for simd
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f c, x, m, rf, gf, bf;
            c = (2 * *srcPtrTempL) - 1;
            c = (1 - RPPABS(c)) * *srcPtrTempS;
            x = c * (1 - abs((fmod((*srcPtrTempH / 60), 2)) - 1));
            m = *srcPtrTempL - c / 2;
            
            if ((0 <= *srcPtrTempH) && (*srcPtrTempH < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= *srcPtrTempH) && (*srcPtrTempH < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= *srcPtrTempH) && (*srcPtrTempH < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= *srcPtrTempH) && (*srcPtrTempH < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= *srcPtrTempH) && (*srcPtrTempH < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= *srcPtrTempH) && (*srcPtrTempH < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempH += 3;
            srcPtrTempS += 3;
            srcPtrTempL += 3;
            dstPtrTempR += 3;
            dstPtrTempG += 3;
            dstPtrTempB += 3;
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
inline RppStatus compute_magnitude_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, U* dstPtr,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr1Temp, *srcPtr2Temp;
    U *dstPtrTemp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;
    Rpp32s srcPtr1Value, srcPtr2Value;
#pragma omp parallel for simd
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        srcPtr1Value = (Rpp32s) *srcPtr1Temp;
        srcPtr2Value = (Rpp32s) *srcPtr2Temp;
        pixel = sqrt((srcPtr1Value * srcPtr1Value) + (srcPtr2Value * srcPtr2Value));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(U) round(pixel);
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

template <typename T, typename U>
inline RppStatus compute_threshold_host(T* srcPtr, RppiSize srcSize, U* dstPtr, 
                                 U min, U max, Rpp32u type, 
                                 RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp;
    U *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (type == 1)
    {
#pragma omp parallel for simd
        for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
        {
            if (*srcPtrTemp < min)
            {
                *dstPtrTemp = (U) 0;
            }
            else if (*srcPtrTemp <= max)
            {
                *dstPtrTemp = (U) 255;
            }
            else
            {
                *dstPtrTemp = (U) 0;
            }
            
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }
    else if (type == 2)
    {
#pragma omp parallel for simd
        for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
        {
            if (RPPABS(*srcPtrTemp) < min)
            {
                *dstPtrTemp = (U) 0;
            }
            else if (RPPABS(*srcPtrTemp) <= max)
            {
                *dstPtrTemp = (U) 255;
            }
            else
            {
                *dstPtrTemp = (U) 0;
            }
            
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }

    return RPP_SUCCESS;

}

template <typename T>
inline RppStatus compute_data_object_copy_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    memcpy(dstPtr, srcPtr, srcSize.height * srcSize.width * channel * sizeof(T));
    
    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_downsampled_image_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize, 
                                         RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp8u checkEven;
    checkEven = (Rpp8u) RPPISEVEN(srcSize.width);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
#pragma omp parallel for
            for (int i = 0; i < dstSize.height; i++)
            {
#pragma omp parallel for simd
                for (int j = 0; j < dstSize.width; j++)
                {
                    *dstPtrTemp = *srcPtrTemp;
                    srcPtrTemp += 2;
                    dstPtrTemp++;
                }
                if (checkEven == 0)
                {
                    srcPtrTemp--;
                }
                srcPtrTemp += srcSize.width;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = srcSize.width * channel;
#pragma omp parallel for
        for (int i = 0; i < dstSize.height; i++)
        {
#pragma omp parallel for simd
            for (int j = 0; j < dstSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *dstPtrTemp = *srcPtrTemp;
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtrTemp += channel;
            }
            if (checkEven == 0)
            {
                srcPtrTemp -= channel;
            }
            srcPtrTemp += elementsInRow;
        }
    }

    return RPP_SUCCESS;
}

inline Rpp32u fogGenerator(Rpp32u srcPtr, Rpp32f fogValue, int colour, int check)
{
    unsigned int fog=0;
    int range;
    if(check >= (240) && fogValue!=0);
    else if(check>=(170))
        range = 1;
    else if(check<=(85))
        range = 2; 
    else 
    range = 3;
    switch(range)
    {
        case 1:
            if(colour==1)
            {
                fog = srcPtr * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
            }
            else if(colour==2)
            {
                fog = srcPtr * (1.5 + fogValue) + (7*fogValue);
            }
            else
            {
                fog = srcPtr * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
            }
            break;
        case 2:
            if(colour==1)
            {
                fog = srcPtr * (1.5 + pow(fogValue,2)) - (fogValue*4) + (130*fogValue);
            }
            else if(colour==2)
            {
                fog = srcPtr * (1.5 + pow(fogValue,2)) + (130*fogValue);
            }
            else
            {
                fog = srcPtr * (1.5 + pow(fogValue,2)) + (fogValue*4) + 130*fogValue;
            }
            break;
        case 3:
            if(colour==1)
            {
                fog = srcPtr * (1.5 + pow(fogValue,1.5)) - (fogValue*4) + 20 + (100*fogValue);
            }
            else if(colour==2)
            {
                fog = srcPtr * (1.5 + pow(fogValue,1.5)) + 20 + (100*fogValue);
            }
            else
            {
                fog = srcPtr * (1.5 + pow(fogValue,1.5)) + (fogValue*4) + (100*fogValue);
            }
            break;
    }
    fog = RPPPIXELCHECK(fog);
    return fog;
}


#endif //RPP_CPU_COMMON_H
