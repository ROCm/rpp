#ifndef RPP_CPU_COMMON_H
#define RPP_CPU_COMMON_H

#include <math.h>
#include <algorithm>
#include <typeinfo>
#include <cstring>
#include <rppdefs.h>
#include <omp.h>
#include <half.hpp>
using halfhpp = half_float::half;
typedef halfhpp Rpp16f;
#include "rpp_cpu_simd.hpp"

#define PI                              3.14159265
#define PI_OVER_180                     0.0174532925
#define ONE_OVER_255                    0.00392157f
#define RAD(deg)                        (deg * PI / 180)
#define RPPABS(a)                       ((a < 0) ? (-a) : (a))
#define RPPMIN2(a,b)                    ((a < b) ? a : b)
#define RPPMIN3(a,b,c)                  ((a < b) && (a < c) ?  a : ((b < c) ? b : c))
#define RPPMAX2(a,b)                    ((a > b) ? a : b)
#define RPPMAX3(a,b,c)                  ((a > b) && (a > c) ?  a : ((b > c) ? b : c))
#define RPPINRANGE(a, x, y)             ((a >= x) && (a <= y) ? 1 : 0)
#define RPPPRANGECHECK(value, a, b)     (value < (Rpp32f) a) ? ((Rpp32f) a) : ((value < (Rpp32f) b) ? value : ((Rpp32f) b))
#define RPPFLOOR(a)                     ((int) a)
#define RPPCEIL(a)                      ((int) (a + 1.0))
#define RPPISEVEN(a)                    ((a % 2 == 0) ? 1 : 0)
#define RPPPIXELCHECK(pixel)            (pixel < (Rpp32f) 0) ? ((Rpp32f) 0) : ((pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255))
#define RPPPIXELCHECKF32(pixel)         (pixel < (Rpp32f) 0) ? ((Rpp32f) 0) : ((pixel < (Rpp32f) 1) ? pixel : ((Rpp32f) 1))
#define RPPPIXELCHECKI8(pixel)          (pixel < (Rpp32f) -128) ? ((Rpp32f) -128) : ((pixel < (Rpp32f) 127) ? pixel : ((Rpp32f) 127))
#define RPPISGREATER(pixel, value)      ((pixel > value) ? 1 : 0)
#define RPPISLESSER(pixel, value)       ((pixel < value) ? 1 : 0)

static uint16_t wyhash16_x;

alignas(64) const Rpp32f sch_mat[16] = {0.701f, -0.299f, -0.300f, 0.0f, -0.587f, 0.413f, -0.588f, 0.0f, -0.114f, -0.114f, 0.886f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
alignas(64) const Rpp32f ssh_mat[16] = {0.168f, -0.328f, 1.250f, 0.0f, 0.330f, 0.035f, -1.050f, 0.0f, -0.497f, 0.292f, -0.203f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

inline uint32_t hash16(uint32_t input, uint32_t key) {
  uint32_t hash = input * key;
  return ((hash >> 16) ^ hash) & 0xFFFF;
}

inline uint16_t wyhash16() {
  wyhash16_x += 0xfc15;
  return hash16(wyhash16_x, 0x2ab);
}

inline uint16_t rand_range16(const uint16_t s) {
    uint16_t x = wyhash16();
    uint32_t m = (uint32_t)x * (uint32_t)s;
    uint16_t l = (uint16_t)m;
    if (l < s) {
        uint16_t t = -s % s;
        while (l < t) {
            x = wyhash16();
            m = (uint32_t)x * (uint32_t)s;
            l = (uint16_t)m;
        }
    }
    return m >> 16;
}

static unsigned int g_seed;

inline void fast_srand( int seed )
{
    g_seed = seed;
}

inline int fastrand()
{
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16)&0x7FFF;
}

inline int power_function(int a, int b)
{
    int product = 1;
    for(int i = 0; i < b; i++)
        product *= product * a;
    return product;
}

template <typename T>
RppStatus subtract_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus add_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus multiply_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus min_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus max_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);

template <typename T>
RppStatus bitwise_AND_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus inclusive_OR_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus exclusive_OR_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);


// Specific Helper Functions

inline Rpp32f gaussian_2d_relative(Rpp32s locI, Rpp32s locJ, Rpp32f std_dev)
{
    Rpp32f relativeGaussian;
    Rpp32f exp1, exp2;
    exp1 = -(locJ * locJ) / (2 * std_dev * std_dev);
    exp2 = -(locI * locI) / (2 * std_dev * std_dev);
    relativeGaussian = exp(exp1 + exp2);

    return relativeGaussian;
}

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
                Rpp32u bufferLength = srcSizeSubImage.width;
                Rpp32u alignedLength = bufferLength & ~15;

                __m128i px0;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrSubImageTemp);
                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrSubImageTemp +=16;
                    dstPtrTemp +=16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = *srcPtrSubImageTemp++;
                }
                srcPtrSubImageTemp += remainingElementsInRow;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = channel * (srcSize.width - srcSizeSubImage.width);
        Rpp32u elementsInRowCrop = channel * srcSizeSubImage.width;
        for (int i = 0; i < srcSizeSubImage.height; i++)
        {
            Rpp32u bufferLength = elementsInRowCrop;
            Rpp32u alignedLength = bufferLength & ~15;

            __m128i px0;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrSubImageTemp);
                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                srcPtrSubImageTemp +=16;
                dstPtrTemp +=16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                *dstPtrTemp++ = *srcPtrSubImageTemp++;
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

template <typename T>
inline RppStatus generate_bressenham_line_host(T *dstPtr, RppiSize dstSize, Rpp32u *endpoints, Rpp32u *rasterCoordinates)
{
    Rpp32u *rasterCoordinatesTemp;
    rasterCoordinatesTemp = rasterCoordinates;

    Rpp32s x0 = *endpoints;
    Rpp32s y0 = *(endpoints + 1);
    Rpp32s x1 = *(endpoints + 2);
    Rpp32s y1 = *(endpoints + 3);

    Rpp32s dx, dy;
    Rpp32s stepX, stepY;

    dx = x1 - x0;
    dy = y1 - y0;

    if (dy < 0)
    {
        dy = -dy;
        stepY = -1;
    }
    else
    {
        stepY = 1;
    }

    if (dx < 0)
    {
        dx = -dx;
        stepX = -1;
    }
    else
    {
        stepX = 1;
    }

    dy <<= 1;
    dx <<= 1;

    if ((0 <= x0) && (x0 < dstSize.width) && (0 <= y0) && (y0 < dstSize.height))
    {
        *(dstPtr + (y0 * dstSize.width) + x0) = (T) 255;
        *rasterCoordinatesTemp = y0;
        rasterCoordinatesTemp++;
        *rasterCoordinatesTemp = x0;
        rasterCoordinatesTemp++;
    }

    if (dx > dy)
    {
        Rpp32s fraction = dy - (dx >> 1);
        while (x0 != x1)
        {
            x0 += stepX;
            if (fraction >= 0)
            {
                y0 += stepY;
                fraction -= dx;
            }
            fraction += dy;
            if ((0 <= x0) && (x0 < dstSize.width) && (0 <= y0) && (y0 < dstSize.height))
            {
                *(dstPtr + (y0 * dstSize.width) + x0) = (T) 255;
                *rasterCoordinatesTemp = y0;
                rasterCoordinatesTemp++;
                *rasterCoordinatesTemp = x0;
                rasterCoordinatesTemp++;
            }
        }
    }
    else
    {
        int fraction = dx - (dy >> 1);
        while (y0 != y1)
        {
            if (fraction >= 0)
            {
                x0 += stepX;
                fraction -= dy;
            }
            y0 += stepY;
            fraction += dx;
            if ((0 <= x0) && (x0 < dstSize.width) && (0 <= y0) && (y0 < dstSize.height))
            {
                *(dstPtr + (y0 * dstSize.width) + x0) = (T) 255;
                *rasterCoordinatesTemp = y0;
                rasterCoordinatesTemp++;
                *rasterCoordinatesTemp = x0;
                rasterCoordinatesTemp++;
            }
        }
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

template<typename T>
inline RppStatus histogram_kernel_host(T* srcPtr, RppiSize srcSize, Rpp32u* histogram,
                                Rpp8u bins,
                                Rpp32u channel)
{
    if (bins == 0)
    {
        *histogram = channel * srcSize.height * srcSize.width;
    }
    else
    {
        Rpp8u rangeInBin = 256 / (bins + 1);
        T *srcPtrTemp;
        srcPtrTemp = srcPtr;
        for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
        {
            *(histogram + (*srcPtrTemp / rangeInBin)) += 1;
            srcPtrTemp++;
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
inline RppStatus accumulate_kernel_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize,
                                        RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr1Temp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;

    Rpp32s pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *srcPtr1Temp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
    }

    return RPP_SUCCESS;

}

template <typename U>
inline RppStatus normalize_kernel_host(U* dstPtrROI, RppiSize dstSize, Rpp32u channel)
{
    U* dstPtrROITemp;
    dstPtrROITemp = dstPtrROI;

    U multiplier = (U) (1.0 / 255.0);

    Rpp32u imageDim = dstSize.height * dstSize.width * channel;

    for (int i = 0; i < imageDim; i++)
    {
        *dstPtrROITemp = *dstPtrROITemp * multiplier;
        dstPtrROITemp++;
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
inline RppStatus resize_kernel_host(T* srcPtr, RppiSize srcSize, U* dstPtr, RppiSize dstSize,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (dstSize.height < 0 || dstSize.width < 0)
        {
            return RPP_ERROR;
        }

        Rpp32f hRatio = (((Rpp32f) (dstSize.height - 1)) / ((Rpp32f) (srcSize.height - 1)));
        Rpp32f wRatio = (((Rpp32f) (dstSize.width - 1)) / ((Rpp32f) (srcSize.width - 1)));
        Rpp32f srcLocationRow, srcLocationColumn, pixel;
        Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
        T *srcPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
        U *dstPtrTemp;
        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;

        if ((typeid(Rpp16f) == typeid(T)) || (typeid(Rpp16f) == typeid(U)))
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

                        *dstPtrTemp = (U) pixel;
                        dstPtrTemp ++;
                    }
                }
                srcPtrTemp += srcSize.height * srcSize.width;
            }
        }
        else
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
#pragma omp simd
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

                        *dstPtrTemp = (U) pixel;
                        dstPtrTemp ++;
                    }
                }
                srcPtrTemp += srcSize.height * srcSize.width;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        if (dstSize.height < 0 || dstSize.width < 0)
        {
            return RPP_ERROR;
        }

        Rpp32f hRatio = (((Rpp32f) (dstSize.height - 1)) / ((Rpp32f) (srcSize.height - 1)));
        Rpp32f wRatio = (((Rpp32f) (dstSize.width - 1)) / ((Rpp32f) (srcSize.width - 1)));
        Rpp32f srcLocationRow, srcLocationColumn, pixel;
        Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
        T *srcPtrTemp;
        U *dstPtrTemp;
        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;

        Rpp32u heightLimit = srcSize.height - 2;
        Rpp32u widthLimit = srcSize.width - 2;

        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRow = ((Rpp32f) i) / hRatio;
            srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
            Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

            if (srcLocationRowFloor > heightLimit)
            {
                srcLocationRowFloor = heightLimit;
            }

            T *srcPtrTopRow, *srcPtrBottomRow;
            srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
            srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

            Rpp32u bufferLength = dstSize.width;
            Rpp32u alignedLength = (bufferLength / 4) * 4;

            Rpp32u srcLocCF[4] = {0};
            Rpp32f param1[4] = {0};
            Rpp32f param2[4] = {0};
            Rpp32f param3[4] = {0};
            Rpp32f param4[4] = {0};

            __m128 pWRatio = _mm_set1_ps(1.0 / wRatio);
            __m128 p0, p2, p4, p5, p6, p7, pColFloor;
            __m128 p1 = _mm_set1_ps(weightedHeight);
            __m128 p3 = _mm_set1_ps(1 - weightedHeight);
            __m128 pOne = _mm_set1_ps(1.0);
            __m128i pxColFloor;

            Rpp64u vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
            {
                p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                p0 = _mm_mul_ps(p0, pWRatio);
                pColFloor = _mm_floor_ps(p0);
                pxColFloor = _mm_cvtps_epi32(pColFloor);
                p0 = _mm_sub_ps(p0, pColFloor);
                p2  = _mm_sub_ps(pOne, p0);

                p4 = _mm_mul_ps(p3, p2);
                p5 = _mm_mul_ps(p3, p0);
                p6 = _mm_mul_ps(p1, p2);
                p7 = _mm_mul_ps(p1, p0);

                _mm_storeu_si128((__m128i*) srcLocCF, pxColFloor);
                _mm_storeu_ps(param1, p4);
                _mm_storeu_ps(param2, p5);
                _mm_storeu_ps(param3, p6);
                _mm_storeu_ps(param4, p7);

                for (int pos = 0; pos < 4; pos++)
                {
                    if (srcLocCF[pos] > widthLimit)
                    {
                        srcLocCF[pos] = widthLimit;
                    }
                    srcLocCF[pos] *= channel;

                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp++ = (U) ((*(srcPtrTopRow + c + srcLocCF[pos])) * param1[pos])
                                            + ((*(srcPtrTopRow + c + srcLocCF[pos] + channel)) * param2[pos])
                                            + ((*(srcPtrBottomRow + c + srcLocCF[pos])) * param3[pos])
                                            + ((*(srcPtrBottomRow + c + srcLocCF[pos] + channel)) * param4[pos]);
                    }
                }
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                srcLocationColumn = ((Rpp32f) vectorLoopCount) / wRatio;
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

                    *dstPtrTemp = (U) pixel;
                    dstPtrTemp ++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

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
inline RppStatus local_binary_pattern_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u remainingElementsInRow, T* centerPixelPtr,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    int pixel = (int) 0;
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

    *dstPtrPixel = (T) RPPPIXELCHECK(pixel);

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

template<typename T>
inline RppStatus canny_non_max_suppression_kernel_host(T* dstPtrPixel, T windowCenter, T *position1Ptr, T *position2Ptr)
{
    if ((windowCenter >= *position1Ptr) && (windowCenter >= *position2Ptr))
    {
        *dstPtrPixel = windowCenter;
    }
    else
    {
        *dstPtrPixel = (T) 0;
    }

    return RPP_SUCCESS;
}

template<typename T>
inline RppStatus canny_hysterisis_edge_tracing_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow, T windowCenter, Rpp32u bound,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;

    for (int m = 0; m < kernelSize; m++)
    {
        for (int n = 0; n < kernelSize; n++)
        {
            if (*srcPtrWindowTemp == (T) 255)
            {
                *dstPtrPixel = (T) 255;
                return RPP_SUCCESS;
            }
            srcPtrWindowTemp++;
        }
        srcPtrWindowTemp += remainingElementsInRow;
    }
    *dstPtrPixel = (T) 0;

    return RPP_SUCCESS;
}

template<typename T, typename U>
inline RppStatus harris_corner_detector_kernel_host(T* srcPtrWindowX, T* srcPtrWindowY, U* dstPtrPixel, RppiSize srcSize,
                                             Rpp32u kernelSize, Rpp32u remainingElementsInRow, Rpp32f kValue, Rpp32f threshold,
                                             RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f pixel;

    T *srcPtrWindowTempX, *srcPtrWindowTempY;
    srcPtrWindowTempX = srcPtrWindowX;
    srcPtrWindowTempY = srcPtrWindowY;

    Rpp32f sumXX = 0, sumYY = 0, sumXY = 0;

    for (int m = 0; m < kernelSize; m++)
    {
        for (int n = 0; n < kernelSize; n++)
        {
            Rpp32f valX = (Rpp32f) *srcPtrWindowTempX;
            Rpp32f valY = (Rpp32f) *srcPtrWindowTempY;
            sumXX += (valX * valX);
            sumYY += (valY * valY);
            sumXY += (valX * valY);

            srcPtrWindowTempX++;
            srcPtrWindowTempY++;
        }
        srcPtrWindowTempX += remainingElementsInRow;
        srcPtrWindowTempY += remainingElementsInRow;
    }
    Rpp32f det = (sumXX * sumYY) - (sumXY * sumXY);
    Rpp32f trace = sumXX + sumYY;
    pixel = (det) - (kValue * trace * trace);

    if (pixel > threshold)
    {
        *dstPtrPixel = (U) pixel;
    }
    else
    {
        *dstPtrPixel = (U) 0;
    }

    return RPP_SUCCESS;
}

template<typename T>
inline RppStatus harris_corner_set_maximum_kernel_host(T* dstPtrWindow, Rpp32u kernelSize, Rpp32u remainingElementsInRow,
                                                  RppiChnFormat chnFormat, Rpp32u channel)
{
    T* dstPtrWindowTemp;
    dstPtrWindowTemp = dstPtrWindow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *dstPtrWindowTemp = (T) 255;
                dstPtrWindowTemp++;
            }
            dstPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *dstPtrWindowTemp = (T) 255;
                dstPtrWindowTemp += channel;
            }
            dstPtrWindowTemp += remainingElementsInRow;
        }
    }

    return RPP_SUCCESS;
}

template<typename T>
inline RppStatus harris_corner_set_minimum_kernel_host(T* dstPtrWindow, Rpp32u kernelSize, Rpp32u remainingElementsInRow,
                                                  RppiChnFormat chnFormat, Rpp32u channel)
{
    T* dstPtrWindowTemp;
    dstPtrWindowTemp = dstPtrWindow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *dstPtrWindowTemp = (T) 0;
                dstPtrWindowTemp++;
            }
            dstPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *dstPtrWindowTemp = (T) 0;
                dstPtrWindowTemp += channel;
            }
            dstPtrWindowTemp += remainingElementsInRow;
        }
    }

    return RPP_SUCCESS;
}

inline RppStatus tensor_index_exchange_kernel_host(Rpp32u *loopCount, Rpp32u *loopCountTransposed, Rpp32u tensorDimension, Rpp32u dimension1, Rpp32u dimension2)
{
    memcpy(loopCountTransposed, loopCount, tensorDimension * sizeof(Rpp32u));

    loopCountTransposed[dimension2] = loopCount[dimension1];
    loopCountTransposed[dimension1] = loopCount[dimension2];

    return RPP_SUCCESS;
}

template<typename T>
inline RppStatus tensor_transpose_iterate_kernel_host(T* srcPtr, T* dstPtr,
                                               Rpp32u tensorDimensionTemp, Rpp32u tensorDimension,
                                               Rpp32u *tensorDimensionValues, Rpp32u *tensorDimensionValuesProduct,
                                               Rpp32u *loopCount, Rpp32u *loopCountTransposed,
                                               Rpp32u dimension1, Rpp32u dimension2)
{
    if (tensorDimensionTemp >= tensorDimension)
    {
        Rpp32u dstPtrLoc = 0;
        for (int i = tensorDimension - 1; i > 0 ; i--)
        {
            dstPtrLoc = dstPtrLoc + (loopCount[i] * tensorDimensionValuesProduct[i - 1]);
        }
        dstPtrLoc += loopCount[0];

        tensor_index_exchange_kernel_host(loopCount, loopCountTransposed, tensorDimension, dimension1, dimension2);

        Rpp32u srcPtrLoc = 0;
        for (int i = tensorDimension - 1; i > 0 ; i--)
        {
            srcPtrLoc = srcPtrLoc + (loopCountTransposed[i] * tensorDimensionValuesProduct[i - 1]);
        }
        srcPtrLoc += loopCountTransposed[0];

        *(dstPtr + dstPtrLoc) = *(srcPtr + srcPtrLoc);

        return RPP_SUCCESS;
    }
    for (int i = 0; i < *(tensorDimensionValues + tensorDimensionTemp); i++)
    {
        *(loopCount + tensorDimensionTemp) = i;
        tensor_transpose_iterate_kernel_host(srcPtr, dstPtr,
                                             tensorDimensionTemp + 1, tensorDimension,
                                             tensorDimensionValues, tensorDimensionValuesProduct,
                                             loopCount, loopCountTransposed,
                                             dimension1, dimension2);
    }

    return RPP_SUCCESS;
}

template<typename T>
inline RppStatus fast_corner_detector_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                           Rpp32u* bresenhamCirclePositions, T threshold, Rpp32u numOfPixels)
{
    T centerPixel = *(srcPtrWindow + (3 * srcSize.width) + 3);
    T max = (T) (RPPPIXELCHECK((Rpp32s) centerPixel + (Rpp32s) threshold));
    T min = (T) (RPPPIXELCHECK((Rpp32s) centerPixel - (Rpp32s) threshold));

    // Find Bresenham Circle for the pixel

    Rpp32u *bresenhamCirclePositionsTemp;
    bresenhamCirclePositionsTemp = bresenhamCirclePositions;

    T *bresenhamCircle = (T*) calloc(16, sizeof(T));
    T *bresenhamCircleTemp;
    bresenhamCircleTemp = bresenhamCircle;

    T* bresenhamCircleOutput = (T*) calloc(16, sizeof(T));

    for (int i = 0; i < 16; i++)
    {
        *bresenhamCircleTemp = *(srcPtrWindow + *bresenhamCirclePositionsTemp);
        bresenhamCircleTemp++;
        bresenhamCirclePositionsTemp++;
    }

    Rpp32u flag = 0;

    *bresenhamCircleOutput = (T) RPPISLESSER(*bresenhamCircle, min);
    *(bresenhamCircleOutput + 8) = (T) RPPISLESSER(*(bresenhamCircle + 8), min);

    if (*bresenhamCircleOutput == 1)
    {
        *(bresenhamCircleOutput + 4) = (T) RPPISLESSER(*(bresenhamCircle + 4), min);
        *(bresenhamCircleOutput + 12) = (T) RPPISLESSER(*(bresenhamCircle + 12), min);
        if (*(bresenhamCircleOutput + 8) == 1)
        {
            if (*(bresenhamCircleOutput + 4) == 1 || *(bresenhamCircleOutput + 12) == 1)
            {
                flag = 1;
            }
        }
        else if (*(bresenhamCircleOutput + 4) == 1 && *(bresenhamCircleOutput + 12) == 1)
        {
            flag = 1;
        }
    }
    else if (*(bresenhamCircleOutput + 8) == 1)
    {
        *(bresenhamCircleOutput + 4) = (T) RPPISLESSER(*(bresenhamCircle + 4), min);
        *(bresenhamCircleOutput + 12) = (T) RPPISLESSER(*(bresenhamCircle + 12), min);
        if (*(bresenhamCircleOutput + 4) == 1 && *(bresenhamCircleOutput + 12) == 1)
        {
            flag = 1;
        }
    }
    if (flag == 0)
    {
        *bresenhamCircleOutput = (T) RPPISGREATER(*bresenhamCircle, max);
        *(bresenhamCircleOutput + 8) = (T) RPPISGREATER(*(bresenhamCircle + 8), max);

        if (*bresenhamCircleOutput == 1)
        {
            *(bresenhamCircleOutput + 4) = (T) RPPISGREATER(*(bresenhamCircle + 4), max);
            *(bresenhamCircleOutput + 12) = (T) RPPISGREATER(*(bresenhamCircle + 12), max);
            if (*(bresenhamCircleOutput + 8) == 1)
            {
                if (*(bresenhamCircleOutput + 4) == 1 || *(bresenhamCircleOutput + 12) == 1)
                {
                    flag = 2;
                }
            }
            else if (*(bresenhamCircleOutput + 4) == 1 && *(bresenhamCircleOutput + 12) == 1)
            {
                flag = 2;
            }
        }
        else if (*(bresenhamCircleOutput + 8) == 1)
        {
            *(bresenhamCircleOutput + 4) = (T) RPPISGREATER(*(bresenhamCircle + 4), max);
            *(bresenhamCircleOutput + 12) = (T) RPPISGREATER(*(bresenhamCircle + 12), max);
            if (*(bresenhamCircleOutput + 4) == 1 && *(bresenhamCircleOutput + 12) == 1)
            {
                flag = 2;
            }
        }
    }
    if (flag == 0)
    {
        *dstPtrPixel = (T) 0;

        return RPP_SUCCESS;
    }
    else if (flag == 1)
    {
        *(bresenhamCircleOutput + 1) = (T) RPPISLESSER(*(bresenhamCircle + 1), min);
        *(bresenhamCircleOutput + 2) = (T) RPPISLESSER(*(bresenhamCircle + 2), min);
        *(bresenhamCircleOutput + 3) = (T) RPPISLESSER(*(bresenhamCircle + 3), min);
        *(bresenhamCircleOutput + 5) = (T) RPPISLESSER(*(bresenhamCircle + 5), min);
        *(bresenhamCircleOutput + 6) = (T) RPPISLESSER(*(bresenhamCircle + 6), min);
        *(bresenhamCircleOutput + 7) = (T) RPPISLESSER(*(bresenhamCircle + 7), min);
        *(bresenhamCircleOutput + 9) = (T) RPPISLESSER(*(bresenhamCircle + 9), min);
        *(bresenhamCircleOutput + 10) = (T) RPPISLESSER(*(bresenhamCircle + 10), min);
        *(bresenhamCircleOutput + 11) = (T) RPPISLESSER(*(bresenhamCircle + 11), min);
        *(bresenhamCircleOutput + 13) = (T) RPPISLESSER(*(bresenhamCircle + 13), min);
        *(bresenhamCircleOutput + 14) = (T) RPPISLESSER(*(bresenhamCircle + 14), min);
        *(bresenhamCircleOutput + 15) = (T) RPPISLESSER(*(bresenhamCircle + 15), min);
    }
    else if (flag == 2)
    {
        *(bresenhamCircleOutput + 1) = (T) RPPISGREATER(*(bresenhamCircle + 1), max);
        *(bresenhamCircleOutput + 2) = (T) RPPISGREATER(*(bresenhamCircle + 2), max);
        *(bresenhamCircleOutput + 3) = (T) RPPISGREATER(*(bresenhamCircle + 3), max);
        *(bresenhamCircleOutput + 5) = (T) RPPISGREATER(*(bresenhamCircle + 5), max);
        *(bresenhamCircleOutput + 6) = (T) RPPISGREATER(*(bresenhamCircle + 6), max);
        *(bresenhamCircleOutput + 7) = (T) RPPISGREATER(*(bresenhamCircle + 7), max);
        *(bresenhamCircleOutput + 9) = (T) RPPISGREATER(*(bresenhamCircle + 9), max);
        *(bresenhamCircleOutput + 10) = (T) RPPISGREATER(*(bresenhamCircle + 10), max);
        *(bresenhamCircleOutput + 11) = (T) RPPISGREATER(*(bresenhamCircle + 11), max);
        *(bresenhamCircleOutput + 13) = (T) RPPISGREATER(*(bresenhamCircle + 13), max);
        *(bresenhamCircleOutput + 14) = (T) RPPISGREATER(*(bresenhamCircle + 14), max);
        *(bresenhamCircleOutput + 15) = (T) RPPISGREATER(*(bresenhamCircle + 15), max);
    }

    // Find maximum contiguous pixels in bresenhamCircleOutput with value 1

    Rpp32u count = 0;
    Rpp32u maxLength = 0;

    for (int i = 0; i < 32; i++)
    {
        if (*(bresenhamCircleOutput + (i % 16)) == 0)
        {
            count = 0;
            if (i >= 16)
            {
                break;
            }
        }
        else
        {
            count++;
            maxLength = RPPMAX2(maxLength, count);
        }
    }

    // Corner Classification

    if (maxLength >= numOfPixels)
    {
        *dstPtrPixel = (T) 255;
    }
    else
    {
        *dstPtrPixel = (T) 0;
    }

    free(bresenhamCircle);
    free(bresenhamCircleOutput);

    return RPP_SUCCESS;
}

template<typename T, typename U>
inline RppStatus fast_corner_detector_score_function_kernel_host(T* srcPtrWindow, U* dstPtrPixel, RppiSize srcSize,
                                                          Rpp32u* bresenhamCirclePositions, U centerPixel)
{
    U* bresenhamCircle = (U*) calloc(16, sizeof(U));
    U *bresenhamCircleTemp;
    bresenhamCircleTemp = bresenhamCircle;
    Rpp32u *bresenhamCirclePositionsTemp;
    bresenhamCirclePositionsTemp = bresenhamCirclePositions;

    for (int i = 0; i < 16; i++)
    {
        *bresenhamCircleTemp = (U) *(srcPtrWindow + *bresenhamCirclePositionsTemp);
        bresenhamCircleTemp++;
        bresenhamCirclePositionsTemp++;
    }

    U score = 0;
    bresenhamCircleTemp = bresenhamCircle;
    for (int i = 0; i < 16; i++)
    {
        score += RPPABS(centerPixel - *bresenhamCircleTemp);
        bresenhamCircleTemp++;
    }

    *dstPtrPixel = score;

    free(bresenhamCircle);

    return RPP_SUCCESS;
}

template<typename T, typename U, typename V>
inline RppStatus hog_single_channel_gradient_computations_kernel_host(T* srcPtr, RppiSize srcSize, U* gradientX, U* gradientY, U* gradientMagnitude, V* gradientDirection,
                                                               Rpp32f* gradientKernel, RppiSize rppiGradientKernelSizeX, RppiSize rppiGradientKernelSizeY)
{
    custom_convolve_image_host(srcPtr, srcSize, gradientX, gradientKernel, rppiGradientKernelSizeX, RPPI_CHN_PLANAR, 1);
    custom_convolve_image_host(srcPtr, srcSize, gradientY, gradientKernel, rppiGradientKernelSizeY, RPPI_CHN_PLANAR, 1);
    compute_magnitude_host(gradientX, gradientY, srcSize, gradientMagnitude, RPPI_CHN_PLANAR, 1);
    compute_gradient_direction_host(gradientX, gradientY, srcSize, gradientDirection, RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;
}

template<typename T, typename U, typename V>
inline RppStatus hog_three_channel_gradient_computations_kernel_host(T* srcPtr, T* srcPtrSingleChannel, RppiSize srcSize,
                                                              U* gradientX0, U* gradientY0, U* gradientX1, U* gradientY1, U* gradientX2, U* gradientY2,
                                                              U* gradientX, U* gradientY,
                                                              U* gradientMagnitude, V* gradientDirection,
                                                              Rpp32f* gradientKernel, RppiSize rppiGradientKernelSizeX, RppiSize rppiGradientKernelSizeY,
                                                              RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u imageDim = srcSize.height * srcSize.width;

    compute_channel_extract_host(srcPtr, srcSize, srcPtrSingleChannel, 0, chnFormat, channel);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientX0, gradientKernel, rppiGradientKernelSizeX, RPPI_CHN_PLANAR, 1);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientY0, gradientKernel, rppiGradientKernelSizeY, RPPI_CHN_PLANAR, 1);

    compute_channel_extract_host(srcPtr, srcSize, srcPtrSingleChannel, 1, chnFormat, channel);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientX1, gradientKernel, rppiGradientKernelSizeX, RPPI_CHN_PLANAR, 1);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientY1, gradientKernel, rppiGradientKernelSizeY, RPPI_CHN_PLANAR, 1);

    compute_channel_extract_host(srcPtr, srcSize, srcPtrSingleChannel, 2, chnFormat, channel);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientX2, gradientKernel, rppiGradientKernelSizeX, RPPI_CHN_PLANAR, 1);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientY2, gradientKernel, rppiGradientKernelSizeY, RPPI_CHN_PLANAR, 1);

    compute_max_host(gradientX0, gradientX1, srcSize, gradientX, channel);
    memcpy(gradientX0, gradientX, imageDim * sizeof(Rpp32s));
    compute_max_host(gradientX0, gradientX2, srcSize, gradientX, channel);

    compute_max_host(gradientY0, gradientY1, srcSize, gradientY, channel);
    memcpy(gradientY0, gradientY, imageDim * sizeof(Rpp32s));
    compute_max_host(gradientY0, gradientY2, srcSize, gradientY, channel);

    compute_magnitude_host(gradientX, gradientY, srcSize, gradientMagnitude, RPPI_CHN_PLANAR, 1);
    compute_gradient_direction_host(gradientX, gradientY, srcSize, gradientDirection, RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;
}



















// Convolution Functions

template<typename T>
inline RppStatus convolve_image_host_batch(T* srcPtrImage, RppiSize srcSize, RppiSize srcSizeMax, T* dstPtrImage,
                                           T* srcPtrBoundedROI, RppiSize srcSizeBoundedROI,
                                           Rpp32f* kernel, RppiSize kernelSize,
                                           Rpp32f x1, Rpp32f y1, Rpp32f x2, Rpp32f y2,
                                           RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u imageDimMax = srcSizeMax.height * srcSizeMax.width;
    Rpp32u imageDimROI = srcSizeBoundedROI.height * srcSizeBoundedROI.width;

    T maxVal = (T)(std::numeric_limits<T>::max());
    T minVal = (T)(std::numeric_limits<T>::min());

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSizeBoundedROI.width - kernelSize.width;

        for(int c = 0; c < channel; c++)
        {
            T *srcPtrBoundedROIChannel, *srcPtrChannel, *dstPtrChannel;
            srcPtrBoundedROIChannel = srcPtrBoundedROI + (c * imageDimROI);
            srcPtrChannel = srcPtrImage + (c * imageDimMax);
            dstPtrChannel = dstPtrImage + (c * imageDimMax);

            Rpp32u roiRowCount = 0;


            for(int i = 0; i < srcSize.height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrChannel + (i * srcSizeMax.width);
                dstPtrTemp = dstPtrChannel + (i * srcSizeMax.width);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, srcSize.width * sizeof(T));

                    dstPtrTemp += srcSizeMax.width;
                    srcPtrTemp += srcSizeMax.width;
                }
                else
                {
                    T *srcPtrWindow;
                    srcPtrWindow = srcPtrBoundedROIChannel + (roiRowCount * srcSizeBoundedROI.width);
                    for(int j = 0; j < srcSize.width; j++)
                    {
                        if((x1 <= j) && (j <= x2 ))
                        {
                            convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                    kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                    chnFormat, channel);

                            srcPtrWindow++;
                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        else
                        {
                            *dstPtrTemp = *srcPtrTemp;

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                    }
                    #pragma omp critical
                    roiRowCount++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = (srcSizeBoundedROI.width - kernelSize.width) * channel;
        Rpp32u elementsInRowBoundedROI = channel * srcSizeBoundedROI.width;
        Rpp32u elementsInRowMax = channel * srcSizeMax.width;
        Rpp32u elementsInRow = channel * srcSize.width;

        Rpp32u roiRowCount = 0;


        for(int i = 0; i < srcSize.height; i++)
        {
            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
            dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

            if (!((y1 <= i) && (i <= y2)))
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                dstPtrTemp += elementsInRowMax;
                srcPtrTemp += elementsInRowMax;
            }
            else
            {
                T *srcPtrWindow;
                srcPtrWindow = srcPtrBoundedROI + (roiRowCount * elementsInRowBoundedROI);
                for(int j = 0; j < srcSize.width; j++)
                {
                    if (!((x1 <= j) && (j <= x2 )))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));

                        dstPtrTemp += channel;
                        srcPtrTemp += channel;
                    }
                    else
                    {
                        for(int c = 0; c < channel; c++)
                        {

                            convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                    kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                    chnFormat, channel);

                            srcPtrWindow++;
                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                    }
                }
                #pragma omp critical
                roiRowCount++;
            }
        }
    }



    return RPP_SUCCESS;
}

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
            for (int i = 0; i < srcSizeSubImage.height; i++)
            {
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
        for (int i = 0; i < srcSizeSubImage.height; i++)
        {
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

template <typename T, typename U>
inline RppStatus custom_convolve_image_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                                  Rpp32f *kernel, RppiSize rppiKernelSize,
                                  RppiChnFormat chnFormat, Rpp32u channel)
{
    if (rppiKernelSize.height % 2 == 0 || rppiKernelSize.width % 2 == 0)
    {
        return RPP_ERROR;
    }

    int boundY = ((rppiKernelSize.height - 1) / 2);
    int boundX = ((rppiKernelSize.width - 1) / 2);

    RppiSize srcSizeMod1, srcSizeMod2;

    srcSizeMod1.height = srcSize.height + boundY;
    srcSizeMod1.width = srcSize.width + boundX;
    T *srcPtrMod1 = (T *)calloc(srcSizeMod1.height * srcSizeMod1.width * channel, sizeof(T));
    generate_corner_padded_image_host(srcPtr, srcSize, srcPtrMod1, srcSizeMod1, 1, chnFormat, channel);

    srcSizeMod2.height = srcSizeMod1.height + boundY;
    srcSizeMod2.width = srcSizeMod1.width + boundX;
    T *srcPtrMod2 = (T *)calloc(srcSizeMod2.height * srcSizeMod2.width * channel, sizeof(T));
    generate_corner_padded_image_host(srcPtrMod1, srcSizeMod1, srcPtrMod2, srcSizeMod2, 4, chnFormat, channel);

    convolve_image_host(srcPtrMod2, srcSizeMod2, dstPtr, srcSize, kernel, rppiKernelSize, chnFormat, channel);

    free(srcPtrMod1);
    free(srcPtrMod2);

    return RPP_SUCCESS;
}

// Compute Functions for RPP Tensor API

inline RppStatus compute_brightness_48_host(__m256 *p, __m256 *pBrightnessParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm256_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm256_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[3] = _mm256_fmadd_ps(p[3], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[4] = _mm256_fmadd_ps(p[4], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[5] = _mm256_fmadd_ps(p[5], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_brightness_48_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[3] = _mm_fmadd_ps(p[3], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[4] = _mm_fmadd_ps(p[4], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[5] = _mm_fmadd_ps(p[5], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[6] = _mm_fmadd_ps(p[6], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[7] = _mm_fmadd_ps(p[7], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[8] = _mm_fmadd_ps(p[8], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[9] = _mm_fmadd_ps(p[9], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[10] = _mm_fmadd_ps(p[10], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[11] = _mm_fmadd_ps(p[11], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_brightness_24_host(__m256 *p, __m256 *pBrightnessParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm256_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm256_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_brightness_24_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[3] = _mm_fmadd_ps(p[3], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[4] = _mm_fmadd_ps(p[4], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[5] = _mm_fmadd_ps(p[5], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_brightness_16_host(__m256 *p, __m256 *pBrightnessParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm256_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_brightness_16_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[3] = _mm_fmadd_ps(p[3], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_brightness_12_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[2] = _mm_fmadd_ps(p[2], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_brightness_8_host(__m256 *p, __m256 *pBrightnessParams)
{
    p[0] = _mm256_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_brightness_8_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment
    p[1] = _mm_fmadd_ps(p[1], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_brightness_4_host(__m128 *p, __m128 *pBrightnessParams)
{
    p[0] = _mm_fmadd_ps(p[0], pBrightnessParams[0], pBrightnessParams[1]);    // brightness adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_exposure_48_host(__m256 *p, __m256 &pExposureParam)
{
    p[0] = _mm256_mul_ps(p[0], pExposureParam);    // exposure adjustment
    p[1] = _mm256_mul_ps(p[1], pExposureParam);    // exposure adjustment
    p[2] = _mm256_mul_ps(p[2], pExposureParam);    // exposure adjustment
    p[3] = _mm256_mul_ps(p[3], pExposureParam);    // exposure adjustment
    p[4] = _mm256_mul_ps(p[4], pExposureParam);    // exposure adjustment
    p[5] = _mm256_mul_ps(p[5], pExposureParam);    // exposure adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_exposure_24_host(__m256 *p, __m256 &pExposureParam)
{
    p[0] = _mm256_mul_ps(p[0], pExposureParam);    // exposure adjustment
    p[1] = _mm256_mul_ps(p[1], pExposureParam);    // exposure adjustment
    p[2] = _mm256_mul_ps(p[2], pExposureParam);    // exposure adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_exposure_16_host(__m256 *p, __m256 &pExposureParam)
{
    p[0] = _mm256_mul_ps(p[0], pExposureParam);    // exposure adjustment
    p[1] = _mm256_mul_ps(p[1], pExposureParam);    // exposure adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_exposure_8_host(__m256 *p, __m256 &pExposureParam)
{
    p[0] = _mm256_mul_ps(p[0], pExposureParam);    // exposure adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_spatter_48_host(__m256 *p, __m256 *pSpatterMaskInv, __m256 *pSpatterMask, __m256 *pSpatterValue)
{
    p[0] = _mm256_fmadd_ps(p[0], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm256_fmadd_ps(p[1], pSpatterMaskInv[1], _mm256_mul_ps(pSpatterValue[0], pSpatterMask[1]));    // spatter adjustment
    p[2] = _mm256_fmadd_ps(p[2], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[1], pSpatterMask[0]));    // spatter adjustment
    p[3] = _mm256_fmadd_ps(p[3], pSpatterMaskInv[1], _mm256_mul_ps(pSpatterValue[1], pSpatterMask[1]));    // spatter adjustment
    p[4] = _mm256_fmadd_ps(p[4], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[2], pSpatterMask[0]));    // spatter adjustment
    p[5] = _mm256_fmadd_ps(p[5], pSpatterMaskInv[1], _mm256_mul_ps(pSpatterValue[2], pSpatterMask[1]));    // spatter adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_spatter_24_host(__m256 *p, __m256 *pSpatterMaskInv, __m256 *pSpatterMask, __m256 *pSpatterValue)
{
    p[0] = _mm256_fmadd_ps(p[0], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm256_fmadd_ps(p[1], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[1], pSpatterMask[0]));    // spatter adjustment
    p[2] = _mm256_fmadd_ps(p[2], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[2], pSpatterMask[0]));    // spatter adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_spatter_16_host(__m256 *p, __m256 *pSpatterMaskInv, __m256 *pSpatterMask, __m256 pSpatterValue)
{
    p[0] = _mm256_fmadd_ps(p[0], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue, pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm256_fmadd_ps(p[1], pSpatterMaskInv[1], _mm256_mul_ps(pSpatterValue, pSpatterMask[1]));    // spatter adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_spatter_8_host(__m256 *p, __m256 *pSpatterMaskInv, __m256 *pSpatterMask, __m256 *pSpatterValue)
{
    p[0] = _mm256_fmadd_ps(p[0], pSpatterMaskInv[0], _mm256_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_spatter_48_host(__m128 *p, __m128 *pSpatterMaskInv, __m128 *pSpatterMask, __m128 *pSpatterValue)
{
    p[0] = _mm_fmadd_ps(p[0], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm_fmadd_ps(p[1], pSpatterMaskInv[1], _mm_mul_ps(pSpatterValue[0], pSpatterMask[1]));    // spatter adjustment
    p[2] = _mm_fmadd_ps(p[2], pSpatterMaskInv[2], _mm_mul_ps(pSpatterValue[0], pSpatterMask[2]));    // spatter adjustment
    p[3] = _mm_fmadd_ps(p[3], pSpatterMaskInv[3], _mm_mul_ps(pSpatterValue[0], pSpatterMask[3]));    // spatter adjustment
    p[4] = _mm_fmadd_ps(p[4], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[1], pSpatterMask[0]));    // spatter adjustment
    p[5] = _mm_fmadd_ps(p[5], pSpatterMaskInv[1], _mm_mul_ps(pSpatterValue[1], pSpatterMask[1]));    // spatter adjustment
    p[6] = _mm_fmadd_ps(p[6], pSpatterMaskInv[2], _mm_mul_ps(pSpatterValue[1], pSpatterMask[2]));    // spatter adjustment
    p[7] = _mm_fmadd_ps(p[7], pSpatterMaskInv[3], _mm_mul_ps(pSpatterValue[1], pSpatterMask[3]));    // spatter adjustment
    p[8] = _mm_fmadd_ps(p[8], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[2], pSpatterMask[0]));    // spatter adjustment
    p[9] = _mm_fmadd_ps(p[9], pSpatterMaskInv[1], _mm_mul_ps(pSpatterValue[2], pSpatterMask[1]));    // spatter adjustment
    p[10] = _mm_fmadd_ps(p[10], pSpatterMaskInv[2], _mm_mul_ps(pSpatterValue[2], pSpatterMask[2]));    // spatter adjustment
    p[11] = _mm_fmadd_ps(p[11], pSpatterMaskInv[3], _mm_mul_ps(pSpatterValue[2], pSpatterMask[3]));    // spatter adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_spatter_16_host(__m128 *p, __m128 *pSpatterMaskInv, __m128 *pSpatterMask, __m128 pSpatterValue)
{
    p[0] = _mm_fmadd_ps(p[0], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue, pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm_fmadd_ps(p[1], pSpatterMaskInv[1], _mm_mul_ps(pSpatterValue, pSpatterMask[1]));    // spatter adjustment
    p[2] = _mm_fmadd_ps(p[2], pSpatterMaskInv[2], _mm_mul_ps(pSpatterValue, pSpatterMask[2]));    // spatter adjustment
    p[3] = _mm_fmadd_ps(p[3], pSpatterMaskInv[3], _mm_mul_ps(pSpatterValue, pSpatterMask[3]));    // spatter adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_spatter_12_host(__m128 *p, __m128 *pSpatterMaskInv, __m128 *pSpatterMask, __m128 *pSpatterValue)
{
    p[0] = _mm_fmadd_ps(p[0], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment
    p[1] = _mm_fmadd_ps(p[1], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[1], pSpatterMask[0]));    // spatter adjustment
    p[2] = _mm_fmadd_ps(p[2], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[2], pSpatterMask[0]));    // spatter adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_spatter_4_host(__m128 *p, __m128 *pSpatterMaskInv, __m128 *pSpatterMask, __m128 *pSpatterValue)
{
    p[0] = _mm_fmadd_ps(p[0], pSpatterMaskInv[0], _mm_mul_ps(pSpatterValue[0], pSpatterMask[0]));    // spatter adjustment

    return RPP_SUCCESS;
}

inline RppStatus compute_cmn_48_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_mul_ps(_mm256_sub_ps(p[0], pCMNParams[0]), pCMNParams[1]);
    p[1] = _mm256_mul_ps(_mm256_sub_ps(p[1], pCMNParams[0]), pCMNParams[1]);
    p[2] = _mm256_mul_ps(_mm256_sub_ps(p[2], pCMNParams[0]), pCMNParams[1]);
    p[3] = _mm256_mul_ps(_mm256_sub_ps(p[3], pCMNParams[0]), pCMNParams[1]);
    p[4] = _mm256_mul_ps(_mm256_sub_ps(p[4], pCMNParams[0]), pCMNParams[1]);
    p[5] = _mm256_mul_ps(_mm256_sub_ps(p[5], pCMNParams[0]), pCMNParams[1]);

    return RPP_SUCCESS;
}

inline RppStatus compute_cmn_24_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_mul_ps(_mm256_sub_ps(p[0], pCMNParams[0]), pCMNParams[1]);
    p[1] = _mm256_mul_ps(_mm256_sub_ps(p[1], pCMNParams[0]), pCMNParams[1]);
    p[2] = _mm256_mul_ps(_mm256_sub_ps(p[2], pCMNParams[0]), pCMNParams[1]);

    return RPP_SUCCESS;
}

inline RppStatus compute_cmn_16_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_mul_ps(_mm256_sub_ps(p[0], pCMNParams[0]), pCMNParams[1]);
    p[1] = _mm256_mul_ps(_mm256_sub_ps(p[1], pCMNParams[0]), pCMNParams[1]);

    return RPP_SUCCESS;
}

inline RppStatus compute_cmn_8_host(__m256 *p, __m256 *pCMNParams)
{
    p[0] = _mm256_mul_ps(_mm256_sub_ps(p[0], pCMNParams[0]), pCMNParams[1]);

    return RPP_SUCCESS;
}

inline RppStatus compute_gridmask_masks_16_host(__m128 *pCol, __m128 *pGridRowRatio, __m128 pCosRatio, __m128 pSinRatio, __m128 pGridRatio, __m128 *pMask)
{
    __m128 pCalc[2];

    pCalc[0] = _mm_fmadd_ps(pCol[0], pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol[0], pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask[0] = _mm_or_ps(pCalc[0], pCalc[1]);

    pCalc[0] = _mm_fmadd_ps(pCol[1], pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol[1], pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask[1] = _mm_or_ps(pCalc[0], pCalc[1]);

    pCalc[0] = _mm_fmadd_ps(pCol[2], pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol[2], pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask[2] = _mm_or_ps(pCalc[0], pCalc[1]);

    pCalc[0] = _mm_fmadd_ps(pCol[3], pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol[3], pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask[3] = _mm_or_ps(pCalc[0], pCalc[1]);

    pCol[0] = _mm_add_ps(pCol[0], xmm_p16);
    pCol[1] = _mm_add_ps(pCol[1], xmm_p16);
    pCol[2] = _mm_add_ps(pCol[2], xmm_p16);
    pCol[3] = _mm_add_ps(pCol[3], xmm_p16);

    return RPP_SUCCESS;
}

inline RppStatus compute_gridmask_masks_4_host(__m128 &pCol, __m128 *pGridRowRatio, __m128 pCosRatio, __m128 pSinRatio, __m128 pGridRatio, __m128 &pMask)
{
    __m128 pCalc[2];

    pCalc[0] = _mm_fmadd_ps(pCol, pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol, pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask = _mm_or_ps(pCalc[0], pCalc[1]);
    pCol = _mm_add_ps(pCol, xmm_p4);

    return RPP_SUCCESS;
}

inline RppStatus compute_gridmask_result_48_host(__m128 *p, __m128 *pMask)
{
    p[0] = _mm_and_ps(p[0], pMask[0]);
    p[1] = _mm_and_ps(p[1], pMask[1]);
    p[2] = _mm_and_ps(p[2], pMask[2]);
    p[3] = _mm_and_ps(p[3], pMask[3]);
    p[4] = _mm_and_ps(p[4], pMask[0]);
    p[5] = _mm_and_ps(p[5], pMask[1]);
    p[6] = _mm_and_ps(p[6], pMask[2]);
    p[7] = _mm_and_ps(p[7], pMask[3]);
    p[8] = _mm_and_ps(p[8], pMask[0]);
    p[9] = _mm_and_ps(p[9], pMask[1]);
    p[10] = _mm_and_ps(p[10], pMask[2]);
    p[11] = _mm_and_ps(p[11], pMask[3]);

    return RPP_SUCCESS;
}

inline RppStatus compute_gridmask_result_16_host(__m128 *p, __m128 *pMask)
{
    p[0] = _mm_and_ps(p[0], pMask[0]);
    p[1] = _mm_and_ps(p[1], pMask[1]);
    p[2] = _mm_and_ps(p[2], pMask[2]);
    p[3] = _mm_and_ps(p[3], pMask[3]);

    return RPP_SUCCESS;
}

inline RppStatus compute_gridmask_result_12_host(__m128 *p, __m128 pMask)
{
    p[0] = _mm_and_ps(p[0], pMask);
    p[1] = _mm_and_ps(p[1], pMask);
    p[2] = _mm_and_ps(p[2], pMask);

    return RPP_SUCCESS;
}

inline RppStatus compute_gridmask_result_4_host(__m128 *p, __m128 pMask)
{
    p[0] = _mm_and_ps(p[0], pMask);

    return RPP_SUCCESS;
}

inline RppStatus compute_color_twist_host(RpptFloatRGB *pixel, Rpp32f brightnessParam, Rpp32f contrastParam, Rpp32f hueParam, Rpp32f saturationParam)
{
    // RGB to HSV

    Rpp32f hue, sat, v, add;
    Rpp32f rf, gf, bf, cmax, cmin, delta;
    rf = pixel->R;
    gf = pixel->G;
    bf = pixel->B;
    cmax = RPPMAX3(rf, gf, bf);
    cmin = RPPMIN3(rf, gf, bf);
    delta = cmax - cmin;
    hue = 0.0f;
    sat = 0.0f;
    add = 0.0f;
    if ((delta != 0) && (cmax != 0))
    {
        sat = delta / cmax;
        if (cmax == rf)
        {
            hue = gf - bf;
            add = 0.0f;
        }
        else if (cmax == gf)
        {
            hue = bf - rf;
            add = 2.0f;
        }
        else
        {
            hue = rf - gf;
            add = 4.0f;
        }
        hue /= delta;
    }
    v = cmax;

    // Modify Hue and Saturation

    hue += hueParam + add;
    if (hue >= 6.0f) hue -= 6.0f;
    if (hue < 0) hue += 6.0f;
    sat *= saturationParam;
    sat = std::max(0.0f, std::min(1.0f, sat));

    // HSV to RGB with brightness/contrast adjustment

    Rpp32s hueIntegerPart = (Rpp32s) hue;
    Rpp32f hueFractionPart = hue - hueIntegerPart;
    Rpp32f vsat = v * sat;
    Rpp32f vsatf = vsat * hueFractionPart;
    Rpp32f p = v - vsat;
    Rpp32f q = v - vsatf;
    Rpp32f t = v - vsat + vsatf;
    switch (hueIntegerPart)
    {
        case 0: rf = v; gf = t; bf = p; break;
        case 1: rf = q; gf = v; bf = p; break;
        case 2: rf = p; gf = v; bf = t; break;
        case 3: rf = p; gf = q; bf = v; break;
        case 4: rf = t; gf = p; bf = v; break;
        case 5: rf = v; gf = p; bf = q; break;
    }
    pixel->R = std::fma(rf, brightnessParam, contrastParam);
    pixel->G = std::fma(gf, brightnessParam, contrastParam);
    pixel->B = std::fma(bf, brightnessParam, contrastParam);

    return RPP_SUCCESS;
}

inline RppStatus compute_color_twist_12_host(__m128 &pVecR, __m128 &pVecG, __m128 &pVecB, __m128 *pColorTwistParams)
{
    __m128 pA, pH, pS, pV, pDelta, pAdd, pIntH;
    __m128 pMask[4];
    __m128i pxIntH;

    // RGB to HSV
    pV = _mm_max_ps(pVecR, _mm_max_ps(pVecG, pVecB));                                                               // cmax = RPPMAX3(rf, gf, bf);
    pS = _mm_min_ps(pVecR, _mm_min_ps(pVecG, pVecB));                                                               // cmin = RPPMIN3(rf, gf, bf);
    pDelta = _mm_sub_ps(pV, pS);                                                                                    // delta = cmax - cmin;
    pH = xmm_p0;                                                                                                    // hue = 0.0f;
    pS = xmm_p0;                                                                                                    // sat = 0.0f;
    pAdd = xmm_p0;                                                                                                  // add = 0.0f;
    pMask[0] = _mm_and_ps(_mm_cmpneq_ps(pDelta, xmm_p0), _mm_cmpneq_ps(pV, xmm_p0));                                // if ((delta != 0) && (cmax != 0)) {
    pS = _mm_div_ps(_mm_and_ps(pMask[0], pDelta), pV);                                                              //     sat = delta / cmax;
    pMask[1] = _mm_cmpeq_ps(pV, pVecR);                                                                             //     Temporarily store cmax == rf comparison
    pMask[2] = _mm_and_ps(pMask[0], pMask[1]);                                                                      //     if (cmax == rf)
    pH = _mm_and_ps(pMask[2], _mm_sub_ps(pVecG, pVecB));                                                            //         hue = gf - bf;
    pAdd = _mm_and_ps(pMask[2], xmm_p0);                                                                            //         add = 0.0f;
    pMask[3] = _mm_cmpeq_ps(pV, pVecG);                                                                             //     Temporarily store cmax == gf comparison
    pMask[2] = _mm_andnot_ps(pMask[1], pMask[3]);                                                                   //     else if (cmax == gf)
    pH = _mm_or_ps(_mm_andnot_ps(pMask[2], pH), _mm_and_ps(pMask[2], _mm_sub_ps(pVecB, pVecR)));                    //         hue = bf - rf;
    pAdd = _mm_or_ps(_mm_andnot_ps(pMask[2], pAdd), _mm_and_ps(pMask[2], xmm_p2));                                  //         add = 2.0f;
    pMask[3] = _mm_andnot_ps(pMask[3], _mm_andnot_ps(pMask[1], pMask[0]));                                          //     else
    pH = _mm_or_ps(_mm_andnot_ps(pMask[3], pH), _mm_and_ps(pMask[3], _mm_sub_ps(pVecR, pVecG)));                    //         hue = rf - gf;
    pAdd = _mm_or_ps(_mm_andnot_ps(pMask[3], pAdd), _mm_and_ps(pMask[3], xmm_p4));                                  //         add = 4.0f;
    pH = _mm_or_ps(_mm_andnot_ps(pMask[0], pH), _mm_and_ps(pMask[0], _mm_div_ps(pH, pDelta)));                      //     hue /= delta; }

    // Modify Hue and Saturation
    pH = _mm_add_ps(pH, _mm_add_ps(pColorTwistParams[2], pAdd));                                                    // hue += hueParam + add;
    pH = _mm_sub_ps(pH, _mm_and_ps(_mm_cmpge_ps(pH, xmm_p6), xmm_p6));                                              // if (hue >= 6.0f) hue -= 6.0f;
    pH = _mm_add_ps(pH, _mm_and_ps(_mm_cmplt_ps(pH, xmm_p0), xmm_p6));                                              // if (hue < 0) hue += 6.0f;
    pS = _mm_mul_ps(pS, pColorTwistParams[3]);                                                                      // sat *= saturationParam;
    pS = _mm_max_ps(xmm_p0, _mm_min_ps(xmm_p1, pS));                                                                // sat = std::max(0.0f, std::min(1.0f, sat));

    // HSV to RGB with brightness/contrast adjustment
    pIntH = _mm_floor_ps(pH);                                                                                       // Rpp32s hueIntegerPart = (Rpp32s) hue;
    pxIntH = _mm_cvtps_epi32(pIntH);                                                                                // Convert to epi32
    pH = _mm_sub_ps(pH, pIntH);                                                                                     // Rpp32f hueFractionPart = hue - hueIntegerPart;
    pS = _mm_mul_ps(pV, pS);                                                                                        // Rpp32f vsat = v * sat;
    pAdd = _mm_mul_ps(pS, pH);                                                                                      // Rpp32f vsatf = vsat * hueFractionPart;
    pA = _mm_sub_ps(pV, pS);                                                                                        // Rpp32f p = v - vsat;
    pH = _mm_sub_ps(pV, pAdd);                                                                                      // Rpp32f q = v - vsatf;
    pS = _mm_add_ps(pA, pAdd);                                                                                      // Rpp32f t = v - vsat + vsatf;
    pVecR = xmm_p0;                                                                                                 // Reset dstPtrR
    pVecG = xmm_p0;                                                                                                 // Reset dstPtrG
    pVecB = xmm_p0;                                                                                                 // Reset dstPtrB
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px0));                                                  // switch (hueIntegerPart) {case 0:
    pVecR = _mm_and_ps(pMask[0], pV);                                                                               //     rf = v;
    pVecG = _mm_and_ps(pMask[0], pS);                                                                               //     gf = t;
    pVecB = _mm_and_ps(pMask[0], pA);                                                                               //     bf = p; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px1));                                                  // case 1:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pH));                                    //     rf = q;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pV));                                    //     gf = v;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pA));                                    //     bf = p; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px2));                                                  // case 2:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pA));                                    //     rf = p;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pV));                                    //     gf = v;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pS));                                    //     bf = t; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px3));                                                  // case 3:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pA));                                    //     rf = p;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pH));                                    //     gf = q;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pV));                                    //     bf = v; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px4));                                                  // case 4:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pS));                                    //     rf = t;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pA));                                    //     gf = p;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pV));                                    //     bf = v; break;
    pMask[0] = _mm_castsi128_ps(_mm_cmpeq_epi32(pxIntH, xmm_px5));                                                  // case 5:
    pVecR = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecR), _mm_and_ps(pMask[0], pV));                                    //     rf = v;
    pVecG = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecG), _mm_and_ps(pMask[0], pA));                                    //     gf = p;
    pVecB = _mm_or_ps(_mm_andnot_ps(pMask[0], pVecB), _mm_and_ps(pMask[0], pH));                                    //     bf = q; break;}
    pVecR = _mm_fmadd_ps(pVecR, pColorTwistParams[0], pColorTwistParams[1]);                                        // dstPtrR = rf * brightnessParam + contrastParam;
    pVecG = _mm_fmadd_ps(pVecG, pColorTwistParams[0], pColorTwistParams[1]);                                        // dstPtrG = gf * brightnessParam + contrastParam;
    pVecB = _mm_fmadd_ps(pVecB, pColorTwistParams[0], pColorTwistParams[1]);                                        // dstPtrB = bf * brightnessParam + contrastParam;

    return RPP_SUCCESS;
}

inline RppStatus compute_color_twist_24_host(__m256 &pVecR, __m256 &pVecG, __m256 &pVecB, __m256 *pColorTwistParams)
{
    __m256 pA, pH, pS, pV, pDelta, pAdd, pIntH;
    __m256 pMask[4];
    __m256i pxIntH;

    // RGB to HSV
    pV = _mm256_max_ps(pVecR, _mm256_max_ps(pVecG, pVecB));                                                            // cmax = RPPMAX3(rf, gf, bf);
    pS = _mm256_min_ps(pVecR, _mm256_min_ps(pVecG, pVecB));                                                            // cmin = RPPMIN3(rf, gf, bf);
    pDelta = _mm256_sub_ps(pV, pS);                                                                                    // delta = cmax - cmin;
    pH = avx_p0;                                                                                                       // hue = 0.0f;
    pS = avx_p0;                                                                                                       // sat = 0.0f;
    pAdd = avx_p0;                                                                                                     // add = 0.0f;
    pMask[0] = _mm256_and_ps(_mm256_cmp_ps(pDelta, avx_p0, _CMP_NEQ_OQ), _mm256_cmp_ps(pV, avx_p0, _CMP_NEQ_OQ));      // if ((delta != 0) && (cmax != 0)) {
    pS = _mm256_div_ps(_mm256_and_ps(pMask[0], pDelta), pV);                                                           //     sat = delta / cmax;
    pMask[1] = _mm256_cmp_ps(pV, pVecR, _CMP_EQ_OQ);                                                                   //     Temporarily store cmax == rf comparison
    pMask[2] = _mm256_and_ps(pMask[0], pMask[1]);                                                                      //     if (cmax == rf)
    pH = _mm256_and_ps(pMask[2], _mm256_sub_ps(pVecG, pVecB));                                                         //         hue = gf - bf;
    pAdd = _mm256_and_ps(pMask[2], avx_p0);                                                                            //         add = 0.0f;
    pMask[3] = _mm256_cmp_ps(pV, pVecG, _CMP_EQ_OQ);                                                                   //     Temporarily store cmax == gf comparison
    pMask[2] = _mm256_andnot_ps(pMask[1], pMask[3]);                                                                   //     else if (cmax == gf)
    pH = _mm256_or_ps(_mm256_andnot_ps(pMask[2], pH), _mm256_and_ps(pMask[2], _mm256_sub_ps(pVecB, pVecR)));           //         hue = bf - rf;
    pAdd = _mm256_or_ps(_mm256_andnot_ps(pMask[2], pAdd), _mm256_and_ps(pMask[2], avx_p2));                            //         add = 2.0f;
    pMask[3] = _mm256_andnot_ps(pMask[3], _mm256_andnot_ps(pMask[1], pMask[0]));                                       //     else
    pH = _mm256_or_ps(_mm256_andnot_ps(pMask[3], pH), _mm256_and_ps(pMask[3], _mm256_sub_ps(pVecR, pVecG)));           //         hue = rf - gf;
    pAdd = _mm256_or_ps(_mm256_andnot_ps(pMask[3], pAdd), _mm256_and_ps(pMask[3], avx_p4));                            //         add = 4.0f;
    pH = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pH), _mm256_and_ps(pMask[0], _mm256_div_ps(pH, pDelta)));             //     hue /= delta; }

    // Modify Hue and Saturation
    pH = _mm256_add_ps(pH, _mm256_add_ps(pColorTwistParams[2], pAdd));                                                 // hue += hueParam + add;
    pH = _mm256_sub_ps(pH, _mm256_and_ps(_mm256_cmp_ps(pH, avx_p6, _CMP_GE_OQ), avx_p6));                              // if (hue >= 6.0f) hue -= 6.0f;
    pH = _mm256_add_ps(pH, _mm256_and_ps(_mm256_cmp_ps(pH, avx_p0, _CMP_LT_OQ), avx_p6));                              // if (hue < 0) hue += 6.0f;
    pS = _mm256_mul_ps(pS, pColorTwistParams[3]);                                                                      // sat *= saturationParam;
    pS = _mm256_max_ps(avx_p0, _mm256_min_ps(avx_p1, pS));                                                             // sat = std::max(0.0f, std::min(1.0f, sat));

    // HSV to RGB with brightness/contrast adjustment
    pIntH = _mm256_floor_ps(pH);                                                                                       // Rpp32s hueIntegerPart = (Rpp32s) hue;
    pxIntH = _mm256_cvtps_epi32(pIntH);                                                                                // Convert to epi32
    pH = _mm256_sub_ps(pH, pIntH);                                                                                     // Rpp32f hueFractionPart = hue - hueIntegerPart;
    pS = _mm256_mul_ps(pV, pS);                                                                                        // Rpp32f vsat = v * sat;
    pAdd = _mm256_mul_ps(pS, pH);                                                                                      // Rpp32f vsatf = vsat * hueFractionPart;
    pA = _mm256_sub_ps(pV, pS);                                                                                        // Rpp32f p = v - vsat;
    pH = _mm256_sub_ps(pV, pAdd);                                                                                      // Rpp32f q = v - vsatf;
    pS = _mm256_add_ps(pA, pAdd);                                                                                      // Rpp32f t = v - vsat + vsatf;
    pVecR = avx_p0;                                                                                                    // Reset dstPtrR
    pVecG = avx_p0;                                                                                                    // Reset dstPtrG
    pVecB = avx_p0;                                                                                                    // Reset dstPtrB
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px0));                                               // switch (hueIntegerPart) {case 0:
    pVecR = _mm256_and_ps(pMask[0], pV);                                                                               //     rf = v;
    pVecG = _mm256_and_ps(pMask[0], pS);                                                                               //     gf = t;
    pVecB = _mm256_and_ps(pMask[0], pA);                                                                               //     bf = p; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px1));                                               // case 1:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pH));                              //     rf = q;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pV));                              //     gf = v;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pA));                              //     bf = p; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px2));                                               // case 2:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pA));                              //     rf = p;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pV));                              //     gf = v;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pS));                              //     bf = t; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px3));                                               // case 3:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pA));                              //     rf = p;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pH));                              //     gf = q;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pV));                              //     bf = v; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px4));                                               // case 4:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pS));                              //     rf = t;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pA));                              //     gf = p;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pV));                              //     bf = v; break;
    pMask[0] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pxIntH, avx_px5));                                               // case 5:
    pVecR = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecR), _mm256_and_ps(pMask[0], pV));                              //     rf = v;
    pVecG = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecG), _mm256_and_ps(pMask[0], pA));                              //     gf = p;
    pVecB = _mm256_or_ps(_mm256_andnot_ps(pMask[0], pVecB), _mm256_and_ps(pMask[0], pH));                              //     bf = q; break;}
    pVecR = _mm256_fmadd_ps(pVecR, pColorTwistParams[0], pColorTwistParams[1]);                                        // dstPtrR = rf * brightnessParam + contrastParam;
    pVecG = _mm256_fmadd_ps(pVecG, pColorTwistParams[0], pColorTwistParams[1]);                                        // dstPtrG = gf * brightnessParam + contrastParam;
    pVecB = _mm256_fmadd_ps(pVecB, pColorTwistParams[0], pColorTwistParams[1]);                                        // dstPtrB = bf * brightnessParam + contrastParam;

    return RPP_SUCCESS;
}

inline RppStatus compute_color_cast_48_host(__m128 *p, __m128 pMul, __m128 *pAdd)
{
    p[0] = _mm_fmadd_ps(_mm_sub_ps(p[0], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[1] = _mm_fmadd_ps(_mm_sub_ps(p[1], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[2] = _mm_fmadd_ps(_mm_sub_ps(p[2], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[3] = _mm_fmadd_ps(_mm_sub_ps(p[3], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[4] = _mm_fmadd_ps(_mm_sub_ps(p[4], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Gs
    p[5] = _mm_fmadd_ps(_mm_sub_ps(p[5], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Gs
    p[6] = _mm_fmadd_ps(_mm_sub_ps(p[6], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Gs
    p[7] = _mm_fmadd_ps(_mm_sub_ps(p[7], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Gs
    p[8] = _mm_fmadd_ps(_mm_sub_ps(p[8], pAdd[2]), pMul, pAdd[2]);    // color_cast adjustment Bs
    p[9] = _mm_fmadd_ps(_mm_sub_ps(p[9], pAdd[2]), pMul, pAdd[2]);    // color_cast adjustment Bs
    p[10] = _mm_fmadd_ps(_mm_sub_ps(p[10], pAdd[2]), pMul, pAdd[2]);    // color_cast adjustment Bs
    p[11] = _mm_fmadd_ps(_mm_sub_ps(p[11], pAdd[2]), pMul, pAdd[2]);    // color_cast adjustment Bs

    return RPP_SUCCESS;
}

inline RppStatus compute_color_cast_12_host(__m128 *p, __m128 pMul, __m128 *pAdd)
{
    p[0] = _mm_fmadd_ps(_mm_sub_ps(p[0], pAdd[0]), pMul, pAdd[0]);    // color_cast adjustment Rs
    p[1] = _mm_fmadd_ps(_mm_sub_ps(p[1], pAdd[1]), pMul, pAdd[1]);    // color_cast adjustment Rs
    p[2] = _mm_fmadd_ps(_mm_sub_ps(p[2], pAdd[2]), pMul, pAdd[2]);    // color_cast adjustment Rs

    return RPP_SUCCESS;
}

inline RppStatus compute_xywh_from_ltrb_host(RpptROIPtr roiPtrInput, RpptROIPtr roiPtrImage)
{
    roiPtrImage->xywhROI.xy.x = roiPtrInput->ltrbROI.lt.x;
    roiPtrImage->xywhROI.xy.y = roiPtrInput->ltrbROI.lt.y;
    roiPtrImage->xywhROI.roiWidth = roiPtrInput->ltrbROI.rb.x - roiPtrInput->ltrbROI.lt.x + 1;
    roiPtrImage->xywhROI.roiHeight = roiPtrInput->ltrbROI.rb.y - roiPtrInput->ltrbROI.lt.y + 1;

    return RPP_SUCCESS;
}

inline RppStatus compute_roi_boundary_check_host(RpptROIPtr roiPtrImage, RpptROIPtr roiPtr, RpptROIPtr roiPtrDefault)
{
    roiPtr->xywhROI.xy.x = std::max(roiPtrDefault->xywhROI.xy.x, roiPtrImage->xywhROI.xy.x);
    roiPtr->xywhROI.xy.y = std::max(roiPtrDefault->xywhROI.xy.y, roiPtrImage->xywhROI.xy.y);
    roiPtr->xywhROI.roiWidth = std::min(roiPtrDefault->xywhROI.roiWidth - roiPtrImage->xywhROI.xy.x, roiPtrImage->xywhROI.roiWidth);
    roiPtr->xywhROI.roiHeight = std::min(roiPtrDefault->xywhROI.roiHeight - roiPtrImage->xywhROI.xy.y, roiPtrImage->xywhROI.roiHeight);

    return RPP_SUCCESS;
}

inline RppStatus compute_roi_validation_host(RpptROIPtr roiPtrInput, RpptROIPtr roiPtr, RpptROIPtr roiPtrDefault, RpptRoiType roiType)
{
    if (roiPtrInput == NULL)
    {
        roiPtr = roiPtrDefault;
    }
    else
    {
        RpptROI roiImage;
        RpptROIPtr roiPtrImage = &roiImage;
        if (roiType == RpptRoiType::LTRB)
            compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
        else if (roiType == RpptRoiType::XYWH)
            roiPtrImage = roiPtrInput;
        compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
    }

    return RPP_SUCCESS;
}

inline RppStatus compute_color_jitter_ctm_host(Rpp32f brightnessParam, Rpp32f contrastParam, Rpp32f hueParam, Rpp32f saturationParam, Rpp32f *ctm)
{
    contrastParam += 1.0f;

    alignas(64) Rpp32f hue_saturation_matrix[16] = {0.299f, 0.299f, 0.299f, 0.0f, 0.587f, 0.587f, 0.587f, 0.0f, 0.114f, 0.114f, 0.114f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    alignas(64) Rpp32f brightness_contrast_matrix[16] = {contrastParam, 0.0f, 0.0f, 0.0f, 0.0f, contrastParam, 0.0f, 0.0f, 0.0f, 0.0f, contrastParam, 0.0f, brightnessParam, brightnessParam, brightnessParam, 1.0f};

    Rpp32f sch = saturationParam * cos(hueParam * PI_OVER_180);
    Rpp32f ssh = saturationParam * sin(hueParam * PI_OVER_180);

    __m128 psch = _mm_set1_ps(sch);
    __m128 pssh = _mm_set1_ps(ssh);
    __m128 p0, p1, p2;

    for (int i = 0; i < 16; i+=4)
    {
        p0 = _mm_loadu_ps(hue_saturation_matrix + i);
        p1 = _mm_loadu_ps(sch_mat + i);
        p2 = _mm_loadu_ps(ssh_mat + i);
        p0 = _mm_fmadd_ps(psch, p1, _mm_fmadd_ps(pssh, p2, p0));
        _mm_storeu_ps(hue_saturation_matrix + i, p0);
    }

    fast_matmul4x4_sse(hue_saturation_matrix, brightness_contrast_matrix, ctm);

    return RPP_SUCCESS;
}

inline RppStatus compute_color_jitter_48_host(__m128 *p, __m128 *pCtm)
{
    __m128 pResult[3];

    pResult[0] = _mm_round_ps(_mm_fmadd_ps(p[0], pCtm[0], _mm_fmadd_ps(p[4], pCtm[1], _mm_fmadd_ps(p[8], pCtm[2], pCtm[3]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment R0-R3
    pResult[1] = _mm_round_ps(_mm_fmadd_ps(p[0], pCtm[4], _mm_fmadd_ps(p[4], pCtm[5], _mm_fmadd_ps(p[8], pCtm[6], pCtm[7]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment G0-G3
    pResult[2] = _mm_round_ps(_mm_fmadd_ps(p[0], pCtm[8], _mm_fmadd_ps(p[4], pCtm[9], _mm_fmadd_ps(p[8], pCtm[10], pCtm[11]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment B0-B3
    p[0] = pResult[0];    // color_jitter adjustment R0-R3
    p[4] = pResult[1];    // color_jitter adjustment G0-G3
    p[8] = pResult[2];    // color_jitter adjustment B0-B3
    pResult[0] = _mm_round_ps(_mm_fmadd_ps(p[1], pCtm[0], _mm_fmadd_ps(p[5], pCtm[1], _mm_fmadd_ps(p[9], pCtm[2], pCtm[3]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment R4-R7
    pResult[1] = _mm_round_ps(_mm_fmadd_ps(p[1], pCtm[4], _mm_fmadd_ps(p[5], pCtm[5], _mm_fmadd_ps(p[9], pCtm[6], pCtm[7]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment G4-G7
    pResult[2] = _mm_round_ps(_mm_fmadd_ps(p[1], pCtm[8], _mm_fmadd_ps(p[5], pCtm[9], _mm_fmadd_ps(p[9], pCtm[10], pCtm[11]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment B4-B7
    p[1] = pResult[0];    // color_jitter adjustment R4-R7
    p[5] = pResult[1];    // color_jitter adjustment G4-G7
    p[9] = pResult[2];    // color_jitter adjustment B4-B7
    pResult[0] = _mm_round_ps(_mm_fmadd_ps(p[2], pCtm[0], _mm_fmadd_ps(p[6], pCtm[1], _mm_fmadd_ps(p[10], pCtm[2], pCtm[3]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment R8-R11
    pResult[1] = _mm_round_ps(_mm_fmadd_ps(p[2], pCtm[4], _mm_fmadd_ps(p[6], pCtm[5], _mm_fmadd_ps(p[10], pCtm[6], pCtm[7]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment G8-G11
    pResult[2] = _mm_round_ps(_mm_fmadd_ps(p[2], pCtm[8], _mm_fmadd_ps(p[6], pCtm[9], _mm_fmadd_ps(p[10], pCtm[10], pCtm[11]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment B8-B11
    p[2] = pResult[0];    // color_jitter adjustment R8-R11
    p[6] = pResult[1];    // color_jitter adjustment G8-G11
    p[10] = pResult[2];    // color_jitter adjustment B8-B11
    pResult[0] = _mm_round_ps(_mm_fmadd_ps(p[3], pCtm[0], _mm_fmadd_ps(p[7], pCtm[1], _mm_fmadd_ps(p[11], pCtm[2], pCtm[3]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment R12-R15
    pResult[1] = _mm_round_ps(_mm_fmadd_ps(p[3], pCtm[4], _mm_fmadd_ps(p[7], pCtm[5], _mm_fmadd_ps(p[11], pCtm[6], pCtm[7]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment G12-G15
    pResult[2] = _mm_round_ps(_mm_fmadd_ps(p[3], pCtm[8], _mm_fmadd_ps(p[7], pCtm[9], _mm_fmadd_ps(p[11], pCtm[10], pCtm[11]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment B12-B15
    p[3] = pResult[0];    // color_jitter adjustment R12-R15
    p[7] = pResult[1];    // color_jitter adjustment G12-G15
    p[11] = pResult[2];    // color_jitter adjustment B12-B15

    return RPP_SUCCESS;
}

inline RppStatus compute_color_jitter_12_host(__m128 *p, __m128 *pCtm)
{
    __m128 pResult[3];

    pResult[0] = _mm_fmadd_ps(p[0], pCtm[0], _mm_fmadd_ps(p[1], pCtm[1], _mm_fmadd_ps(p[2], pCtm[2], pCtm[3])));    // color_jitter adjustment R0-R3
    pResult[1] = _mm_fmadd_ps(p[0], pCtm[4], _mm_fmadd_ps(p[1], pCtm[5], _mm_fmadd_ps(p[2], pCtm[6], pCtm[7])));    // color_jitter adjustment G0-G3
    pResult[2] = _mm_fmadd_ps(p[0], pCtm[8], _mm_fmadd_ps(p[1], pCtm[9], _mm_fmadd_ps(p[2], pCtm[10], pCtm[11])));    // color_jitter adjustment B0-B3
    p[0] = pResult[0];    // color_jitter adjustment R0-R3
    p[1] = pResult[1];    // color_jitter adjustment G0-G3
    p[2] = pResult[2];    // color_jitter adjustment B0-B3

    return RPP_SUCCESS;
}

// Compute Functions for RPP Image API

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
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < dstSize.height; i++)
            {
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
        for (int i = 0; i < dstSize.height; i++)
        {
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
inline RppStatus compute_add_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32s pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
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
inline RppStatus compute_bitwise_AND_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                           Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp8u pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp8u) (*srcPtr1Temp)) & ((Rpp8u) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

template <typename T, typename U>
inline RppStatus compute_inclusive_OR_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                            Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp8u pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp8u) (*srcPtr1Temp)) | ((Rpp8u) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

template <typename T, typename U>
inline RppStatus compute_exclusive_OR_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                            Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp8u pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp8u) (*srcPtr1Temp)) ^ ((Rpp8u) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

template <typename T, typename U>
inline RppStatus compute_min_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = RPPMIN2(*srcPtr1Temp, ((T)*srcPtr2Temp));
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

template <typename T, typename U>
inline RppStatus compute_max_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = RPPMAX2(*srcPtr1Temp, ((T)*srcPtr2Temp));
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
        Rpp64u imageDim = srcSize.height * srcSize.width;
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + (imageDim);
        srcPtrTempB = srcPtr + (2 * imageDim);
        dstPtrTempH = dstPtr;
        dstPtrTempS = dstPtr + (imageDim);
        dstPtrTempV = dstPtr + (2 * imageDim);

        Rpp64u bufferLength = srcSize.height * srcSize.width;
        Rpp64u alignedLength = bufferLength & ~3;

        __m128i const zero = _mm_setzero_si128();
        __m128 pDiv = _mm_set1_ps(255.0);
        __m128 pMul = _mm_set1_ps(360.0);
        __m128i px0, px1, px2;
        __m128 xR, xG, xB;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
        {
            px0 =  _mm_loadu_si128((__m128i *)srcPtrTempR);
            px1 =  _mm_loadu_si128((__m128i *)srcPtrTempG);
            px2 =  _mm_loadu_si128((__m128i *)srcPtrTempB);

            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
            px1 = _mm_unpacklo_epi8(px1, zero);    // pixels 0-7
            px2 = _mm_unpacklo_epi8(px2, zero);    // pixels 0-7

            xR = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
            xG = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 0-3
            xB = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px2, zero));    // pixels 0-3

            xR = _mm_div_ps(xR, pDiv);
            xG = _mm_div_ps(xG, pDiv);
            xB = _mm_div_ps(xB, pDiv);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]

            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction. If H >= 1 then H - 1
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Multiply by 360
            xG = _mm_mul_ps(xG, pMul);

            _mm_storeu_ps(dstPtrTempH, xG);
            _mm_storeu_ps(dstPtrTempS, xS);
            _mm_storeu_ps(dstPtrTempV, xV);

            srcPtrTempR += 4;
            srcPtrTempG += 4;
            srcPtrTempB += 4;
            dstPtrTempH += 4;
            dstPtrTempS += 4;
            dstPtrTempV += 4;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
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
        Rpp64u imageDim = srcSize.height * srcSize.width;
        srcPtrTempH = srcPtr;
        srcPtrTempS = srcPtr + (imageDim);
        srcPtrTempV = srcPtr + (2 * imageDim);
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + (imageDim);
        dstPtrTempB = dstPtr + (2 * imageDim);

        Rpp64u bufferLength = srcSize.height * srcSize.width;
        Rpp64u alignedLength = bufferLength & ~3;

        __m128 pDiv = _mm_set1_ps(360.0);
        __m128 pMul = _mm_set1_ps(255.0);

        __m128 h0, h1, h2, h3;
        h0 = _mm_set1_ps(1.0);
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        __m128i px1, px2, px3;

        Rpp8u arrayR[4];
        Rpp8u arrayG[4];
        Rpp8u arrayB[4];

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
        {
            // Load
            h1 =  _mm_loadu_si128((__m128i *)srcPtrTempH);
            h2 =  _mm_loadu_si128((__m128i *)srcPtrTempS);
            h3 =  _mm_loadu_si128((__m128i *)srcPtrTempV);

            h1 = _mm_div_ps(h1, pDiv);

            _MM_TRANSPOSE4_PS (h0, h1, h2, h3);

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            _MM_TRANSPOSE4_PS (x0, x1, x2, x3);

            x1 = _mm_mul_ps(x1, pMul);
            x2 = _mm_mul_ps(x2, pMul);
            x3 = _mm_mul_ps(x3, pMul);

            px1 = _mm_cvtps_epi32(x1);
            px2 = _mm_cvtps_epi32(x2);
            px3 = _mm_cvtps_epi32(x3);

            px1 = _mm_packs_epi32(px1, px1);
            px1 = _mm_packus_epi16(px1, px1);
            *((int*)arrayR) = _mm_cvtsi128_si32(px1);

            px2 = _mm_packs_epi32(px2, px2);
            px2 = _mm_packus_epi16(px2, px2);
            *((int*)arrayG) = _mm_cvtsi128_si32(px2);

            px3 = _mm_packs_epi32(px3, px3);
            px3 = _mm_packus_epi16(px3, px3);
            *((int*)arrayB) = _mm_cvtsi128_si32(px3);

            memcpy(dstPtrTempR, arrayR, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempG, arrayG, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempB, arrayB, 4 * sizeof(Rpp8u));

            srcPtrTempH += 4;
            srcPtrTempS += 4;
            srcPtrTempV += 4;
            dstPtrTempR += 4;
            dstPtrTempG += 4;
            dstPtrTempB += 4;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            Rpp32f c, x, m, rf, gf, bf;
            c = *srcPtrTempV * *srcPtrTempS;
            x = c * (1 - abs(int(fmod((*srcPtrTempH / 60), 2)) - 1));
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

        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f c, x, m, rf, gf, bf;
            c = *srcPtrTempV * *srcPtrTempS;
            x = c * (1 - abs(int(fmod((*srcPtrTempH / 60), 2)) - 1));
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

template <typename T>
inline RppStatus compute_magnitude_ROI_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                                            Rpp32f x1, Rpp32f y1, Rpp32f x2, Rpp32f y2,
                                            RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u imageDim = srcSize.height * srcSize.width;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            T *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
            srcPtr1Channel = srcPtr1 + (c * imageDim);
            srcPtr2Channel = srcPtr2 + (c * imageDim);
            dstPtrChannel = dstPtr + (c * imageDim);


            for(int i = 0; i < srcSize.height; i++)
            {
                Rpp32f pixel;

                T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Channel + (i * srcSize.width);
                srcPtr2Temp = srcPtr2Channel + (i * srcSize.width);
                dstPtrTemp = dstPtrChannel + (i * srcSize.width);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtr1Temp, srcSize.width * sizeof(T));

                    srcPtr1Temp += srcSize.width;
                    srcPtr2Temp += srcSize.width;
                    dstPtrTemp += srcSize.width;
                }
                else
                {
                    for(int j = 0; j < srcSize.width; j++)
                    {
                        if((x1 <= j) && (j <= x2 ))
                        {
                            Rpp32s srcPtr1Value = (Rpp32s) *srcPtr1Temp;
                            Rpp32s srcPtr2Value = (Rpp32s) *srcPtr2Temp;
                            pixel = sqrt((srcPtr1Value * srcPtr1Value) + (srcPtr2Value * srcPtr2Value));
                            pixel = RPPPIXELCHECK(pixel);
                            *dstPtrTemp =(T) round(pixel);

                            srcPtr1Temp++;
                            srcPtr2Temp++;
                            dstPtrTemp++;
                        }
                        else
                        {
                            *dstPtrTemp = *srcPtr1Temp;

                            srcPtr1Temp++;
                            srcPtr2Temp++;
                            dstPtrTemp++;
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;


        for(int i = 0; i < srcSize.height; i++)
        {
            Rpp32f pixel;

            T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
            srcPtr1Temp = srcPtr1 + (i * elementsInRow);
            srcPtr2Temp = srcPtr2 + (i * elementsInRow);
            dstPtrTemp = dstPtr + (i * elementsInRow);

            if (!((y1 <= i) && (i <= y2)))
            {
                memcpy(dstPtrTemp, srcPtr1Temp, elementsInRow * sizeof(T));

                srcPtr1Temp += elementsInRow;
                srcPtr2Temp += elementsInRow;
                dstPtrTemp += elementsInRow;
            }
            else
            {
                for(int j = 0; j < srcSize.width; j++)
                {
                    if (!((x1 <= j) && (j <= x2 )))
                    {
                        memcpy(dstPtrTemp, srcPtr1Temp, channel * sizeof(T));

                        srcPtr1Temp += channel;
                        srcPtr2Temp += channel;
                        dstPtrTemp += channel;
                    }
                    else
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            Rpp32s srcPtr1Value = (Rpp32s) *srcPtr1Temp;
                            Rpp32s srcPtr2Value = (Rpp32s) *srcPtr2Temp;
                            pixel = sqrt((srcPtr1Value * srcPtr1Value) + (srcPtr2Value * srcPtr2Value));
                            pixel = RPPPIXELCHECK(pixel);
                            *dstPtrTemp =(T) round(pixel);

                            srcPtr1Temp++;
                            srcPtr2Temp++;
                            dstPtrTemp++;
                        }
                    }
                }
            }
        }
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

            for (int i = 0; i < dstSize.height; i++)
            {
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

        for (int i = 0; i < dstSize.height; i++)
        {
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

template <typename T>
inline RppStatus compute_channel_extract_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                       Rpp32u extractChannelNumber,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    if (extractChannelNumber != 0 && extractChannelNumber != 1 && extractChannelNumber != 2)
    {
        return RPP_ERROR;
    }

    T *srcPtrTemp, *dstPtrTemp;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        srcPtrTemp = srcPtr + (extractChannelNumber * srcSize.height * srcSize.width);
        memcpy(dstPtrTemp, srcPtrTemp, srcSize.height * srcSize.width * sizeof(T));
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrTemp = srcPtr + extractChannelNumber;
        for (int i = 0; i < srcSize.height * srcSize.width; i++)
        {
            *dstPtrTemp = *srcPtrTemp;
            srcPtrTemp = srcPtrTemp + channel;
            dstPtrTemp++;
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
inline RppStatus compute_gradient_direction_host(T* gradientX, T* gradientY, RppiSize srcSize, U* gradientDirection,
                                          RppiChnFormat chnFormat, Rpp32u channel)
{
    T *gradientXTemp, *gradientYTemp;
    U *gradientDirectionTemp;
    gradientXTemp = gradientX;
    gradientYTemp = gradientY;
    gradientDirectionTemp = gradientDirection;

    Rpp32f pixel;

    for (int i = 0; i < (srcSize.height * srcSize.width * channel); i++)
    {
        if (*gradientXTemp != 0)
        {
            *gradientDirectionTemp = atan((Rpp32f) *gradientYTemp / (Rpp32f) *gradientXTemp);
        }
        else
        {
            if (*gradientYTemp > 0)
            {
                *gradientDirectionTemp = ((Rpp32f) PI) / 2.0;
            }
            else if (*gradientYTemp < 0)
            {
                *gradientDirectionTemp = ((Rpp32f) PI) / 2.0 * -1.0;
            }
            else if (*gradientYTemp == 0)
            {
                *gradientDirectionTemp = 0.0;
            }
        }
        gradientDirectionTemp++;
        gradientXTemp++;
        gradientYTemp++;
    }

    return RPP_SUCCESS;
}

inline Rpp32u fogGenerator(Rpp32u srcPtr, Rpp32f fogValue, int colour, int check)
{
    int fog = 0;
    int range = 3;
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

inline RppStatus compute_image_location_host(RppiSize *batch_srcSizeMax, int batchCount, Rpp32u *loc, Rpp32u channel)
{
    for (int m = 0; m < batchCount; m++)
    {
        *loc += (batch_srcSizeMax[m].height * batch_srcSizeMax[m].width);
    }
    *loc *= channel;

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_1_channel_minmax_host(T *srcPtr, RppiSize srcSize, RppiSize srcSizeMax,
                                                    T *min, T *max,
                                                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp;
    srcPtrTemp = srcPtr;

    __m128i pMin, pMax;

    for (int i = 0; i < srcSize.height; i++)
    {
        pMin = _mm_set1_epi8(*min);
        pMax = _mm_set1_epi8(*max);

        int bufferLength = srcSize.width;
        int alignedLength = bufferLength & ~15;

        __m128i px0;

        int vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
        {
            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
            pMin = _mm_min_epu8(px0, pMin);
            pMax = _mm_max_epu8(px0, pMax);
            srcPtrTemp +=16;
        }
        *min = (T) HorMin(pMin);
        *max = (T) HorMax(pMax);
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            if (*srcPtrTemp < *min)
            {
                *min = *srcPtrTemp;
            }
            if (*srcPtrTemp > *max)
            {
                *max = *srcPtrTemp;
            }
            srcPtrTemp++;
        }
        srcPtrTemp += (srcSizeMax.width - srcSize.width);
    }

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_3_channel_minmax_host(T *srcPtr, RppiSize srcSize, RppiSize srcSizeMax,
                                               T *min, T *max,
                                               RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp;
    srcPtrTemp = srcPtr;

    __m128i pMin, pMax;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            T minTemp, maxTemp;
            minTemp = *(min + c);
            maxTemp = *(max + c);
            for (int i = 0; i < srcSize.height; i++)
            {
                pMin = _mm_set1_epi8(minTemp);
                pMax = _mm_set1_epi8(maxTemp);

                int bufferLength = srcSize.width;
                int alignedLength = bufferLength & ~15;

                __m128i px0;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                    pMin = _mm_min_epu8(px0, pMin);
                    pMax = _mm_max_epu8(px0, pMax);
                    srcPtrTemp +=16;
                }
                minTemp = (T) HorMin(pMin);
                maxTemp = (T) HorMax(pMax);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if (*srcPtrTemp < minTemp)
                    {
                        minTemp = *srcPtrTemp;
                    }
                    if (*srcPtrTemp > maxTemp)
                    {
                        maxTemp = *srcPtrTemp;
                    }
                    srcPtrTemp++;
                }
                srcPtrTemp += (srcSizeMax.width - srcSize.width);
            }
            *(min + c) = minTemp;
            *(max + c) = maxTemp;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u elementsInRowMax = channel * srcSizeMax.width;

        T minRTemp, maxRTemp, minGTemp, maxGTemp, minBTemp, maxBTemp;
        minRTemp = *min;
        maxRTemp = *max;
        minGTemp = *(min + 1);
        maxGTemp = *(max + 1);
        minBTemp = *(min + 2);
        maxBTemp = *(max + 2);

        pMin = _mm_set1_epi8(minRTemp);
        pMax = _mm_set1_epi8(maxRTemp);

        for (int i = 0; i < srcSize.height; i++)
        {
            int bufferLength = elementsInRow;
            int alignedLength = bufferLength & ~14;

            __m128i px0;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                pMin = _mm_min_epu8(px0, pMin);
                pMax = _mm_max_epu8(px0, pMax);
                srcPtrTemp +=15;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
            {
                if (*srcPtrTemp < minRTemp)
                {
                    minRTemp = *srcPtrTemp;
                }
                if (*srcPtrTemp > maxRTemp)
                {
                    maxRTemp = *srcPtrTemp;
                }
                srcPtrTemp++;
                if (*srcPtrTemp < minGTemp)
                {
                    minGTemp = *srcPtrTemp;
                }
                if (*srcPtrTemp > maxGTemp)
                {
                    maxGTemp = *srcPtrTemp;
                }
                srcPtrTemp++;
                if (*srcPtrTemp < minBTemp)
                {
                    minBTemp = *srcPtrTemp;
                }
                if (*srcPtrTemp > maxBTemp)
                {
                    maxBTemp = *srcPtrTemp;
                }
                srcPtrTemp++;
            }
            srcPtrTemp += (elementsInRowMax - elementsInRow);
        }

        T minVector[16], maxVector[16];
        _mm_storeu_si128((__m128i *)minVector, pMin);
        _mm_storeu_si128((__m128i *)maxVector, pMax);

        minRTemp = RPPMIN2(RPPMIN3(minVector[0], minVector[3], minVector[6]), RPPMIN3(minVector[9], minVector[12], minRTemp));
        minGTemp = RPPMIN2(RPPMIN3(minVector[1], minVector[4], minVector[7]), RPPMIN3(minVector[10], minVector[13], minGTemp));
        minBTemp = RPPMIN2(RPPMIN3(minVector[2], minVector[5], minVector[8]), RPPMIN3(minVector[11], minVector[14], minBTemp));

        maxRTemp = RPPMAX2(RPPMAX3(maxVector[0], maxVector[3], maxVector[6]), RPPMAX3(maxVector[9], maxVector[12], maxRTemp));
        maxGTemp = RPPMAX2(RPPMAX3(maxVector[1], maxVector[4], maxVector[7]), RPPMAX3(maxVector[10], maxVector[13], maxGTemp));
        maxBTemp = RPPMAX2(RPPMAX3(maxVector[2], maxVector[5], maxVector[8]), RPPMAX3(maxVector[11], maxVector[14], maxBTemp));

        *min = minRTemp;
        *max = maxRTemp;
        *(min + 1) = minGTemp;
        *(max + 1) = maxGTemp;
        *(min + 2) = minBTemp;
        *(max + 2) = maxBTemp;
    }

    return RPP_SUCCESS;
}

inline RppStatus compute_histogram_location_host(Rpp32u *batch_bins, int batchCount, Rpp32u *locHist)
{
    for (int m = 0; m < batchCount; m++)
    {
        *locHist += batch_bins[m];
    }

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_unpadded_from_padded_host(T* srcPtrPadded, RppiSize srcSize, RppiSize srcSizeMax, T* dstPtrUnpadded,
                                                   RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrPaddedChannel, *srcPtrPaddedRow, *dstPtrUnpaddedRow;
    Rpp32u imageDimMax = srcSizeMax.height * srcSizeMax.width;
    dstPtrUnpaddedRow = dstPtrUnpadded;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrPaddedChannel = srcPtrPadded + (c * imageDimMax);
            for (int i = 0; i < srcSize.height; i++)
            {
                srcPtrPaddedRow = srcPtrPaddedChannel + (i * srcSizeMax.width);
                memcpy(dstPtrUnpaddedRow, srcPtrPaddedRow, srcSize.width * sizeof(T));
                dstPtrUnpaddedRow += srcSize.width;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRowMax = channel * srcSizeMax.width;
        Rpp32u elementsInRow = channel * srcSize.width;
        for (int i = 0; i < srcSize.height; i++)
        {
            srcPtrPaddedRow = srcPtrPadded + (i * elementsInRowMax);
            memcpy(dstPtrUnpaddedRow, srcPtrPaddedRow, elementsInRow * sizeof(T));
            dstPtrUnpaddedRow += elementsInRow;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_padded_from_unpadded_host(T* srcPtrUnpadded, RppiSize srcSize, RppiSize dstSizeMax, T* dstPtrPadded,
                                                   RppiChnFormat chnFormat, Rpp32u channel)
{
    T *dstPtrPaddedChannel, *dstPtrPaddedRow, *srcPtrUnpaddedRow;
    Rpp32u imageDimMax = dstSizeMax.height * dstSizeMax.width;
    srcPtrUnpaddedRow = srcPtrUnpadded;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            dstPtrPaddedChannel = dstPtrPadded + (c * imageDimMax);
            for (int i = 0; i < srcSize.height; i++)
            {
                dstPtrPaddedRow = dstPtrPaddedChannel + (i * dstSizeMax.width);
                memcpy(dstPtrPaddedRow, srcPtrUnpaddedRow, srcSize.width * sizeof(T));
                srcPtrUnpaddedRow += srcSize.width;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRowMax = channel * dstSizeMax.width;
        Rpp32u elementsInRow = channel * srcSize.width;
        for (int i = 0; i < srcSize.height; i++)
        {
            dstPtrPaddedRow = dstPtrPadded + (i * elementsInRowMax);
            memcpy(dstPtrPaddedRow, srcPtrUnpaddedRow, elementsInRow * sizeof(T));
            srcPtrUnpaddedRow += elementsInRow;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_planar_to_packed_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                        Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    for (int c = 0; c < channel; c++)
    {
        dstPtrTemp += c;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                *dstPtrTemp = *srcPtrTemp;
                srcPtrTemp++;
                dstPtrTemp += 3;
            }
        }
        dstPtrTemp = dstPtr;
    }

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_packed_to_planar_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                        Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    for (int c = 0; c < channel; c++)
    {
        srcPtrTemp += c;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                *dstPtrTemp = *srcPtrTemp;
                dstPtrTemp++;
                srcPtrTemp += 3;
            }
        }
        srcPtrTemp = srcPtr;
    }

    return RPP_SUCCESS;
}

/* Resize helper functions */
inline RppStatus compute_dst_size_cap_host(RpptImagePatchPtr dstImgSize, RpptDescPtr dstDescPtr)
{
    dstImgSize->width = std::min(dstImgSize->width, dstDescPtr->w);
    dstImgSize->height = std::min(dstImgSize->height, dstDescPtr->h);

    return RPP_SUCCESS;
}

inline RppStatus compute_resize_src_loc(Rpp32s dstLocation, Rpp32f scale, Rpp32u limit, Rpp32s &srcLoc, Rpp32f *weight, Rpp32f offset = 0, Rpp32u srcStride = 1)
{
    Rpp32f srcLocation = ((Rpp32f) dstLocation) * scale + offset;
    Rpp32s srcLocationFloor = (Rpp32s) RPPFLOOR(srcLocation);
    weight[0] = srcLocation - srcLocationFloor;
    weight[1] = 1 - weight[0];
    srcLoc = ((srcLocationFloor > limit) ? limit : srcLocationFloor) * srcStride;

    return RPP_SUCCESS;
}

inline RppStatus compute_resize_src_loc_avx(__m256 &pDstLoc, __m256 &pScale, __m256 &pLimit, Rpp32s *srcLoc, __m256 *pWeight, __m256 pOffset = avx_p0, bool hasRGBChannels = false)
{
    __m256 pLoc = _mm256_fmadd_ps(pDstLoc, pScale, pOffset);
    pDstLoc = _mm256_add_ps(pDstLoc, avx_p8);
    __m256 pLocFloor = _mm256_floor_ps(pLoc);
    pWeight[0] = _mm256_sub_ps(pLoc, pLocFloor);
    pWeight[1] = _mm256_sub_ps(avx_p1, pWeight[0]);
    pLocFloor = _mm256_max_ps(_mm256_min_ps(pLocFloor, pLimit), avx_p0);
    if(hasRGBChannels)
        pLocFloor = _mm256_mul_ps(pLocFloor, avx_p3);
    __m256i pxLocFloor = _mm256_cvtps_epi32(pLocFloor);
    _mm256_storeu_si256((__m256i*) srcLoc, pxLocFloor);

    return RPP_SUCCESS;
}

inline RppStatus compute_bilinear_coefficients(Rpp32f *weightParams, Rpp32f *bilinearCoeffs)
{
    bilinearCoeffs[0] = weightParams[1] * weightParams[3];    // (1 - weightedHeight) * (1 - weightedWidth)
    bilinearCoeffs[1] = weightParams[1] * weightParams[2];    // (1 - weightedHeight) * weightedWidth
    bilinearCoeffs[2] = weightParams[0] * weightParams[3];    // weightedHeight * (1 - weightedWidth)
    bilinearCoeffs[3] = weightParams[0] * weightParams[2];    // weightedHeight * weightedWidth

    return RPP_SUCCESS;
}

inline RppStatus compute_bilinear_coefficients_avx(__m256 *pWeightParams, __m256 *pBilinearCoeffs)
{
    pBilinearCoeffs[0] = _mm256_mul_ps(pWeightParams[1], pWeightParams[3]);    // (1 - weightedHeight) * (1 - weightedWidth)
    pBilinearCoeffs[1] = _mm256_mul_ps(pWeightParams[1], pWeightParams[2]);    // (1 - weightedHeight) * weightedWidth
    pBilinearCoeffs[2] = _mm256_mul_ps(pWeightParams[0], pWeightParams[3]);    // weightedHeight * (1 - weightedWidth)
    pBilinearCoeffs[3] = _mm256_mul_ps(pWeightParams[0], pWeightParams[2]);    // weightedHeight * weightedWidth

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_bilinear_interpolation_1c(T **srcRowPtrsForInterp, Rpp32s loc, Rpp32f *bilinearCoeffs, T *dstPtr)
{
    *dstPtr = (T)(((*(srcRowPtrsForInterp[0] + loc)) * bilinearCoeffs[0]) +         // TopRow 1st Pixel * coeff0
                  ((*(srcRowPtrsForInterp[0] + loc + 1)) * bilinearCoeffs[1]) +     // TopRow 2nd Pixel * coeff1
                  ((*(srcRowPtrsForInterp[1] + loc)) * bilinearCoeffs[2]) +         // BottomRow 1st Pixel * coeff2
                  ((*(srcRowPtrsForInterp[1] + loc + 1)) * bilinearCoeffs[3]));     // BottomRow 2nd Pixel * coeff3

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_bilinear_interpolation_3c_pkd(T **srcRowPtrsForInterp, Rpp32s loc, Rpp32f *bilinearCoeffs, T *dstPtrR, T *dstPtrG, T *dstPtrB)
{
    Rpp32s channels = 3;
    *dstPtrR = (T)(((*(srcRowPtrsForInterp[0] + loc)) * bilinearCoeffs[0]) +        // TopRow R01 Pixel * coeff0
                   ((*(srcRowPtrsForInterp[0] + loc + 3)) * bilinearCoeffs[1]) +    // TopRow R02 Pixel * coeff1
                   ((*(srcRowPtrsForInterp[1] + loc)) * bilinearCoeffs[2]) +        // BottomRow R01 Pixel * coeff2
                   ((*(srcRowPtrsForInterp[1] + loc + 3)) * bilinearCoeffs[3]));    // BottomRow R02 Pixel * coeff3
    *dstPtrG = (T)(((*(srcRowPtrsForInterp[0] + loc + 1)) * bilinearCoeffs[0]) +    // TopRow G01 Pixel * coeff0
                   ((*(srcRowPtrsForInterp[0] + loc + 4)) * bilinearCoeffs[1]) +    // TopRow G02 Pixel * coeff1
                   ((*(srcRowPtrsForInterp[1] + loc + 1)) * bilinearCoeffs[2]) +    // BottomRow G01 Pixel * coeff2
                   ((*(srcRowPtrsForInterp[1] + loc + 4)) * bilinearCoeffs[3]));    // BottomRow G02 Pixel * coeff3
    *dstPtrB = (T)(((*(srcRowPtrsForInterp[0] + loc + 2)) * bilinearCoeffs[0]) +    // TopRow B01 Pixel * coeff0
                   ((*(srcRowPtrsForInterp[0] + loc + 5)) * bilinearCoeffs[1]) +    // TopRow B02 Pixel * coeff1
                   ((*(srcRowPtrsForInterp[1] + loc + 2)) * bilinearCoeffs[2]) +    // BottomRow B01 Pixel * coeff2
                   ((*(srcRowPtrsForInterp[1] + loc + 5)) * bilinearCoeffs[3]));    // BottomRow B02 Pixel * coeff3

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_bilinear_interpolation_3c_pln(T **srcRowPtrsForInterp, Rpp32s loc, Rpp32f *bilinearCoeffs, T *dstPtrR, T *dstPtrG, T *dstPtrB)
{
    compute_bilinear_interpolation_1c(srcRowPtrsForInterp, loc, bilinearCoeffs, dstPtrR);
    compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[2], loc, bilinearCoeffs, dstPtrG);
    compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[4], loc, bilinearCoeffs, dstPtrB);

    return RPP_SUCCESS;
}

inline RppStatus compute_bilinear_interpolation_1c_avx(__m256 *pSrcPixels, __m256 *pBilinearCoeffs, __m256 &pDstPixels)
{
    pDstPixels = _mm256_fmadd_ps(pSrcPixels[3], pBilinearCoeffs[3], _mm256_fmadd_ps(pSrcPixels[2], pBilinearCoeffs[2],
                 _mm256_fmadd_ps(pSrcPixels[1], pBilinearCoeffs[1], _mm256_mul_ps(pSrcPixels[0], pBilinearCoeffs[0]))));

    return RPP_SUCCESS;
}

inline RppStatus compute_bilinear_interpolation_3c_avx(__m256 *pSrcPixels, __m256 *pBilinearCoeffs, __m256 *pDstPixels)
{
    compute_bilinear_interpolation_1c_avx(&pSrcPixels[0], &pBilinearCoeffs[0], pDstPixels[0]);
    compute_bilinear_interpolation_1c_avx(&pSrcPixels[4], &pBilinearCoeffs[0], pDstPixels[1]);
    compute_bilinear_interpolation_1c_avx(&pSrcPixels[8], &pBilinearCoeffs[0], pDstPixels[2]);

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_src_row_ptrs_for_interpolation(T **rowPtrsForInterp, T *srcPtr, Rpp32s loc, RpptDescPtr descPtr)
{
    rowPtrsForInterp[0] = srcPtr + loc * descPtr->strides.hStride;          // TopRow for bilinear interpolation
    rowPtrsForInterp[1]  = rowPtrsForInterp[0] + descPtr->strides.hStride;  // BottomRow for bilinear interpolation

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_src_row_ptrs_for_interpolation_pln(T **rowPtrsForInterp, T *srcPtr, Rpp32s loc, RpptDescPtr descPtr)
{
    rowPtrsForInterp[0] = srcPtr + loc * descPtr->strides.hStride;          // TopRow for bilinear interpolation (R channel)
    rowPtrsForInterp[1]  = rowPtrsForInterp[0] + descPtr->strides.hStride;  // BottomRow for bilinear interpolation (R channel)
    rowPtrsForInterp[2] = rowPtrsForInterp[0] + descPtr->strides.cStride;   // TopRow for bilinear interpolation (G channel)
    rowPtrsForInterp[3] = rowPtrsForInterp[1] + descPtr->strides.cStride;   // BottomRow for bilinear interpolation (G channel)
    rowPtrsForInterp[4] = rowPtrsForInterp[2] + descPtr->strides.cStride;   // TopRow for bilinear interpolation (B channel)
    rowPtrsForInterp[5] = rowPtrsForInterp[3] + descPtr->strides.cStride;   // BottomRow for bilinear interpolation (B channel)

    return RPP_SUCCESS;
}
#endif //RPP_CPU_COMMON_H