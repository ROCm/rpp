#ifndef RPP_CPU_COMMON_H
#define RPP_CPU_COMMON_H

#include <math.h>
#include <algorithm>

#include <rppdefs.h>

#define PI 3.14159265
#define RAD(deg)                (deg * PI / 180)
#define RPPABS(a)               ((a < 0) ? (-a) : (a))
#define RPPMIN2(a,b)            ((a < b) ? a : b)
#define RPPMIN3(a,b,c)          ((a < b) && (a < c) ?  a : ((b < c) ? b : c))
#define RPPMAX2(a,b)            ((a > b) ? a : b)
#define RPPMAX3(a,b,c)          ((a > b) && (a > c) ?  a : ((b > c) ? b : c))
#define RPPGAUSSIAN(x, sigma)   (exp(-(pow(x, 2))/(2 * pow(sigma, 2))) / (2 * PI * pow(sigma, 2)))
#define RPPDISTANCE(x, y, i, j) (sqrt(pow(x - i, 2) + pow(y - j, 2)))
#define RPPINRANGE(a, x, y)     ((a >= x) && (a <= y) ? 1 : 0)
#define RPPFLOOR(a)             ((int) a)
#define RPPCEIL(a)              ((int) (a + 1.0))
#define RPPISEVEN(a)            ((a % 2 == 0) ? 1 : 0)




// Generate Functions

RppStatus generate_gaussian_kernel_host(Rpp32f stdDev, Rpp32f* kernel, unsigned int kernelSize);

RppStatus generate_gaussian_kernel_asymmetric_host(Rpp32f stdDev, Rpp32f* kernel, Rpp32u kernelSizeX, Rpp32u kernelSizeY);

template <typename T>
RppStatus generate_bilateral_kernel_host(Rpp32f multiplierI, Rpp32f multiplierS, Rpp32f multiplier, Rpp32f* kernel, unsigned int kernelSize, int bound, 
                                         T* srcPtrWindow, RppiSize srcSizeMod, Rpp32u remainingElementsInRow, Rpp32u incrementToWindowCenter, 
                                         RppiChnFormat chnFormat, unsigned int channel);

template <typename T>
RppStatus generate_evenly_padded_image_host(T* srcPtr, RppiSize srcSize, T* srcPtrMod, RppiSize srcSizeMod, 
                                     RppiChnFormat chnFormat, unsigned int channel);

RppStatus generate_box_kernel_host(Rpp32f* kernel, unsigned int kernelSize);

template <typename T>
RppStatus generate_crop_host(T* srcPtr, RppiSize srcSize, T* srcPtrSubImage, RppiSize srcSizeSubImage, T* dstPtr, 
                             RppiChnFormat chnFormat, unsigned int channel);





// Kernels for functions

template<typename T>
RppStatus convolution_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize, 
                                       Rpp32f* kernel, unsigned int kernelSize, Rpp32u remainingElementsInRow, 
                                       RppiChnFormat chnFormat, unsigned int channel);

template<typename T>
RppStatus histogram_kernel_host(T* srcPtr, RppiSize srcSize, Rpp32u* histogram, 
                                Rpp32u bins, 
                                unsigned int channel);

template <typename T>
RppStatus resize_kernel_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           RppiChnFormat chnFormat, unsigned int channel);





// Convolution Functions

template<typename T>
RppStatus convolve_image_host(T* srcPtrMod, RppiSize srcSizeMod, T* dstPtr, RppiSize srcSize, 
                        Rpp32f* kernel, unsigned int kernelSize, 
                        RppiChnFormat chnFormat, unsigned int channel);

template<typename T>
RppStatus convolve_subimage_host(T* srcPtrMod, RppiSize srcSizeMod, T* dstPtr, RppiSize srcSizeSubImage, RppiSize srcSize, 
                        Rpp32f* kernel, unsigned int kernelSize, 
                        RppiChnFormat chnFormat, unsigned int channel);





// Compute Functions

template<typename T>
RppStatus compute_subimage_location_host(T* ptr, T** ptrSubImage, 
                                         RppiSize size, RppiSize *sizeSubImage, 
                                         unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2, 
                                         RppiChnFormat chnFormat, unsigned int channel);

template<typename T>
RppStatus compute_transpose_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize, 
                                 RppiChnFormat chnFormat, unsigned int channel);

template <typename T, typename U>
RppStatus compute_multiply_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   unsigned int channel);

#endif //RPP_CPU_COMMON_H