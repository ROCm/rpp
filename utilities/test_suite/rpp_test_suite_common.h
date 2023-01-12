#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "rpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half/half.hpp>
#include <fstream>
// #include "HOST_NEW/helpers/testSuite_helper.hpp"
// #include "HIP_NEW/helpers/testSuite_helper.hpp"
#include <experimental/filesystem>

using namespace cv;
using namespace std;

#ifndef TESTSUITE_HELPER
#define TESTSUITE_HELPER

#include "rppi.h"

RppStatus compute_image_location_host(RppiSize batch_srcSizeMax, int batchCount, Rpp32u *loc, Rpp32u channel)
{
    for (int m = 0; m < batchCount; m++)
    {
        *loc += (batch_srcSizeMax.height * batch_srcSizeMax.width);
    }
    *loc *= channel;

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

#endif

void remove_substring(string &str, string &pattern)
{
    std::string::size_type i = str.find(pattern);
    while (i != std::string::npos)
    {
        str.erase(i, pattern.length());
        i = str.find(pattern, i);
   }
}

std::string get_interpolation_type(unsigned int val, RpptInterpolationType &interpolationType)
{
    switch(val)
    {
        case 0:
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
            return "NearestNeighbor";
        }
        case 2:
        {
            interpolationType = RpptInterpolationType::BICUBIC;
            return "Bicubic";
        }
        case 3:
        {
            interpolationType = RpptInterpolationType::LANCZOS;
            return "Lanczos";
        }
        case 4:
        {
            interpolationType = RpptInterpolationType::TRIANGULAR;
            return "Triangular";
        }
        case 5:
        {
            interpolationType = RpptInterpolationType::GAUSSIAN;
            return "Gaussian";
        }
        default:
        {
            interpolationType = RpptInterpolationType::BILINEAR;
            return "Bilinear";
        }
    }
}

std::string get_noise_type(unsigned int val)
{
    switch(val)
    {
        case 0: return "SaltAndPepper";
        case 1: return "Gaussian";
        case 2: return "Shot";
        default:return "SaltAndPepper";
    }
}

void set_data_type(int ip_bitDepth, string &funcName, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    if (ip_bitDepth == 0)
    {
        funcName += "_u8_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;
    }
    else if (ip_bitDepth == 1)
    {
        funcName += "_f16_";
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 2)
    {
        funcName += "_f32_";
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 3)
    {
        funcName += "_u8_f16_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 4)
    {
        funcName += "_u8_f32_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 5)
    {
        funcName += "_i8_";
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;
    }
    else if (ip_bitDepth == 6)
    {
        funcName += "_u8_i8_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::I8;
    }
}

void set_nchw_strides(RpptDescPtr descPtr)
{
    // set strides
    if (descPtr->layout == RpptLayout::NHWC)
    {
        descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
        descPtr->strides.hStride = descPtr->c * descPtr->w;
        descPtr->strides.wStride = descPtr->c;
        descPtr->strides.cStride = 1;
    }
    else if(descPtr->layout == RpptLayout::NCHW)
    {
        descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
        descPtr->strides.cStride = descPtr->w * descPtr->h;
        descPtr->strides.hStride = descPtr->w;
        descPtr->strides.wStride = 1;
    }
}

void convert_pln3_to_pkd3(Rpp8u *output, RpptDescPtr descPtr)
{
    unsigned long long bufferSize = (unsigned long long)descPtr->h * (unsigned long long)descPtr->w * (unsigned long long)descPtr->c * (unsigned long long)descPtr->n + (unsigned long long)descPtr->offsetInBytes;
    Rpp8u *outputCopy = (Rpp8u *)calloc(bufferSize, 1);
    memcpy(outputCopy, output, bufferSize);

    Rpp8u *outputTemp, *outputCopyTemp;
    outputTemp = output + descPtr->offsetInBytes;
    outputCopyTemp = outputCopy + descPtr->offsetInBytes;

    for (int count = 0; count < descPtr->n; count++)
    {
        Rpp8u *outputCopyTempR, *outputCopyTempG, *outputCopyTempB;
        outputCopyTempR = outputCopyTemp;
        outputCopyTempG = outputCopyTempR + descPtr->strides.cStride;
        outputCopyTempB = outputCopyTempG + descPtr->strides.cStride;

        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(descPtr->n)
        for (int i = 0; i < descPtr->h; i++)
        {
            for (int j = 0; j < descPtr->w; j++)
            {
                *outputTemp = *outputCopyTempR;
                outputTemp++;
                outputCopyTempR++;
                *outputTemp = *outputCopyTempG;
                outputTemp++;
                outputCopyTempG++;
                *outputTemp = *outputCopyTempB;
                outputTemp++;
                outputCopyTempB++;
            }
        }

        outputCopyTemp += descPtr->strides.nStride;
    }

    free(outputCopy);
}

void convert_pkd3_to_pln3(Rpp8u *input, RpptDescPtr srcDescPtr)
{
    unsigned long long bufferSize = ((unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)srcDescPtr->n) + (unsigned long long)srcDescPtr->offsetInBytes;
    Rpp8u *inputCopy = (Rpp8u *)calloc(bufferSize, sizeof(Rpp8u));
    memcpy(inputCopy, input, bufferSize * sizeof(Rpp8u));

    Rpp8u *inputTemp, *inputCopyTemp;
    inputTemp = input + srcDescPtr->offsetInBytes;

    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(srcDescPtr->n)
    for (int count = 0; count < srcDescPtr->n; count++)
    {
        Rpp8u *inputTempR, *inputTempG, *inputTempB;
        inputTempR = inputTemp + count * srcDescPtr->strides.nStride;
        inputTempG = inputTempR + srcDescPtr->strides.cStride;
        inputTempB = inputTempG + srcDescPtr->strides.cStride;
        Rpp8u *inputCopyTemp = inputCopy + srcDescPtr->offsetInBytes + count * srcDescPtr->strides.nStride;

        for (int i = 0; i < srcDescPtr->h; i++)
        {
            for (int j = 0; j < srcDescPtr->w; j++)
            {
                *inputTempR = *inputCopyTemp;
                inputCopyTemp++;
                inputTempR++;
                *inputTempG = *inputCopyTemp;
                inputCopyTemp++;
                inputTempG++;
                *inputTempB = *inputCopyTemp;
                inputCopyTemp++;
                inputTempB++;
            }
        }
    }

    free(inputCopy);
}

template <typename T>
void compare_output(T* output, string func, string funcName, RpptDescPtr srcDescPtr)
{
    bool isEqual = true;
    string ref_path = get_current_dir_name();
    string pattern = "HOST/build";
    remove_substring(ref_path, pattern);
    string ref_file = ref_path + "REFERENCE_OUTPUT/" + funcName + "/"+ func + ".csv";
    ifstream file(ref_file);

    vector<vector<string>> refOutput;
    vector<string> row;
    string line, word;
    if(file.is_open())
    {
        while(getline(file, line))
        {
            row.clear();
            stringstream str(line);
            while(getline(str, word, ','))
            row.push_back(word);
            refOutput.push_back(row);
        }
    }
    else
        cout<<"Could not open the file\n";

    for(int i = 0; i < refOutput.size(); i++)
    {
        for(int j = 0; j < refOutput[i].size(); j++)
        {
            if( stoi(refOutput[i][j]) != *output)
            {
                isEqual = false;
                break;
            }
            output++;
        }
        cout<<"\n";
    }
    if(isEqual == true)
        cout<<func<<" unit_test "<<"PASS \n";
    else
        cout<<func<<" unit_test "<<"FAIL \n";
}