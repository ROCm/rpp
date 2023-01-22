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
#include <fstream>
#include <experimental/filesystem>

using namespace cv;
using namespace std;

#define CUTOFF 0

std::map<int, string> augmentationMap =
{
    {0, "brightness"},
    {1, "gamma_correction"},
    {2, "blend"},
    {4, "contrast"},
    {8, "noise"},
    {13, "exposure"},
    {20, "flip"},
    {21, "resize"},
    {23, "rotate"},
    {24, "warp_affine"},
    {30, "non_linear_blend"},
    {31, "color_cast"},
    {36, "color_twist"},
    {37, "crop"},
    {38, "crop_mirror_normalize"},
    {39, "resize_crop_mirror"},
    {40, "erode"},
    {41, "dilate"},
    {49, "box_filter"},
    {70, "copy"},
    {80, "resize_mirror_normalize"},
    {83, "grid_mask"},
    {84, "spatter"},
    {85, "swap_channels"},
    {86, "color_to_greyscale"}
};

template <typename T>
inline T validate_pixel_range(T pixel)
{
    pixel = (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255));
    return pixel;
}

inline std::string get_interpolation_type(unsigned int val, RpptInterpolationType &interpolationType)
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

inline std::string get_noise_type(unsigned int val)
{
    switch(val)
    {
        case 0: return "SaltAndPepper";
        case 1: return "Gaussian";
        case 2: return "Shot";
        default:return "SaltAndPepper";
    }
}

inline void set_data_type(int ip_bitDepth, string &funcName, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
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

inline void set_nchw_values(RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, int noOfImages, int maxHeight, int maxWidth, int maxDstHeight, int maxDstWidth, int inputChannel, int layoutType, bool pln1OutTypeCase, string backend)
{
    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    if (backend == "HIP")
        srcDescPtr->offsetInBytes = 64;
    else
        srcDescPtr->offsetInBytes = 0;
    dstDescPtr->offsetInBytes = 0;

    srcDescPtr->n = noOfImages;
    srcDescPtr->h = maxHeight;
    srcDescPtr->w = maxWidth;
    srcDescPtr->c = inputChannel;

    dstDescPtr->n = noOfImages;
    dstDescPtr->h = maxDstHeight;
    dstDescPtr->w = maxDstWidth;
    if (layoutType == 0 || layoutType == 1)
        dstDescPtr->c = (pln1OutTypeCase) ? 1 : inputChannel;
    else
        dstDescPtr->c = inputChannel;

    // Optionally set w stride as a multiple of 8 for src/dst
    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;
}

inline void set_strides(RpptDescPtr descPtr)
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

inline void convert_pln3_to_pkd3(Rpp8u *output, RpptDescPtr descPtr)
{
    unsigned long long bufferSize = ((unsigned long long)descPtr->h * (unsigned long long)descPtr->w * (unsigned long long)descPtr->c * (unsigned long long)descPtr->n) + descPtr->offsetInBytes;
    Rpp8u *outputCopy = (Rpp8u *)calloc(bufferSize, sizeof(Rpp8u));
    memcpy(outputCopy, output, bufferSize * sizeof(Rpp8u));

    Rpp8u *outputCopyTemp;
    outputCopyTemp = outputCopy + descPtr->offsetInBytes;

    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(descPtr->n)
    for (int count = 0; count < descPtr->n; count++)
    {
        Rpp8u *outputCopyTempR, *outputCopyTempG, *outputCopyTempB;
        outputCopyTempR = outputCopyTemp + count * descPtr->strides.nStride;
        outputCopyTempG = outputCopyTempR + descPtr->strides.cStride;
        outputCopyTempB = outputCopyTempG + descPtr->strides.cStride;
        Rpp8u *outputTemp = output + descPtr->offsetInBytes + count * descPtr->strides.nStride;

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
    }

    free(outputCopy);
}

inline void convert_pkd3_to_pln3(Rpp8u *input, RpptDescPtr descPtr)
{
    unsigned long long bufferSize = ((unsigned long long)descPtr->h * (unsigned long long)descPtr->w * (unsigned long long)descPtr->c * (unsigned long long)descPtr->n) + descPtr->offsetInBytes;
    Rpp8u *inputCopy = (Rpp8u *)calloc(bufferSize, sizeof(Rpp8u));
    memcpy(inputCopy, input, bufferSize * sizeof(Rpp8u));

    Rpp8u *inputTemp, *inputCopyTemp;
    inputTemp = input + descPtr->offsetInBytes;

    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(descPtr->n)
    for (int count = 0; count < descPtr->n; count++)
    {
        Rpp8u *inputTempR, *inputTempG, *inputTempB;
        inputTempR = inputTemp + count * descPtr->strides.nStride;
        inputTempG = inputTempR + descPtr->strides.cStride;
        inputTempB = inputTempG + descPtr->strides.cStride;
        Rpp8u *inputCopyTemp = inputCopy + descPtr->offsetInBytes + count * descPtr->strides.nStride;

        for (int i = 0; i < descPtr->h; i++)
        {
            for (int j = 0; j < descPtr->w; j++)
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

inline void remove_substring(string &str, string &pattern)
{
    std::string::size_type i = str.find(pattern);
    while (i != std::string::npos)
    {
        str.erase(i, pattern.length());
        i = str.find(pattern, i);
   }
}

template <typename T>
inline void compare_output(T* output, string func, string funcName, RpptDescPtr descPtr, string backend, RpptROI *roiPtr, int noOfImages)
{
    bool isEqual = true;
    string refPath = get_current_dir_name();
    string pattern = backend + "/build";
    remove_substring(refPath, pattern);
    string refFile = refPath + "REFERENCE_OUTPUT/" + funcName + "/"+ func + ".csv";

    if(backend == "HIP")
        refFile.replace(refFile.find("HIP_"), 4, "");
    else
        refFile.replace(refFile.find("HOST_"), 5, "");

    ifstream file(refFile);
    Rpp8u *refOutput;
    refOutput = (Rpp8u *)malloc(noOfImages * descPtr->strides.nStride * sizeof(Rpp8u));
    string line,word;
    int index = 0;

    // Load the refennce output values from files and store in vector
    if(file.is_open())
    {
        while(getline(file, line))
        {
            stringstream str(line);
            while(getline(str, word, ','))
            {
                refOutput[index] = stoi(word);
                index++;
            }
        }
    }
    else
    {
        cout<<"Could not open the reference output. Please check the path specified\n";
        return;
    }

    for(int c = 0; c < noOfImages; c++)
    {
        Rpp8u *outputTemp = output + c * descPtr->strides.nStride;
        Rpp8u *outputTempRef = refOutput + c * descPtr->strides.nStride;
        for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
        {
            Rpp8u *rowTemp = outputTemp + i * descPtr->strides.hStride;
            Rpp8u *rowTempRef = outputTempRef + i * descPtr->strides.hStride;
            for(int j = 0; j < roiPtr->xywhROI.roiWidth; j++)
            {
                Rpp8u *outVal = rowTemp + j;
                Rpp8u *outRefVal = rowTempRef + j;
                int diff = abs(*outVal - *outRefVal);
                if(diff > CUTOFF)
                {
                    isEqual = false;
                    break;
                }
            }
        }
    }
    if(isEqual == true)
        cout<< func << ": " << "PASSED \n";
    else
        cout<< func << ": " << "FAILED \n";
}

inline void write_image(string dst, Rpp8u *output, RpptDescPtr dstDescPtr, int layoutType, string imageNames[], RpptImagePatch *dstImgSizes, bool pln1OutTypeCase)
{
    mkdir(dst.c_str(), 0700);
    dst += "/";

    Rpp64u count = 0;
    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
    Rpp8u *offsettedOutput = output + dstDescPtr->offsetInBytes;
    for (int j = 0; j < dstDescPtr->n; j++)
    {
        int height = dstImgSizes[j].height;
        int width = dstImgSizes[j].width;
        int outputSize = height * width * dstDescPtr->c;

        Rpp8u *tempOutput = (Rpp8u *)calloc(outputSize, sizeof(Rpp8u));
        Rpp8u *tempOutputRow;
        tempOutputRow = tempOutput;
        Rpp32u elementsInRow = width * dstDescPtr->c;
        Rpp8u *outputRow = offsettedOutput + count;

        for (int k = 0; k < height; k++)
        {
            memcpy(tempOutputRow, outputRow, elementsInRow * sizeof(Rpp8u));
            tempOutputRow += elementsInRow;
            outputRow += elementsInRowMax;
        }
        count += dstDescPtr->strides.nStride;

        string temp;
        temp = dst;
        temp += imageNames[j];

        Mat matOutputImage;
        if (layoutType == 0 || layoutType == 1)
            matOutputImage = (pln1OutTypeCase) ? Mat(height, width, CV_8UC1, tempOutput) : Mat(height, width, CV_8UC3, tempOutput);
        else if (layoutType == 2)
            matOutputImage = Mat(height, width, CV_8UC1, tempOutput);

        imwrite(temp, matOutputImage);
        free(tempOutput);
    }
}