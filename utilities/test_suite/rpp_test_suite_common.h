/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "filesystem.h"
#include "rpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <turbojpeg.h>

#ifdef GPU_SUPPORT
    #include <hip/hip_fp16.h>
#else
    #include <half/half.hpp>
    using half_float::half;
#endif

typedef half Rpp16f;

using namespace cv;
using namespace std;

#define CUTOFF 1
#define DEBUG_MODE 0
#define MAX_IMAGE_DUMP 20
#define MAX_BATCH_SIZE 512
#define GOLDEN_OUTPUT_MAX_HEIGHT 150    // Golden outputs are generated with MAX_HEIGHT set to 150. Changing this constant will result in QA test failures
#define GOLDEN_OUTPUT_MAX_WIDTH 150     // Golden outputs are generated with MAX_WIDTH set to 150. Changing this constant will result in QA test failures

std::map<int, string> augmentationMap =
{
    {0, "brightness"},
    {1, "gamma_correction"},
    {2, "blend"},
    {4, "contrast"},
    {13, "exposure"},
    {29, "water"},
    {31, "color_cast"},
    {34, "lut"},
    {36, "color_twist"},
    {37, "crop"},
    {38, "crop_mirror_normalize"},
    {49, "box_filter"},
    {54, "gaussian_filter"},
    {84, "spatter"},
    {87, "tensor_sum"}
};

template <typename T>
inline T validate_pixel_range(T pixel)
{
    pixel = (pixel < static_cast<Rpp32f>(0)) ? (static_cast<Rpp32f>(0)) : ((pixel < static_cast<Rpp32f>(255)) ? pixel : (static_cast<Rpp32f>(255)));
    return pixel;
}

// replicates the last image in a batch to fill the remaining images in a batch
void replicate_last_image_to_fill_batch(const string& lastFilePath, vector<string>& imageNamesPath, vector<string>& imageNames, const string& lastFileName, int noOfImages, int batchCount)
{
    int remainingImages = batchCount - (noOfImages % batchCount);
    std::string filePath = lastFilePath;
    std::string fileName = lastFileName;
    if (noOfImages > 0 && ( noOfImages < batchCount || noOfImages % batchCount != 0 ))
    {
        for (int i = 0; i < remainingImages; i++)
        {
            imageNamesPath.push_back(filePath);
            imageNames.push_back(fileName);
        }
    }
}

inline size_t get_size_of_data_type(RpptDataType dataType)
{
    if(dataType == RpptDataType::U8)
        return sizeof(Rpp8u);
    else if(dataType == RpptDataType::I8)
        return sizeof(Rpp8s);
    else if(dataType == RpptDataType::F16)
        return sizeof(Rpp16f);
    else if(dataType == RpptDataType::F32)
        return sizeof(Rpp32f);
    else
        return 0;
}

// returns the interpolation type used for image resizing or scaling operations.
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

// returns the noise type applied to an image
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

// returns number of input channels according to layout type
inline int set_input_channels(int layoutType)
{
    if(layoutType == 0 || layoutType == 1)
        return 3;
    else
        return 1;
}

//returns function type
inline string set_function_type(int layoutType, int pln1OutTypeCase, int outputFormatToggle, string backend)
{
    string funcType;
    if(layoutType == 0)
    {
        funcType = "Tensor_" + backend + "_PKD3";
        if (pln1OutTypeCase)
            funcType += "_toPLN1";
        else
        {
            if (outputFormatToggle)
                funcType += "_toPLN3";
            else
                funcType += "_toPKD3";
        }
    }
    else if (layoutType == 1)
    {
        funcType = "Tensor_" + backend + "_PLN3";
        if (pln1OutTypeCase)
            funcType += "_toPLN1";
        else
        {
            if (outputFormatToggle)
                funcType += "_toPKD3";
            else
                funcType += "_toPLN3";
        }
    }
    else
    {
       funcType = "Tensor_" + backend + "_PLN1";
       funcType += "_toPLN1";
    }

    return funcType;
}

// sets descriptor data types of src/dst
inline void set_descriptor_data_type(int ip_bitDepth, string &funcName, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
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

// sets descriptor layout of src/dst
inline void set_descriptor_layout( RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, int layoutType, bool pln1OutTypeCase, int outputFormatToggle)
{
    if(layoutType == 0)
    {
        srcDescPtr->layout = RpptLayout::NHWC;
        // Set src/dst layouts in tensor descriptors
        if (pln1OutTypeCase)
            dstDescPtr->layout = RpptLayout::NCHW;
        else
        {
            if (outputFormatToggle == 0)
                dstDescPtr->layout = RpptLayout::NHWC;
            else if (outputFormatToggle == 1)
                dstDescPtr->layout = RpptLayout::NCHW;
        }
    }
    else if(layoutType == 1)
    {
        srcDescPtr->layout = RpptLayout::NCHW;
        // Set src/dst layouts in tensor descriptors
        if (pln1OutTypeCase)
            dstDescPtr->layout = RpptLayout::NCHW;
        else
        {
            if (outputFormatToggle == 0)
                dstDescPtr->layout = RpptLayout::NCHW;
            else if (outputFormatToggle == 1)
                dstDescPtr->layout = RpptLayout::NHWC;
        }
    }
    else
    {
        // Set src/dst layouts in tensor descriptors
        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
    }
}

// sets values of maxHeight and maxWidth
inline void set_max_dimensions(vector<string>imagePaths, int& maxHeight, int& maxWidth)
{
    tjhandle tjInstance = tjInitDecompress();
    for (const std::string& imagePath : imagePaths)
    {
        FILE* jpegFile = fopen(imagePath.c_str(), "rb");
        if (!jpegFile) {
            std::cerr << "Error opening file: " << imagePath << std::endl;
            continue;
        }

        fseek(jpegFile, 0, SEEK_END);
        long fileSize = ftell(jpegFile);
        fseek(jpegFile, 0, SEEK_SET);

        std::vector<unsigned char> jpegBuffer(fileSize);
        fread(jpegBuffer.data(), 1, fileSize, jpegFile);
        fclose(jpegFile);

        int jpegSubsamp;
        int width, height;
        if (tjDecompressHeader2(tjInstance, jpegBuffer.data(), jpegBuffer.size(), &width, &height, &jpegSubsamp) == -1) {
            std::cerr << "Error decompressing file: " << imagePath << std::endl;
            continue;
        }

        maxWidth = max(maxWidth, width);
        maxHeight = max(maxHeight, height);
    }
    tjDestroy(tjInstance);
}

// sets roi xywh values and dstImg sizes
inline void  set_src_and_dst_roi(vector<string>::const_iterator imagePathsStart, vector<string>::const_iterator imagePathsEnd, RpptROI *roiTensorPtrSrc, RpptROI *roiTensorPtrDst, RpptImagePatchPtr dstImgSizes)
{
    tjhandle tjInstance = tjInitDecompress();
    int i = 0;
    for (auto imagePathIter = imagePathsStart; imagePathIter != imagePathsEnd; ++imagePathIter, i++)
    {
        const string& imagePath = *imagePathIter;
        FILE* jpegFile = fopen(imagePath.c_str(), "rb");
        if (!jpegFile) {
            std::cerr << "Error opening file: " << imagePath << std::endl;
            continue;
        }

        fseek(jpegFile, 0, SEEK_END);
        long fileSize = ftell(jpegFile);
        fseek(jpegFile, 0, SEEK_SET);

        std::vector<unsigned char> jpegBuffer(fileSize);
        fread(jpegBuffer.data(), 1, fileSize, jpegFile);
        fclose(jpegFile);

        int jpegSubsamp;
        int width, height;
        if (tjDecompressHeader2(tjInstance, jpegBuffer.data(), jpegBuffer.size(), &width, &height, &jpegSubsamp) == -1) {
            std::cerr << "Error decompressing file: " << imagePath << std::endl;
            continue;
        }

        roiTensorPtrSrc[i].xywhROI = {0, 0, width, height};
        roiTensorPtrDst[i].xywhROI = {0, 0, width, height};
        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth;
        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight;
    }
    tjDestroy(tjInstance);
}

// sets generic descriptor dimensions and strides of src/dst
inline void set_generic_descriptor(RpptGenericDescPtr descriptorPtr3D, int noOfImages, int maxX, int maxY, int maxZ, int numChannels, int offsetInBytes, int layoutType)
{
    descriptorPtr3D->numDims = 5;
    descriptorPtr3D->offsetInBytes = offsetInBytes;
    descriptorPtr3D->dataType = RpptDataType::F32;

    if (layoutType == 0)
    {
        descriptorPtr3D->layout = RpptLayout::NCDHW;
        descriptorPtr3D->dims[0] = noOfImages;
        descriptorPtr3D->dims[1] = numChannels;
        descriptorPtr3D->dims[2] = maxZ;
        descriptorPtr3D->dims[3] = maxY;
        descriptorPtr3D->dims[4] = maxX;
    }
    else if (layoutType == 1)
    {
        descriptorPtr3D->layout = RpptLayout::NDHWC;
        descriptorPtr3D->dims[0] = noOfImages;
        descriptorPtr3D->dims[1] = maxZ;
        descriptorPtr3D->dims[2] = maxY;
        descriptorPtr3D->dims[3] = maxX;
        descriptorPtr3D->dims[4] = numChannels;
    }

    descriptorPtr3D->strides[0] = descriptorPtr3D->dims[1] * descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[1] = descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[2] = descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[3] = descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[4] = 1;
}

// sets descriptor dimensions and strides of src/dst
inline void set_descriptor_dims_and_strides(RpptDescPtr descPtr, int noOfImages, int maxHeight, int maxWidth, int numChannels, int offsetInBytes)
{
    descPtr->numDims = 4;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->n = noOfImages;
    descPtr->h = maxHeight;
    descPtr->w = maxWidth;
    descPtr->c = numChannels;

    // Optionally set w stride as a multiple of 8 for src/dst
    descPtr->w = ((descPtr->w / 8) * 8) + 8;
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

inline void set_roi_values(RpptROI *roi, RpptROI *roiTensorPtrSrc, RpptRoiType roiType, int batchSize)
{
    if(roiType == RpptRoiType::XYWH)
        for (int i = 0; i < batchSize; i++)
            roiTensorPtrSrc[i].xywhROI = roi->xywhROI;
    else if(roiType == RpptRoiType::LTRB)
        for (int i = 0; i < batchSize; i++)
            roiTensorPtrSrc[i].ltrbROI = roi->ltrbROI;
}

inline void convert_roi(RpptROI *roiTensorPtrSrc, RpptRoiType roiType, int batchSize)
{
    if(roiType == RpptRoiType::LTRB)
    {
        for (int i = 0; i < batchSize; i++)
        {
            RpptRoiXywh roi = roiTensorPtrSrc[i].xywhROI;
            roiTensorPtrSrc[i].ltrbROI = {roi.xy.x, roi.xy.y, roi.roiWidth - roi.xy.x, roi.roiHeight - roi.xy.y};
        }
    }
    else
    {
        for (int i = 0; i < batchSize; i++)
        {
            RpptRoiLtrb roi = roiTensorPtrSrc[i].ltrbROI;
            roiTensorPtrSrc[i].xywhROI = {roi.lt.x, roi.lt.y, roi.rb.x - roi.lt.x + 1, roi.rb.y - roi.lt.y + 1};
        }
    }
}

// Convert inputs to correponding bit depth specified by user
inline void convert_input_bitdepth(void *input, void *input_second, Rpp8u *inputu8, Rpp8u *inputu8Second, int inputBitDepth, Rpp64u ioBufferSize, Rpp64u inputBufferSize, RpptDescPtr srcDescPtr, bool dualInputCase, Rpp32f conversionFactor)
{
    if (inputBitDepth == 0 || inputBitDepth == 3 || inputBitDepth == 4)
    {
        memcpy(input, inputu8, inputBufferSize);
        if(dualInputCase)
            memcpy(input_second, inputu8Second, inputBufferSize);
    }
    else if (inputBitDepth == 1)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp16f *inputf16Temp, *inputf16SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputf16Temp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(input) + srcDescPtr->offsetInBytes);
        for (int i = 0; i < ioBufferSize; i++)
            *inputf16Temp++ = static_cast<Rpp16f>((static_cast<float>(*inputTemp++)) * conversionFactor);

        if(dualInputCase)
        {
            inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
            inputf16SecondTemp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(input_second) + srcDescPtr->offsetInBytes);
            for (int i = 0; i < ioBufferSize; i++)
                *inputf16SecondTemp++ = static_cast<Rpp16f>((static_cast<float>(*inputSecondTemp++)) * conversionFactor);
        }
    }
    else if (inputBitDepth == 2)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp32f *inputf32Temp, *inputf32SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputf32Temp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(input) + srcDescPtr->offsetInBytes);
        for (int i = 0; i < ioBufferSize; i++)
            *inputf32Temp++ = (static_cast<Rpp32f>(*inputTemp++)) * conversionFactor;

        if(dualInputCase)
        {
            inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
            inputf32SecondTemp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(input_second) + srcDescPtr->offsetInBytes);
            for (int i = 0; i < ioBufferSize; i++)
                *inputf32SecondTemp++ = (static_cast<Rpp32f>(*inputSecondTemp++)) * conversionFactor;
        }
    }
    else if (inputBitDepth == 5)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp8s *inputi8Temp, *inputi8SecondTemp;

        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputi8Temp = static_cast<Rpp8s *>(input) + srcDescPtr->offsetInBytes;
        for (int i = 0; i < ioBufferSize; i++)
            *inputi8Temp++ = static_cast<Rpp8s>((static_cast<Rpp32s>(*inputTemp++)) - 128);

        if(dualInputCase)
        {
            inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
            inputi8SecondTemp = static_cast<Rpp8s *>(input_second) + srcDescPtr->offsetInBytes;
            for (int i = 0; i < ioBufferSize; i++)
                *inputi8SecondTemp++ = static_cast<Rpp8s>((static_cast<Rpp32s>(*inputSecondTemp++)) - 128);
        }
    }
}

// Reconvert other bit depths to 8u for output display purposes
inline void convert_output_bitdepth_to_u8(void *output, Rpp8u *outputu8, int inputBitDepth, Rpp64u oBufferSize, Rpp64u outputBufferSize, RpptDescPtr dstDescPtr, Rpp32f invConversionFactor)
{
    if (inputBitDepth == 0)
    {
        memcpy(outputu8, output, outputBufferSize);
    }
    else if ((inputBitDepth == 1) || (inputBitDepth == 3))
    {
        Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp16f *outputf16Temp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(output) + dstDescPtr->offsetInBytes);
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = static_cast<Rpp8u>(validate_pixel_range(static_cast<float>(*outputf16Temp) * invConversionFactor));
            outputf16Temp++;
            outputTemp++;
        }
    }
    else if ((inputBitDepth == 2) || (inputBitDepth == 4))
    {
        Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp32f *outputf32Temp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(output) + dstDescPtr->offsetInBytes);
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = static_cast<Rpp8u>(validate_pixel_range(*outputf32Temp * invConversionFactor));
            outputf32Temp++;
            outputTemp++;
        }
    }
    else if ((inputBitDepth == 5) || (inputBitDepth == 6))
    {
        Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp8s *outputi8Temp = static_cast<Rpp8s *>(output) + dstDescPtr->offsetInBytes;
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = static_cast<Rpp8u>(validate_pixel_range((static_cast<Rpp32s>(*outputi8Temp) + 128)));
            outputi8Temp++;
            outputTemp++;
        }
    }
}

// updates dstImg sizes
inline void update_dst_sizes_with_roi(RpptROI *roiTensorPtrSrc, RpptImagePatchPtr dstImageSize, RpptRoiType roiType, int batchSize)
{
    if(roiType == RpptRoiType::XYWH)
    {
        for (int i = 0; i < batchSize; i++)
        {
            dstImageSize[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth;
            dstImageSize[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight;
        }
    }
    else if(roiType == RpptRoiType::LTRB)
    {
        for (int i = 0; i < batchSize; i++)
        {
            dstImageSize[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
            dstImageSize[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
        }
    }
}

// converts image data from PLN3 to PKD3
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

// converts image data from PKD3 to PLN3
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

// Opens a folder and recursively search for .jpg files
void open_folder(const string& folderPath, vector<string>& imageNames, vector<string>& imageNamesPath)
{
    auto src_dir = opendir(folderPath.c_str());
    struct dirent* entity;
    std::string fileName = " ";

    if (src_dir == nullptr)
        std::cerr << "\n ERROR: Failed opening the directory at " <<folderPath;

    while((entity = readdir(src_dir)) != nullptr)
    {
        string entry_name(entity->d_name);
        if (entry_name == "." || entry_name == "..")
            continue;
        fileName = entity->d_name;
        std::string filePath = folderPath;
        filePath.append("/");
        filePath.append(entity->d_name);
        fs::path pathObj(filePath);
        if(fs::exists(pathObj) && fs::is_directory(pathObj))
            open_folder(filePath, imageNames, imageNamesPath);

        if (fileName.size() > 4 && fileName.substr(fileName.size() - 4) == ".jpg")
        {
            imageNamesPath.push_back(filePath);
            imageNames.push_back(entity->d_name);
        }
    }
    if(imageNames.empty())
        std::cerr << "\n Did not load any file from " << folderPath;

    closedir(src_dir);
}

// Searches for .jpg files in input folders
void search_jpg_files(const string& folder_path, vector<string>& imageNames, vector<string>& imageNamesPath)
{
    vector<string> entry_list;
    string full_path = folder_path;
    auto sub_dir = opendir(folder_path.c_str());
    if (!sub_dir)
    {
        std::cerr << "ERROR: Failed opening the directory at "<< folder_path << std::endl;
        exit(0);
    }

    struct dirent* entity;
    while ((entity = readdir(sub_dir)) != nullptr)
    {
        string entry_name(entity->d_name);
        if (entry_name == "." || entry_name == "..")
            continue;
        entry_list.push_back(entry_name);
    }
    closedir(sub_dir);
    sort(entry_list.begin(), entry_list.end());

    for (unsigned dir_count = 0; dir_count < entry_list.size(); ++dir_count)
    {
        string subfolder_path = full_path + "/" + entry_list[dir_count];
        fs::path pathObj(subfolder_path);
        if (fs::exists(pathObj) && fs::is_regular_file(pathObj))
        {
            // ignore files with extensions .tar, .zip, .7z
            auto file_extension_idx = subfolder_path.find_last_of(".");
            if (file_extension_idx != std::string::npos)
            {
                std::string file_extension = subfolder_path.substr(file_extension_idx+1);
                if ((file_extension == "tar") || (file_extension == "zip") || (file_extension == "7z") || (file_extension == "rar"))
                    continue;
            }
            if (entry_list[dir_count].size() > 4 && entry_list[dir_count].substr(entry_list[dir_count].size() - 4) == ".jpg")
            {
                imageNames.push_back(entry_list[dir_count]);
                imageNamesPath.push_back(subfolder_path);
            }
        }
        else if (fs::exists(pathObj) && fs::is_directory(pathObj))
            open_folder(subfolder_path, imageNames, imageNamesPath);
    }
}

// Read a batch of images using the OpenCV library
inline void read_image_batch_opencv(Rpp8u *input, RpptDescPtr descPtr, vector<string>::const_iterator imagesNamesStart)
{
    for(int i = 0; i < descPtr->n; i++)
    {
        Rpp8u *inputTemp = input + (i * descPtr->strides.nStride);
        string inputImagePath = *(imagesNamesStart + i);
        Mat image, imageBgr;
        if (descPtr->c == 3)
        {
            imageBgr = imread(inputImagePath, 1);
            cvtColor(imageBgr, image, COLOR_BGR2RGB);
        }
        else if (descPtr->c == 1)
            image = imread(inputImagePath, 0);

        int width = image.cols;
        int height = image.rows;
        Rpp32u elementsInRow = width * descPtr->c;
        Rpp8u *inputImage = image.data;
        for (int j = 0; j < height; j++)
        {
            memcpy(inputTemp, inputImage, elementsInRow * sizeof(Rpp8u));
            inputImage += elementsInRow;
            inputTemp += descPtr->w * descPtr->c;;
        }
    }
}

// Read a batch of images using the turboJpeg decoder
inline void read_image_batch_turbojpeg(Rpp8u *input, RpptDescPtr descPtr, vector<string>::const_iterator imagesNamesStart)
{
    tjhandle m_jpegDecompressor = tjInitDecompress();

    // Loop through the input images
    for (int i = 0; i < descPtr->n; i++)
    {
        // Read the JPEG compressed data from a file
        std::string inputImagePath = *(imagesNamesStart + i);
        FILE* fp = fopen(inputImagePath.c_str(), "rb");
        if(!fp)
            std::cerr << "\n unable to open file : "<<inputImagePath;
        fseek(fp, 0, SEEK_END);
        long jpegSize = ftell(fp);
        rewind(fp);
        unsigned char* jpegBuf = (unsigned char*)malloc(jpegSize);
        fread(jpegBuf, 1, jpegSize, fp);
        fclose(fp);

        // Decompress the JPEG data into an RGB image buffer
        int width, height, subsamp, color_space;
        if(tjDecompressHeader2(m_jpegDecompressor, jpegBuf, jpegSize, &width, &height, &color_space) != 0)
            std::cerr << "\n Jpeg image decode failed in tjDecompressHeader2";
        Rpp8u* rgbBuf;
        int elementsInRow;
        if(descPtr->c == 3)
        {
            elementsInRow = width * descPtr->c;
            rgbBuf= (Rpp8u*)malloc(width * height * 3);
            if(tjDecompress2(m_jpegDecompressor, jpegBuf, jpegSize, rgbBuf, width, width * 3, height, TJPF_RGB, TJFLAG_ACCURATEDCT) != 0)
                std::cerr << "\n Jpeg image decode failed ";
        }
        else
        {
            elementsInRow = width;
            rgbBuf= (Rpp8u*)malloc(width * height);
            if(tjDecompress2(m_jpegDecompressor, jpegBuf, jpegSize, rgbBuf, width, width, height, TJPF_GRAY, 0) != 0)
                std::cerr << "\n Jpeg image decode failed ";
        }
        // Copy the decompressed image buffer to the RPP input buffer
        Rpp8u *inputTemp = input + descPtr->offsetInBytes + (i * descPtr->strides.nStride);
        for (int j = 0; j < height; j++)
        {
            memcpy(inputTemp, rgbBuf + j * elementsInRow, elementsInRow * sizeof(Rpp8u));
            inputTemp += descPtr->w * descPtr->c;
        }
        // Clean up
        free(jpegBuf);
        free(rgbBuf);
    }

    // Clean up
    tjDestroy(m_jpegDecompressor);
}

// Write a batch of images using the OpenCV library
inline void write_image_batch_opencv(string outputFolder, Rpp8u *output, RpptDescPtr dstDescPtr, vector<string>::const_iterator imagesNamesStart, RpptImagePatch *dstImgSizes, int maxImageDump)
{
    // create output folder
    mkdir(outputFolder.c_str(), 0700);
    outputFolder += "/";
    static int cnt = 1;
    static int imageCnt = 0;

    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
    Rpp8u *offsettedOutput = output + dstDescPtr->offsetInBytes;
    for (int j = 0; (j < dstDescPtr->n) && (imageCnt < maxImageDump) ; j++, imageCnt++)
    {
        Rpp32u height = dstImgSizes[j].height;
        Rpp32u width = dstImgSizes[j].width;
        Rpp32u elementsInRow = width * dstDescPtr->c;
        Rpp32u outputSize = height * width * dstDescPtr->c;
        Rpp8u *tempOutput = (Rpp8u *)calloc(outputSize, sizeof(Rpp8u));
        Rpp8u *tempOutputRow = tempOutput;
        Rpp8u *outputRow = offsettedOutput + j * dstDescPtr->strides.nStride;
        for (int k = 0; k < height; k++)
        {
            memcpy(tempOutputRow, outputRow, elementsInRow * sizeof(Rpp8u));
            tempOutputRow += elementsInRow;
            outputRow += elementsInRowMax;
        }
        string outputImagePath = outputFolder + *(imagesNamesStart + j);
        Mat matOutputImage, matOutputImageRgb;
        if (dstDescPtr->c == 1)
            matOutputImage = Mat(height, width, CV_8UC1, tempOutput);
        else if (dstDescPtr->c == 2)
            matOutputImage = Mat(height, width, CV_8UC2, tempOutput);
        else if (dstDescPtr->c == 3)
        {
            matOutputImageRgb = Mat(height, width, CV_8UC3, tempOutput);
            cvtColor(matOutputImageRgb, matOutputImage, COLOR_RGB2BGR);
        }

        fs::path pathObj(outputImagePath);
        if (fs::exists(pathObj))
        {
            std::string outPath = outputImagePath.substr(0, outputImagePath.find_last_of('.')) + "_" + to_string(cnt) + outputImagePath.substr(outputImagePath.find_last_of('.'));
            imwrite(outPath, matOutputImage);
            cnt++;
        }
        else
            imwrite(outputImagePath, matOutputImage);
        free(tempOutput);
    }
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

// compares the output of PKD3-PKD3 variants
void compare_outputs_pkd(Rpp8u* output, Rpp8u* refOutput, RpptDescPtr dstDescPtr, RpptImagePatch *dstImgSizes, int refOutputHeight, int refOutputWidth, int refOutputSize, int &fileMatch)
{
    Rpp8u *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef;
    for(int imageCnt = 0; imageCnt < dstDescPtr->n; imageCnt++)
    {
        outputTemp = output + imageCnt * dstDescPtr->strides.nStride;
        outputTempRef = refOutput + imageCnt * refOutputSize;
        int height = dstImgSizes[imageCnt].height;
        int width = dstImgSizes[imageCnt].width * dstDescPtr->c;;
        int matched_idx = 0;
        int refOutputHstride = refOutputWidth * dstDescPtr->c;

        for(int i = 0; i < height; i++)
        {
            rowTemp = outputTemp + i * dstDescPtr->strides.hStride;
            rowTempRef = outputTempRef + i * refOutputHstride;
            for(int j = 0; j < width; j++)
            {
                outVal = rowTemp + j;
                outRefVal = rowTempRef + j;
                int diff = abs(*outVal - *outRefVal);
                if(diff <= CUTOFF)
                    matched_idx++;
            }
        }
        if(matched_idx == (height * width) && matched_idx !=0)
            fileMatch++;
    }
}

// compares the output of PLN3-PLN3 and PLN1-PLN1 variants
void compare_outputs_pln(Rpp8u* output, Rpp8u* refOutput, RpptDescPtr dstDescPtr, RpptImagePatch *dstImgSizes, int refOutputHeight, int refOutputWidth, int refOutputSize, int &fileMatch)
{
    Rpp8u *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef, *outputTempChn, *outputTempRefChn;
    for(int imageCnt = 0; imageCnt < dstDescPtr->n; imageCnt++)
    {
        outputTemp = output + imageCnt * dstDescPtr->strides.nStride;
        outputTempRef = refOutput + imageCnt * refOutputSize;
        int height = dstImgSizes[imageCnt].height;
        int width = dstImgSizes[imageCnt].width;
        int matched_idx = 0;
        int refOutputHstride = refOutputWidth;
        int refOutputCstride = refOutputHeight * refOutputWidth;

        for(int c = 0; c < dstDescPtr->c; c++)
        {
            outputTempChn = outputTemp + c * dstDescPtr->strides.cStride;
            outputTempRefChn = outputTempRef + c * refOutputCstride;
            for(int i = 0; i < height; i++)
            {
                rowTemp = outputTempChn + i * dstDescPtr->strides.hStride;
                rowTempRef = outputTempRefChn + i * refOutputHstride;
                for(int j = 0; j < width; j++)
                {
                    outVal = rowTemp + j;
                    outRefVal = rowTempRef + j ;
                    int diff = abs(*outVal - *outRefVal);
                    if(diff <= CUTOFF)
                        matched_idx++;
                }
            }
        }
        if(matched_idx == (height * width * dstDescPtr->c) && matched_idx !=0)
            fileMatch++;
    }
}

template <typename T>
inline void compare_output(T* output, string funcName, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, RpptImagePatch *dstImgSizes, int noOfImages, string interpolationTypeName, string noiseTypeName, int testCase, string dst)
{
    string func = funcName;
    string refPath = get_current_dir_name();
    string pattern = "/build";
    string refFile = "";
    int refOutputWidth = ((GOLDEN_OUTPUT_MAX_WIDTH / 8) * 8) + 8;    // obtain next multiple of 8 after GOLDEN_OUTPUT_MAX_WIDTH
    int refOutputHeight = GOLDEN_OUTPUT_MAX_HEIGHT;
    int refOutputSize = refOutputHeight * refOutputWidth * dstDescPtr->c;

    remove_substring(refPath, pattern);
    string dataType[4] = {"_u8_", "_f16_", "_f32_", "_i8_"};

    if(srcDescPtr->dataType == dstDescPtr->dataType)
        func += dataType[srcDescPtr->dataType];
    else
    {
        func = func + dataType[srcDescPtr->dataType];
        func.resize(func.size() - 1);
        func += dataType[dstDescPtr->dataType];
    }

    if(dstDescPtr->layout == RpptLayout::NHWC)
        func += "Tensor_PKD3";
    else
    {
        if (dstDescPtr->c == 3)
            func += "Tensor_PLN3";
        else
            func += "Tensor_PLN1";
    }
    if(testCase == 21 ||testCase == 23 || testCase == 24)
        refFile = refPath + "/../REFERENCE_OUTPUT/" + funcName + "/"+ func + "_interpolationType" + interpolationTypeName + ".csv";
    else if(testCase == 8)
        refFile = refPath + "/../REFERENCE_OUTPUT/" + funcName + "/"+ func + "_noiseType" + noiseTypeName + ".csv";
    else
        refFile = refPath + "/../REFERENCE_OUTPUT/" + funcName + "/"+ func + ".csv";

    ifstream file(refFile);
    Rpp8u *refOutput;
    refOutput = (Rpp8u *)malloc(dstDescPtr->n * refOutputSize * sizeof(Rpp8u));
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

    int fileMatch = 0;
    if(dstDescPtr->layout == RpptLayout::NHWC)
        compare_outputs_pkd(output, refOutput, dstDescPtr, dstImgSizes, refOutputHeight, refOutputWidth, refOutputSize, fileMatch);
    else
        compare_outputs_pln(output, refOutput, dstDescPtr, dstImgSizes, refOutputHeight, refOutputWidth, refOutputSize, fileMatch);

    std::cerr << std::endl << "Results for " << func << " :" << std::endl;
    std::string status = func + ": ";
    if(fileMatch == dstDescPtr->n)
    {
        std::cerr << "PASSED!" << std::endl;
        status += "PASSED";
    }
    else
    {
        std::cerr << "FAILED! " << fileMatch << "/" << dstDescPtr->n << " outputs are matching with reference outputs" << std::endl;
        status += "FAILED";
    }

    // Append the QA results to file
    std::string qaResultsPath = dst + "/QA_results.txt";
    std:: ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open())
    {
        qaResults << status << std::endl;
        qaResults.close();
    }
}

inline void compare_reduction_output(Rpp64u* output, string funcName, RpptDescPtr srcDescPtr, int testCase, string dst)
{
    string func = funcName;
    string refPath = get_current_dir_name();
    string pattern = "/build";
    string refFile = "";

    remove_substring(refPath, pattern);
    string dataType[4] = {"_u8_", "_f16_", "_f32_", "_i8_"};

    func += dataType[srcDescPtr->dataType];

    if(srcDescPtr->layout == RpptLayout::NHWC)
        func += "Tensor_PKD3";
    else
    {
        if (srcDescPtr->c == 3)
            func += "Tensor_PLN3";
        else
            func += "Tensor_PLN1";
    }

    refFile = refPath + "/../REFERENCE_OUTPUT/" + funcName + "/"+ func + ".csv";

    ifstream file(refFile);
    Rpp64u *refOutput;
    refOutput = (Rpp64u *)calloc(srcDescPtr->n * 4, sizeof(Rpp64u));
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

    int fileMatch = 0;
    int matched_values = 0;
    if(srcDescPtr->c == 1)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            int diff = output[i] - refOutput[i];
            if(diff <= CUTOFF)
                fileMatch++;
        }
    }
    else
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            matched_values = 0;
            for(int j = 0; j < 4; j++)
            {
                int diff = output[(i * 4) + j] - refOutput[(i * 4) + j];
                if(diff <= CUTOFF)
                    matched_values++;
            }
            if(matched_values == 4)
                fileMatch++;
        }
    }

    std::cerr << std::endl << "Results for " << func << " :" << std::endl;
    std::string status = func + ": ";
    if(fileMatch == srcDescPtr->n)
    {
        std::cerr << "PASSED!" << std::endl;
        status += "PASSED";
    }
    else
    {
        std::cerr << "FAILED! " << fileMatch << "/" << srcDescPtr->n << " outputs are matching with reference outputs" << std::endl;
        status += "FAILED";
    }

    // Append the QA results to file
    std::string qaResultsPath = dst + "/QA_results.txt";
    std:: ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open())
    {
        qaResults << status << std::endl;
        qaResults.close();
    }
}
