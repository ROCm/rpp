/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <dirent.h>
#include <filesystem.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

#define DEBUG_MODE 0
#define MAX_IMAGE_DUMP 20
#define MAX_BATCH_SIZE 512
#define GOLDEN_OUTPUT_MAX_HEIGHT 150    // Golden outputs are generated with MAX_HEIGHT set to 150. Changing this constant will result in QA test failures
#define GOLDEN_OUTPUT_MAX_WIDTH 150     // Golden outputs are generated with MAX_WIDTH set to 150. Changing this constant will result in QA test failures
#define LENS_CORRECTION_GOLDEN_OUTPUT_MAX_HEIGHT 480    // Lens correction golden outputs are generated with MAX_HEIGHT set to 480. Changing this constant will result in QA test failures
#define LENS_CORRECTION_GOLDEN_OUTPUT_MAX_WIDTH 640     // Lens correction golden outputs are generated with MAX_WIDTH set to 640. Changing this constant will result in QA test failures

#define CHECK_RETURN_STATUS(x) do { \
    int retval = (x); \
    if (retval != 0) { \
        fprintf(stderr, "Runtime error: %s returned %d at %s:%d", #x, retval, __FILE__, __LINE__); \
        exit(-1); \
    } \
} while (0)

std::map<int, string> augmentationMap =
{
    {0, "brightness"},
    {1, "gamma_correction"},
    {2, "blend"},
    {4, "contrast"},
    {5, "pixelate"},
    {6, "jitter"},
    {8, "noise"},
    {13, "exposure"},
    {20, "flip"},
    {21, "resize"},
    {23, "rotate"},
    {24, "warp_afffine"},
    {26, "lens_correction"},
    {29, "water"},
    {30, "non_linear_blend"},
    {31, "color_cast"},
    {32, "erase"},
    {33, "crop_and_patch"},
    {34, "lut"},
    {35, "glitch"},
    {36, "color_twist"},
    {37, "crop"},
    {38, "crop_mirror_normalize"},
    {39, "resize_crop_mirror"},
    {45, "color_temperature"},
    {46, "vignette"},
    {49, "box_filter"},
    {54, "gaussian_filter"},
    {61, "magnitude"},
    {63, "phase"},
    {65, "bitwise_and"},
    {68, "bitwise_or"},
    {70, "copy"},
    {79, "remap"},
    {80, "resize_mirror_normalize"},
    {81, "color_jitter"},
    {82, "ricap"},
    {83, "gridmask"},
    {84, "spatter"},
    {85, "swap_channels"},
    {86, "color_to_greyscale"},
    {87, "tensor_sum"},
    {88, "tensor_min"},
    {89, "tensor_max"},
    {90, "tensor_mean"},
    {91, "tensor_stddev"},
    {92, "slice"},
    {93, "jpeg_compression_distortion"}
};

// Golden outputs for Tensor min Kernel
std::map<int, std::vector<Rpp8u>> TensorMinReferenceOutputs =
{
    {1, {1, 1, 7}},
    {3, {0, 0, 0, 0, 2, 0, 0, 0, 7, 9, 0, 0}}
};

// Golden outputs for Tensor max Kernel
std::map<int, std::vector<Rpp8u>> TensorMaxReferenceOutputs =
{
    {1, {239, 245, 255}},
    {3, {255, 240, 236, 255, 255, 242, 241, 255, 253, 255, 255, 255}}
};

// Golden outputs for Tensor sum Kernel
std::map<int, std::vector<uint64_t>> TensorSumReferenceOutputs =
{
    {1, {334225, 813471, 2631125}},
    {3, {348380, 340992, 262616, 951988, 1056552, 749506, 507441, 2313499, 2170646, 2732368, 3320699, 8223713}}
};

// Golden outputs for Tensor mean Kernel
std::map<int, std::vector<float>> TensorMeanReferenceOutputs =
{
    {1, {133.690, 81.347, 116.939}},
    {3, {139.352, 136.397, 105.046, 126.932, 105.655, 74.951, 50.744, 77.117, 96.473, 121.439, 147.587, 121.833}}
};

// Golden outputs for Tensor stddev Kernel
std::map<int, std::vector<float>> TensorStddevReferenceOutputs =
{
    {1, {49.583, 54.623, 47.649}},
    {3, {57.416, 47.901, 53.235, 55.220, 68.471, 55.735, 46.668, 61.880, 47.462, 49.039, 67.269, 59.130}}
};

template <typename T>
inline T validate_pixel_range(T pixel)
{
    pixel = (pixel < static_cast<Rpp32f>(0)) ? (static_cast<Rpp32f>(0)) : ((pixel < static_cast<Rpp32f>(255)) ? pixel : (static_cast<Rpp32f>(255)));
    return pixel;
}

// replicates the last image in a batch to fill the remaining images in a batch
void replicate_last_file_to_fill_batch(const string& lastFilePath, vector<string>& imageNamesPath, vector<string>& imageNames, const string& lastFileName, int noOfImages, int batchCount)
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
inline void set_max_dimensions(vector<string>imagePaths, int& maxHeight, int& maxWidth, int& imagesMixed)
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

        if((maxWidth && maxWidth != width) || (maxHeight && maxHeight != height))
            imagesMixed = 1;

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

// sets generic descriptor dimensions and strides of src/dst for slice functionality
inline void set_generic_descriptor_slice(RpptDescPtr srcDescPtr, RpptGenericDescPtr descriptorPtr3D, int batchSize)
{
    descriptorPtr3D->offsetInBytes = 0;
    descriptorPtr3D->dataType = srcDescPtr->dataType;
    descriptorPtr3D->layout = srcDescPtr->layout;
    if(srcDescPtr->c == 3)
    {
        descriptorPtr3D->numDims = 4;
        descriptorPtr3D->dims[0] = batchSize;
        if (srcDescPtr->layout == RpptLayout::NHWC)
        {
            descriptorPtr3D->dims[1] = srcDescPtr->h;
            descriptorPtr3D->dims[2] = srcDescPtr->w;
            descriptorPtr3D->dims[3] = srcDescPtr->c;
        }
        else
        {
            descriptorPtr3D->dims[1] = srcDescPtr->c;
            descriptorPtr3D->dims[2] = srcDescPtr->h;
            descriptorPtr3D->dims[3] = srcDescPtr->w;
        }
        descriptorPtr3D->strides[0] = descriptorPtr3D->dims[1] * descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3];
        descriptorPtr3D->strides[1] = descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3];
        descriptorPtr3D->strides[2] = descriptorPtr3D->dims[3];
    }
    else
    {
        descriptorPtr3D->numDims = 3;
        descriptorPtr3D->dims[0] = batchSize;
        descriptorPtr3D->dims[1] = srcDescPtr->h;
        descriptorPtr3D->dims[2] = srcDescPtr->w;
        descriptorPtr3D->strides[0] = descriptorPtr3D->dims[1] * descriptorPtr3D->dims[2];
        descriptorPtr3D->strides[1] = descriptorPtr3D->dims[2];
    }
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

// Opens a folder and recursively search for files with given extension
void open_folder(const string& folderPath, vector<string>& imageNames, vector<string>& imageNamesPath, string extension)
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
            open_folder(filePath, imageNames, imageNamesPath, extension);

        if (fileName.size() > 4 && fileName.substr(fileName.size() - 4) == extension)
        {
            imageNamesPath.push_back(filePath);
            imageNames.push_back(entity->d_name);
        }
    }
    if(imageNames.empty())
        std::cerr << "\n Did not load any file from " << folderPath;

    closedir(src_dir);
}

// Searches for files with the provided extensions in input folders
void search_files_recursive(const string& folder_path, vector<string>& imageNames, vector<string>& imageNamesPath, string extension)
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
            if (entry_list[dir_count].size() > 4 && entry_list[dir_count].substr(entry_list[dir_count].size() - 4) == extension)
            {
                imageNames.push_back(entry_list[dir_count]);
                imageNamesPath.push_back(subfolder_path);
            }
        }
        else if (fs::exists(pathObj) && fs::is_directory(pathObj))
            open_folder(subfolder_path, imageNames, imageNamesPath, extension);
    }
}

// replicates the last image in a batch to fill the remaining images in a batch
void replicate_last_file_to_fill_batch(const string& lastFilePath, vector<string>& imageNamesPath, vector<string>& imageNames, const string& lastFileName, int noOfImages, int batchCount)
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

template <typename T>
inline void read_bin_file(string refFile, T *binaryContent)
{
    FILE *fp;
    fp = fopen(refFile.c_str(), "rb");
    if(!fp)
    {
        std::cout << "\n unable to open file : "<<refFile;
        exit(0);
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    if (fsize == 0)
    {
        std::cout << "File is empty";
        exit(0);
    }

    fseek(fp, 0, SEEK_SET);
    fread(binaryContent, fsize, 1, fp);
    fclose(fp);
}