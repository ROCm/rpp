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
#include "rpp.h"
#include "../rpp_test_suite_common.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <hip/hip_fp16.h>
#include <fstream>

#define DEBUG_MODE 0

typedef half Rpp16f;

using namespace cv;
using namespace std;

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

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 13;

    char *src = argv[1];
    char *srcSecond = argv[2];
    string dst = argv[3];

    int inputBitDepth = atoi(argv[4]);
    unsigned int outputFormatToggle = atoi(argv[5]);
    int testCase = atoi(argv[6]);
    int numIterations = atoi(argv[8]);
    int testType = atoi(argv[9]);     // 0 for unit and 1 for performance test
    int layoutType = atoi(argv[10]); // 0 for pkd3 / 1 for pln3 / 2 for pln1
    int qaFlag = atoi(argv[12]);
    int decoderType = atoi(argv[13]);

    bool additionalParamCase = (testCase == 8 || testCase == 21 || testCase == 23|| testCase == 24 || testCase == 40 || testCase == 41 || testCase == 49);
    bool kernelSizeCase = (testCase == 40 || testCase == 41 || testCase == 49);
    bool interpolationTypeCase = (testCase == 21 || testCase == 23 || testCase == 24);
    bool noiseTypeCase = (testCase == 8);
    bool pln1OutTypeCase = (testCase == 86);
    bool reductionTypeCase = (testCase == 88 || testCase == 89 || testCase == 90);

    unsigned int verbosity = atoi(argv[11]);
    unsigned int additionalParam = additionalParamCase ? atoi(argv[7]) : 1;

    if (verbosity == 1)
    {
        printf("\nInputs for this test case are:");
        printf("\nsrc1 = %s", argv[1]);
        printf("\nsrc2 = %s", argv[2]);
        if (testType == 0)
            printf("\ndst = %s", argv[3]);
        printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[4]);
        printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[5]);
        printf("\ncase number (0:86) = %s", argv[6]);
        printf("\nnumber of times to run = %s", argv[8]);
        printf("\ntest type - (0 = unit tests / 1 = performance tests) = %s", argv[9]);
        printf("\nlayout type - (0 = PKD3/ 1 = PLN3/ 2 = PLN1) = %s", argv[10]);
        printf("\nqa mode - 0/1 = %s", argv[12]);
        printf("\ndecoder type - (0 = TurboJPEG / 1 = OpenCV) = %s", argv[13]);
    }

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:86> <number of iterations > 0> <verbosity = 0/1>>\n");
        return -1;
    }

    if (layoutType == 2)
    {
        if(testCase == 36 || testCase == 31 || testCase == 86)
        {
            printf("\ncase %d does not exist for PLN1 layout\n", testCase);
            return -1;
        }
        else if (outputFormatToggle != 0)
        {
            printf("\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = 0\n");
            return -1;
        }
    }

    // Get function name
    string funcName = augmentationMap[testCase];
    if (funcName.empty())
    {
        if (testType == 0)
            printf("\ncase %d is not supported\n", testCase);

        return -1;
    }

    // Determine the number of input channels based on the specified layout type
    int inputChannels = set_input_channels(layoutType);

    // Determine the type of function to be used based on the specified layout type
    string funcType = set_function_type(layoutType, pln1OutTypeCase, outputFormatToggle, "HIP");

    // Initialize tensor descriptors
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    // Set src/dst layout types in tensor descriptors
    set_descriptor_layout( srcDescPtr, dstDescPtr, layoutType, pln1OutTypeCase, outputFormatToggle);

    // Set src/dst data types in tensor descriptors
    set_descriptor_data_type(inputBitDepth, funcName, srcDescPtr, dstDescPtr);

    // Other initializations
    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxHeight = 0, maxWidth = 0;
    int maxDstHeight = 0, maxDstWidth = 0;
    Rpp64u count = 0;
    Rpp64u ioBufferSize = 0;
    Rpp64u oBufferSize = 0;
    static int noOfImages = 0;
    Mat image, imageSecond;

    // String ops on input path
    string inputPath = src;
    inputPath += "/";
    string inputPathSecond = srcSecond;
    inputPathSecond += "/";

    string func = funcName;
    func += funcType;

    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    std::string interpolationTypeName = "";
    if (kernelSizeCase)
    {
        char additionalParam_char[2];
        std::sprintf(additionalParam_char, "%u", additionalParam);
        func += "_kSize";
        func += additionalParam_char;
    }
    else if (interpolationTypeCase)
    {
        interpolationTypeName = get_interpolation_type(additionalParam, interpolationType);
        func += "_interpolationType";
        func += interpolationTypeName.c_str();
    }
    else if (noiseTypeCase)
    {
        std::string noiseTypeName;
        noiseTypeName = get_noise_type(additionalParam);
        func += "_noiseType";
        func += noiseTypeName.c_str();
    }

    if(!qaFlag)
    {
        dst += "/";
        dst += func;
    }

    // Get number of images and image Names
    struct dirent *de;
    DIR *dr = opendir(src);
    vector<string> imageNames;
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfImages += 1;
        imageNames.push_back(de->d_name);
    }
    closedir(dr);

    if(!noOfImages)
    {
        std::cerr<<"Not able to find any images in the folder specified. Please check the input path";
        exit(0);
    }

    if(qaFlag)
        sort(imageNames.begin(), imageNames.end());

    // Initialize ROI tensors for src/dst
    RpptROI *roiTensorPtrSrc, *roiTensorPtrDst;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    hipHostMalloc(&roiTensorPtrDst, noOfImages * sizeof(RpptROI));

    // Initialize the ImagePatch for dst
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));

    // Set ROI tensors types for src/dst
    RpptRoiType roiTypeSrc, roiTypeDst;
    roiTypeSrc = RpptRoiType::XYWH;
    roiTypeDst = RpptRoiType::XYWH;

    // Set maxHeight, maxWidth and ROIs for src/dst
    const int images = noOfImages;

    for(int i = 0; i < imageNames.size(); i++)
    {
        string temp = inputPath;
        temp += imageNames[i];
        if (layoutType == 0 || layoutType == 1)
            image = imread(temp, 1);
        else
            image = imread(temp, 0);

        roiTensorPtrSrc[i].xywhROI = {0, 0, image.cols, image.rows};
        roiTensorPtrDst[i].xywhROI = {0, 0, image.cols, image.rows};
        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth;
        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight;

        maxHeight = std::max(maxHeight, roiTensorPtrSrc[i].xywhROI.roiHeight);
        maxWidth = std::max(maxWidth, roiTensorPtrSrc[i].xywhROI.roiWidth);
        maxDstHeight = std::max(maxDstHeight, roiTensorPtrDst[i].xywhROI.roiHeight);
        maxDstWidth = std::max(maxDstWidth, roiTensorPtrDst[i].xywhROI.roiWidth);

        count++;
    }

    // Check if any of maxWidth and maxHeight is less than or equal to 0
    if(maxHeight <= 0 || maxWidth <= 0)
    {
        std::cerr<<"Unable to read images properly.Please check the input path of the files specified";
        exit(0);
    }

    Rpp32u outputChannels = inputChannels;
    if(pln1OutTypeCase)
        outputChannels = 1;
    Rpp32u offsetInBytes = 0;

    // Set numDims, offset, n/c/h/w values, strides for src/dst
    set_descriptor_dims_and_strides(srcDescPtr, noOfImages, maxHeight, maxWidth, inputChannels, offsetInBytes);
    set_descriptor_dims_and_strides(dstDescPtr, noOfImages, maxDstHeight, maxDstWidth, outputChannels, offsetInBytes);

    // Set buffer sizes in pixels for src/dst
    ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
    oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

    // Set buffer sizes in bytes for src/dst (including offsets)
    Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
    Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
    Rpp64u inputBufferSize = ioBufferSize * get_size_of_data_type(srcDescPtr->dataType) + srcDescPtr->offsetInBytes;
    Rpp64u outputBufferSize = oBufferSize * get_size_of_data_type(dstDescPtr->dataType) + dstDescPtr->offsetInBytes;

    // Initialize 8u host buffers for src/dst
    Rpp8u *inputu8 = static_cast<Rpp8u *>(calloc(ioBufferSizeInBytes_u8, 1));
    Rpp8u *inputu8Second = static_cast<Rpp8u *>(calloc(ioBufferSizeInBytes_u8, 1));
    Rpp8u *outputu8 = static_cast<Rpp8u *>(calloc(oBufferSizeInBytes_u8, 1));
    if (testCase == 40) memset(inputu8, 0xFF, ioBufferSizeInBytes_u8);

    Rpp8u *offsettedInput, *offsettedInputSecond;
    offsettedInput = inputu8 + srcDescPtr->offsetInBytes;
    offsettedInputSecond = inputu8Second + srcDescPtr->offsetInBytes;
    string imageNamesPath[images];
    string imageNamesPathSecond[images];
    for(int i = 0; i < images; i++)
    {
        imageNamesPath[i] = inputPath + "/" + imageNames[i];
        imageNamesPathSecond[i] = inputPathSecond + "/" + imageNames[i];
    }

    // Read images
    if(decoderType == 0)
    {
        read_image_batch_turbojpeg(inputu8, srcDescPtr, imageNamesPath);
        read_image_batch_turbojpeg(inputu8Second, srcDescPtr, imageNamesPathSecond);
    }
    else
    {
        read_image_batch_opencv(inputu8, srcDescPtr, imageNamesPath);
        read_image_batch_opencv(inputu8Second, srcDescPtr, imageNamesPathSecond);
    }

    // if the input layout requested is PLN3, convert PKD3 inputs to PLN3 for first and second input batch
    if (layoutType == 1)
    {
        convert_pkd3_to_pln3(inputu8, srcDescPtr);
        convert_pkd3_to_pln3(inputu8Second, srcDescPtr);
    }

    // Factors to convert U8 data to F32, F16 data to 0-1 range and reconvert them back to 0 -255 range
    Rpp32f conversionFactor = 1.0f / 255.0;
    if(testCase == 38)
        conversionFactor = 1.0;
    Rpp32f invConversionFactor = 1.0f / conversionFactor;

    void *input, *input_second, *output;
    void *d_input, *d_input_second, *d_output;
    input = static_cast<Rpp8u *>(calloc(inputBufferSize, 1));
    input_second = static_cast<Rpp8u *>(calloc(inputBufferSize, 1));
    output = static_cast<Rpp8u *>(calloc(outputBufferSize, 1));

    // Convert inputs to correponding bit depth specified by user
    if (inputBitDepth == 0 || inputBitDepth == 3 || inputBitDepth == 4)
    {
        memcpy(input, inputu8, inputBufferSize);
        memcpy(input_second, inputu8Second, inputBufferSize);
    }
    else if (inputBitDepth == 1)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp16f *inputf16Temp, *inputf16SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
        inputf16Temp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(input) + srcDescPtr->offsetInBytes);
        inputf16SecondTemp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(input_second) + srcDescPtr->offsetInBytes);

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf16Temp++ = static_cast<Rpp16f>((static_cast<float>(*inputTemp++)) * conversionFactor);
            *inputf16SecondTemp++ = static_cast<Rpp16f>((static_cast<float>(*inputSecondTemp++)) * conversionFactor);
        }
    }
    else if (inputBitDepth == 2)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp32f *inputf32Temp, *inputf32SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
        inputf32Temp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(input) + srcDescPtr->offsetInBytes);
        inputf32SecondTemp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(input_second) + srcDescPtr->offsetInBytes);

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf32Temp++ = (static_cast<Rpp32f>(*inputTemp++)) * conversionFactor;
            *inputf32SecondTemp++ = (static_cast<Rpp32f>(*inputSecondTemp++)) * conversionFactor;
        }
    }
    else if (inputBitDepth == 5)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp8s *inputi8Temp, *inputi8SecondTemp;

        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
        inputi8Temp = static_cast<Rpp8s *>(input) + srcDescPtr->offsetInBytes;
        inputi8SecondTemp = static_cast<Rpp8s *>(input_second) + srcDescPtr->offsetInBytes;

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputi8Temp++ = static_cast<Rpp8s>((static_cast<Rpp32s>(*inputTemp++)) - 128);
            *inputi8SecondTemp++ = static_cast<Rpp8s>((static_cast<Rpp32s>(*inputSecondTemp++)) - 128);
        }
    }

    // Initialize buffers for any reductionType functions
    Rpp32f *reductionFuncResult;
    Rpp32u reductionFuncResultArrLength = srcDescPtr->n * 4;
    reductionFuncResult = (Rpp32f *)calloc(reductionFuncResultArrLength, sizeof(Rpp32f));
    void *d_reductionFuncResult;
    hipMalloc(&d_reductionFuncResult, reductionFuncResultArrLength * sizeof(Rpp32f));
    hipMemcpy(d_reductionFuncResult, reductionFuncResult, reductionFuncResultArrLength * sizeof(Rpp32f), hipMemcpyHostToDevice);

    // Allocate hip memory for src/dst and copy decoded inputs to hip buffers
    hipMalloc(&d_input, inputBufferSize);
    hipMalloc(&d_input_second, inputBufferSize);
    hipMalloc(&d_output, outputBufferSize);
    hipMemcpy(d_input, input, inputBufferSize, hipMemcpyHostToDevice);
    hipMemcpy(d_input_second, input_second, inputBufferSize, hipMemcpyHostToDevice);
    hipMemcpy(d_output, output, outputBufferSize, hipMemcpyHostToDevice);

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double wallTime;
    string testCaseName;

    // Uncomment to run test case with an xywhROI override
    // roi.xywhROI = {0, 0, 25, 25};
    // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
    // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

    // Uncomment to run test case with an ltrbROI override
    // roiTypeSrc = RpptRoiType::LTRB;
    // roi.ltrbROI = {10, 10, 40, 40};
    // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
    // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

    // case-wise RPP API and measure time script for Unit and Performance test
    printf("\nRunning %s %d times (each time with a batch size of %d images) and computing mean statistics...", func.c_str(), numIterations, noOfImages);
    for (int perfRunCount = 0; perfRunCount < numIterations; perfRunCount++)
    {
        double startWallTime, endWallTime;
        switch (testCase)
        {
        case 0:
        {
            testCaseName = "brightness";

            Rpp32f alpha[images];
            Rpp32f beta[images];
            for (i = 0; i < images; i++)
            {
                alpha[i] = 1.75;
                beta[i] = 50;
            }

            startWallTime = omp_get_wtime();
            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                rppt_brightness_gpu(d_input, srcDescPtr, d_output, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
            else
                missingFuncFlag = 1;

            break;
        }
        case 2:
        {
            testCaseName = "blend";

            Rpp32f alpha[images];
            for (i = 0; i < images; i++)
            {
                alpha[i] = 0.4;
            }

            startWallTime = omp_get_wtime();
            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                rppt_blend_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
            else
                missingFuncFlag = 1;

            break;
        }
        case 4:
        {
            testCaseName = "contrast";

            Rpp32f contrastFactor[images];
            Rpp32f contrastCenter[images];
            for (i = 0; i < images; i++)
            {
                contrastFactor[i] = 2.96;
                contrastCenter[i] = 128;
            }

            startWallTime = omp_get_wtime();
            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                rppt_contrast_gpu(d_input, srcDescPtr, d_output, dstDescPtr, contrastFactor, contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle);
            else
                missingFuncFlag = 1;

            break;
        }
        case 13:
        {
            testCaseName = "exposure";

            Rpp32f exposureFactor[images];
            for (i = 0; i < images; i++)
            {
                exposureFactor[i] = 1.4;
            }

            startWallTime = omp_get_wtime();
            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                rppt_exposure_gpu(d_input, srcDescPtr, d_output, dstDescPtr, exposureFactor, roiTensorPtrSrc, roiTypeSrc, handle);
            else
                missingFuncFlag = 1;

            break;
        }
        case 31:
        {
            testCaseName = "color_cast";

            RpptRGB rgbTensor[images];
            Rpp32f alphaTensor[images];

            for (i = 0; i < images; i++)
            {
                rgbTensor[i].R = 0;
                rgbTensor[i].G = 0;
                rgbTensor[i].B = 100;
                alphaTensor[i] = 0.5;
            }

            startWallTime = omp_get_wtime();
            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                rppt_color_cast_gpu(d_input, srcDescPtr, d_output, dstDescPtr, rgbTensor, alphaTensor, roiTensorPtrSrc, roiTypeSrc, handle);
            else
                missingFuncFlag = 1;

            break;
        }
        case 34:
        {
            testCaseName = "lut";

            Rpp32f *lutBuffer;
            hipHostMalloc(&lutBuffer, 65536 * sizeof(Rpp32f));
            hipMemset(lutBuffer, 0, 65536 * sizeof(Rpp32f));
            Rpp8u *lut8u = reinterpret_cast<Rpp8u *>(lutBuffer);
            Rpp16f *lut16f = reinterpret_cast<Rpp16f *>(lutBuffer);
            Rpp32f *lut32f = reinterpret_cast<Rpp32f *>(lutBuffer);
            Rpp8s *lut8s = reinterpret_cast<Rpp8s *>(lutBuffer);
            if (inputBitDepth == 0)
                for (j = 0; j < 256; j++)
                    lut8u[j] = (Rpp8u)(255 - j);
            else if (inputBitDepth == 3)
                for (j = 0; j < 256; j++)
                    lut16f[j] = (Rpp16f)((255 - j) * ONE_OVER_255);
            else if (inputBitDepth == 4)
                for (j = 0; j < 256; j++)
                    lut32f[j] = (Rpp32f)((255 - j) * ONE_OVER_255);
            else if (inputBitDepth == 5)
                for (j = 0; j < 256; j++)
                    lut8s[j] = (Rpp8s)(255 - j - 128);

            startWallTime = omp_get_wtime();
            if (inputBitDepth == 0)
                rppt_lut_gpu(d_input, srcDescPtr, d_output, dstDescPtr, lut8u, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (inputBitDepth == 3)
                rppt_lut_gpu(d_input, srcDescPtr, d_output, dstDescPtr, lut16f, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (inputBitDepth == 4)
                rppt_lut_gpu(d_input, srcDescPtr, d_output, dstDescPtr, lut32f, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (inputBitDepth == 5)
                rppt_lut_gpu(d_input, srcDescPtr, d_output, dstDescPtr, lut8s, roiTensorPtrSrc, roiTypeSrc, handle);
            else
                missingFuncFlag = 1;

            break;

            hipHostFree(lutBuffer);
        }
        case 36:
        {
            testCaseName = "color_twist";

            Rpp32f brightness[images];
            Rpp32f contrast[images];
            Rpp32f hue[images];
            Rpp32f saturation[images];
            for (i = 0; i < images; i++)
            {
                brightness[i] = 1.4;
                contrast[i] = 0.0;
                hue[i] = 60.0;
                saturation[i] = 1.9;
            }

            startWallTime = omp_get_wtime();
            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
            else
                missingFuncFlag = 1;

            break;
        }
        case 38:
        {
            testCaseName = "crop_mirror_normalize";
            Rpp32f multiplier[images * srcDescPtr->c];
            Rpp32f offset[images * srcDescPtr->c];
            Rpp32u mirror[images];
            if (srcDescPtr->c == 3)
            {
                Rpp32f meanParam[3] = { 60.0f, 80.0f, 100.0f };
                Rpp32f stdDevParam[3] = { 0.9f, 0.9f, 0.9f };
                Rpp32f offsetParam[3] = { - meanParam[0] / stdDevParam[0], - meanParam[1] / stdDevParam[1], - meanParam[2] / stdDevParam[2] };
                Rpp32f multiplierParam[3] = {  1.0f / stdDevParam[0], 1.0f / stdDevParam[1], 1.0f / stdDevParam[2] };

                for (i = 0, j = 0; i < images; i++, j += 3)
                {
                    multiplier[j] = multiplierParam[0];
                    offset[j] = offsetParam[0];
                    multiplier[j + 1] = multiplierParam[1];
                    offset[j + 1] = offsetParam[1];
                    multiplier[j + 2] = multiplierParam[2];
                    offset[j + 2] = offsetParam[2];
                    mirror[i] = 1;
                }
            }
            else if(srcDescPtr->c == 1)
            {
                Rpp32f meanParam = 100.0f;
                Rpp32f stdDevParam = 0.9f;
                Rpp32f offsetParam = - meanParam / stdDevParam;
                Rpp32f multiplierParam = 1.0f / stdDevParam;

                for (i = 0; i < images; i++)
                {
                    multiplier[i] = multiplierParam;
                    offset[i] = offsetParam;
                    mirror[i] = 1;
                }
            }

            for (i = 0; i < images; i++)
            {
                roiTensorPtrDst[i].xywhROI.xy.x = 10;
                roiTensorPtrDst[i].xywhROI.xy.y = 10;
                dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
            }

            startWallTime = omp_get_wtime();
            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 3 || inputBitDepth == 4 || inputBitDepth == 5)
                rppt_crop_mirror_normalize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, offset, multiplier, mirror, roiTensorPtrDst, roiTypeSrc, handle);
            else
                missingFuncFlag = 1;

            break;
        }
        case 89:
        {
            testCaseName = "image_min";

            if(outputFormatToggle == 1)
                missingFuncFlag = 1;

            if(srcDescPtr->c == 1)
                reductionFuncResultArrLength = srcDescPtr->n;

            startWallTime = omp_get_wtime();
            if (inputBitDepth == 0)
                rppt_image_min_gpu(d_input, srcDescPtr, d_reductionFuncResult, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
            else
                missingFuncFlag = 1;

            break;
        }
        case 90:
        {
            testCaseName = "image_max";

            if(outputFormatToggle == 1)
                missingFuncFlag = 1;

            if(srcDescPtr->c == 1)
                reductionFuncResultArrLength = srcDescPtr->n;

            startWallTime = omp_get_wtime();
            if (inputBitDepth == 0)
                rppt_image_max_gpu(d_input, srcDescPtr, d_reductionFuncResult, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
            else
                missingFuncFlag = 1;

            break;
        }
        default:
            missingFuncFlag = 1;
            break;
        }

        hipDeviceSynchronize();
        endWallTime = omp_get_wtime();
        wallTime = endWallTime - startWallTime;
        if (missingFuncFlag == 1)
        {
            printf("\nThe functionality %s doesn't yet exist in RPP\n", func.c_str());
            return -1;
        }

        maxWallTime = max(maxWallTime, wallTime);
        minWallTime = min(minWallTime, wallTime);
        avgWallTime += wallTime ;
    }

    wallTime *= 1000;
    maxWallTime *= 1000;
    minWallTime *= 1000;
    avgWallTime *= 1000;
    if (testType == 0)
    {
        cout << "\n\nGPU Backend Wall Time: " << wallTime <<" ms/batch"<< endl;

        // Display results for reduction functions
        if (reductionTypeCase)
        {
            printf("\nReduction result (Batch of n channel images produces n+1 results per image in batch): ");
            hipMemcpy(reductionFuncResult, d_reductionFuncResult, reductionFuncResultArrLength * sizeof(Rpp32f), hipMemcpyDeviceToHost);
            std::cerr<<"Reduction function length is:"<<reductionFuncResultArrLength<<std::endl;
            for (int i = 0; i < reductionFuncResultArrLength; i++)
                printf(" %0.3f\n", reductionFuncResult[i]);
        }

        // Reconvert other bit depths to 8u for output display purposes
        if (!reductionTypeCase)
        {
            if (inputBitDepth == 0)
            {
                hipMemcpy(output, d_output, outputBufferSize, hipMemcpyDeviceToHost);
                memcpy(outputu8, output, outputBufferSize);
            }
            else if ((inputBitDepth == 1) || (inputBitDepth == 3))
            {
                hipMemcpy(output, d_output, outputBufferSize, hipMemcpyDeviceToHost);
                Rpp8u *outputTemp;
                outputTemp = outputu8 + dstDescPtr->offsetInBytes;
                Rpp16f *outputf16Temp;
                outputf16Temp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(output) + dstDescPtr->offsetInBytes);
                for (int i = 0; i < oBufferSize; i++)
                {
                    *outputTemp = static_cast<Rpp8u>(validate_pixel_range(static_cast<float>(*outputf16Temp) * invConversionFactor));
                    outputf16Temp++;
                    outputTemp++;
                }
            }
            else if ((inputBitDepth == 2) || (inputBitDepth == 4))
            {
                hipMemcpy(output, d_output, outputBufferSize, hipMemcpyDeviceToHost);
                Rpp8u *outputTemp;
                outputTemp = outputu8 + dstDescPtr->offsetInBytes;
                Rpp32f *outputf32Temp;
                outputf32Temp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(output) + dstDescPtr->offsetInBytes);
                for (int i = 0; i < oBufferSize; i++)
                {
                    *outputTemp = static_cast<Rpp8u>(validate_pixel_range(*outputf32Temp * invConversionFactor));
                    outputf32Temp++;
                    outputTemp++;
                }
            }
            else if ((inputBitDepth == 5) || (inputBitDepth == 6))
            {
                hipMemcpy(output, d_output, outputBufferSize, hipMemcpyDeviceToHost);
                Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
                Rpp8s *outputi8Temp = static_cast<Rpp8s *>(output) + dstDescPtr->offsetInBytes;
                for (int i = 0; i < oBufferSize; i++)
                {
                    *outputTemp = static_cast<Rpp8u>(validate_pixel_range((static_cast<Rpp32s>(*outputi8Temp)) + 128));
                    outputi8Temp++;
                    outputTemp++;
                }
            }
        }

        // If DEBUG_MODE is set to 1 dump the outputs to csv files for debugging
        if(DEBUG_MODE)
        {
            std::ofstream refFile;
            refFile.open(func + ".csv");
            for (int i = 0; i < oBufferSize; i++)
                refFile << static_cast<int>(*(outputu8 + i)) << ",";
            refFile.close();
        }

        /*Compare the output of the function with golden outputs only if
          1.QA Flag is set
          2.input bit depth 0 (Input U8 && Output U8)
          3.source and destination layout are the same*/
        if(qaFlag && inputBitDepth == 0 && (srcDescPtr->layout == dstDescPtr->layout))
            compare_output<Rpp8u>(outputu8, testCaseName, srcDescPtr, dstDescPtr, dstImgSizes, noOfImages, interpolationTypeName, testCase, dst);

        // Calculate exact dstROI in XYWH format for OpenCV dump
        if (roiTypeSrc == RpptRoiType::LTRB)
            convert_roi(roiTensorPtrDst, RpptRoiType::XYWH, dstDescPtr->n);

        // Check if the ROI values for each input is within the bounds of the max buffer allocated
        RpptROI roiDefault;
        RpptROIPtr roiPtrDefault = &roiDefault;
        roiPtrDefault->xywhROI =  {0, 0, static_cast<Rpp32s>(dstDescPtr->w), static_cast<Rpp32s>(dstDescPtr->h)};
        for (int i = 0; i < dstDescPtr->n; i++)
        {
            roiTensorPtrDst[i].xywhROI.roiWidth = std::min(roiPtrDefault->xywhROI.roiWidth - roiTensorPtrDst[i].xywhROI.xy.x, roiTensorPtrDst[i].xywhROI.roiWidth);
            roiTensorPtrDst[i].xywhROI.roiHeight = std::min(roiPtrDefault->xywhROI.roiHeight - roiTensorPtrDst[i].xywhROI.xy.y, roiTensorPtrDst[i].xywhROI.roiHeight);
            roiTensorPtrDst[i].xywhROI.xy.x = std::max(roiPtrDefault->xywhROI.xy.x, roiTensorPtrDst[i].xywhROI.xy.x);
            roiTensorPtrDst[i].xywhROI.xy.y = std::max(roiPtrDefault->xywhROI.xy.y, roiTensorPtrDst[i].xywhROI.xy.y);
        }

        // Convert any PLN3 outputs to the corresponding PKD3 version for OpenCV dump
        if (layoutType == 0 || layoutType == 1)
        {
            if ((dstDescPtr->c == 3) && (dstDescPtr->layout == RpptLayout::NCHW))
                convert_pln3_to_pkd3(outputu8, dstDescPtr);
        }
        rppDestroyGPU(handle);

        // OpenCV dump (if testType is unit test and QA mode is not set)
        if(!qaFlag && (reductionTypeCase == 0))
            write_image_batch_opencv(dst, outputu8, dstDescPtr, imageNames, dstImgSizes);
    }
    else
    {
        // Display measured times
        avgWallTime /= numIterations;
        cout << fixed <<"\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime << endl;
    }

    // Free memory
    hipHostFree(roiTensorPtrSrc);
    hipHostFree(roiTensorPtrDst);
    hipHostFree(dstImgSizes);
    free(input);
    free(input_second);
    free(output);
    free(inputu8);
    free(inputu8Second);
    free(outputu8);
    free(reductionFuncResult);
    hipFree(d_input);
    hipFree(d_input_second);
    hipFree(d_output);
    hipFree(d_reductionFuncResult);
    return 0;
}
