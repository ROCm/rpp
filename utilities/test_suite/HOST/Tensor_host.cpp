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
#include <half/half.hpp>
#include <fstream>

#define DEBUG_MODE 1
#define MAX_IMAGE_DUMP 20
#define MAX_HEIGHT 150
#define MAX_WIDTH 150

using namespace cv;
using namespace std;

using half_float::half;
typedef half Rpp16f;

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
    int batchSize = atoi(argv[14]);

    bool additionalParamCase = (testCase == 8 || testCase == 21 || testCase == 23 || testCase == 24);
    bool dualInputCase = (testCase == 2);
    bool interpolationTypeCase = (testCase == 21 || testCase == 23 || testCase == 24);
    bool noiseTypeCase = (testCase == 8);
    bool pln1OutTypeCase = (testCase == 86);
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
    string funcType = set_function_type(layoutType, pln1OutTypeCase, outputFormatToggle, "HOST");

    // Initialize tensor descriptors
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    // Set src/dst layout types in tensor descriptors
    set_descriptor_layout(srcDescPtr, dstDescPtr, layoutType, pln1OutTypeCase, outputFormatToggle);

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
    std::string noiseTypeName = "";

    if (interpolationTypeCase)
    {
        interpolationTypeName = get_interpolation_type(additionalParam, interpolationType);
        func += "_interpolationType";
        func += interpolationTypeName.c_str();
    }
    else if (noiseTypeCase)
    {
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
    vector<string> imageNames, imageNamesSecond, imageNamesPath, imageNamesPathSecond;
    search_jpg_files(src, imageNames, imageNamesPath);
    if(dualInputCase)
    {
        search_jpg_files(srcSecond, imageNamesSecond, imageNamesPathSecond);
        if(imageNames.size() != imageNamesSecond.size())
        {
            std::cerr <<" \n The number of images in the input folders must be the same.";
            exit(0);
        }
    }
    noOfImages = imageNames.size();

    if(noOfImages < batchSize)
    {
        replicate_last_image_to_fill_batch(imageNamesPath[noOfImages - 1], imageNamesPath, imageNames, imageNames[noOfImages - 1], noOfImages, batchSize);
        if(dualInputCase)
            replicate_last_image_to_fill_batch(imageNamesPathSecond[noOfImages - 1], imageNamesPathSecond, imageNamesSecond, imageNamesSecond[noOfImages - 1], noOfImages, batchSize);
    }

    if(!noOfImages)
    {
        std::cerr<<"Not able to find any images in the folder specified. Please check the input path";
        exit(0);
    }

    if(qaFlag || DEBUG_MODE)
    {
        sort(imageNames.begin(), imageNames.end());
        if(dualInputCase)
            sort(imageNamesSecond.begin(), imageNamesSecond.end());
    }

    // Check if any of maxWidth and maxHeight is less than or equal to 0
    if(MAX_HEIGHT <= 0 || MAX_WIDTH <= 0)
    {
        std::cerr<<"Unable to read images properly.Please check the input path of the files specified";
        exit(0);
    }

    // Initialize ROI tensors for src/dst
    RpptROI *roiTensorPtrSrc = static_cast<RpptROI *>(calloc(batchSize, sizeof(RpptROI)));
    RpptROI *roiTensorPtrDst = static_cast<RpptROI *>(calloc(batchSize, sizeof(RpptROI)));

    // Initialize the ImagePatch for dst
    RpptImagePatch *dstImgSizes = static_cast<RpptImagePatch *>(calloc(batchSize, sizeof(RpptImagePatch)));

    // Set ROI tensors types for src/dst
    RpptRoiType roiTypeSrc, roiTypeDst;
    roiTypeSrc = RpptRoiType::XYWH;
    roiTypeDst = RpptRoiType::XYWH;

    // Initialize roi that can be updated in case-wise augmentations if needed
    RpptROI roi;

    Rpp32u outputChannels = inputChannels;
    if(pln1OutTypeCase)
        outputChannels = 1;
    Rpp32u offsetInBytes = 0;

    // Set numDims, offset, n/c/h/w values, strides for src/dst
    set_descriptor_dims_and_strides(srcDescPtr, batchSize, MAX_HEIGHT, MAX_WIDTH, inputChannels, offsetInBytes);
    set_descriptor_dims_and_strides(dstDescPtr, batchSize, MAX_HEIGHT, MAX_WIDTH, outputChannels, offsetInBytes);

    // Factors to convert U8 data to F32, F16 data to 0-1 range and reconvert them back to 0 -255 range
    Rpp32f conversionFactor = 1.0f / 255.0;
    if(testCase == 38)
        conversionFactor = 1.0;
    Rpp32f invConversionFactor = 1.0f / conversionFactor;

    // Set buffer sizes in pixels for src/dst
    ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)batchSize;
    oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)batchSize;

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

    void *input, *input_second, *output;

    input = static_cast<Rpp8u *>(calloc(inputBufferSize, 1));
    input_second = static_cast<Rpp8u *>(calloc(inputBufferSize, 1));
    output = static_cast<Rpp8u *>(calloc(outputBufferSize, 1));

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, batchSize);

    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double cpuTime, wallTime;
    string testCaseName;

    // Uncomment to run test case with an xywhROI override
    // roi.xywhROI = {0, 0, 25, 25};
    // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, batchSize);
    // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, batchSize);

    // Uncomment to run test case with an ltrbROI override
    // roiTypeSrc = RpptRoiType::LTRB;
    // convert_roi(roiTensorPtrSrc, roiTypeSrc, batchSize);
    // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, batchSize);

    // case-wise RPP API and measure time script for Unit and Performance test
    printf("\nRunning %s %d times (each time with a batch size of %d images) and computing mean statistics...", func.c_str(), numIterations, batchSize);
    for (int perfRunCount = 0; perfRunCount < numIterations; perfRunCount++)
    {
        for(int iterCount = 0; iterCount < (int)imageNames.size() / batchSize; iterCount++)
        {
            vector<string>::const_iterator imagesPathStart = imageNamesPath.begin() + (iterCount * batchSize);
            vector<string>::const_iterator imagesPathEnd = imagesPathStart + batchSize;
            vector<string>::const_iterator imageNamesStart = imageNames.begin() + (iterCount * batchSize);
            vector<string>::const_iterator imageNamesEnd = imageNamesStart + batchSize;
            vector<string>::const_iterator imagesPathSecondStart = imageNamesPathSecond.begin() + (iterCount * batchSize);
            vector<string>::const_iterator imagesPathSecondEnd = imagesPathSecondStart + batchSize;

            // Set ROIs for src/dst
            set_roi(imagesPathStart, imagesPathEnd, roiTensorPtrSrc, roiTensorPtrDst, dstImgSizes);

            //Read images
            if(decoderType == 0)
                read_image_batch_turbojpeg(inputu8, srcDescPtr, imagesPathStart);
            else
                read_image_batch_opencv(inputu8, srcDescPtr, imagesPathStart);
            // if the input layout requested is PLN3, convert PKD3 inputs to PLN3 for first and second input batch
            if (layoutType == 1)
                convert_pkd3_to_pln3(inputu8, srcDescPtr);

            if(dualInputCase)
            {
                if(decoderType == 0)
                    read_image_batch_turbojpeg(inputu8Second, srcDescPtr, imagesPathSecondStart);
                else
                    read_image_batch_opencv(inputu8Second, srcDescPtr, imagesPathSecondStart);
                if (layoutType == 1)
                    convert_pkd3_to_pln3(inputu8Second, srcDescPtr);
            }

            convert_input_bitdepth(input, input_second, inputu8, inputu8Second, inputBitDepth, ioBufferSize, inputBufferSize, srcDescPtr, dualInputCase, conversionFactor);

            clock_t startCpuTime, endCpuTime;
            double startWallTime, endWallTime;
            switch (testCase)
            {
                case 0:
                {
                    testCaseName = "brightness";
                    Rpp32f alpha[batchSize];
                    Rpp32f beta[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        alpha[i] = 1.75;
                        beta[i] = 50;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_brightness_host(input, srcDescPtr, output, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 1:
                {
                    testCaseName = "gamma_correction";

                    Rpp32f gammaVal[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        gammaVal[i] = 1.9;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_gamma_correction_host(input, srcDescPtr, output, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 2:
                {
                    testCaseName = "blend";

                    Rpp32f alpha[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        alpha[i] = 0.4;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_blend_host(input, input_second, srcDescPtr, output, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 4:
                {
                    testCaseName = "contrast";

                    Rpp32f contrastFactor[batchSize];
                    Rpp32f contrastCenter[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        contrastFactor[i] = 2.96;
                        contrastCenter[i] = 128;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_contrast_host(input, srcDescPtr, output, dstDescPtr, contrastFactor, contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 8:
                {
                    testCaseName = "noise";

                    switch(additionalParam)
                    {
                        case 0:
                        {
                            Rpp32f noiseProbabilityTensor[batchSize];
                            Rpp32f saltProbabilityTensor[batchSize];
                            Rpp32f saltValueTensor[batchSize];
                            Rpp32f pepperValueTensor[batchSize];
                            Rpp32u seed = 1255459;
                            for (i = 0; i < batchSize; i++)
                            {
                                noiseProbabilityTensor[i] = 0.1f;
                                saltProbabilityTensor[i] = 0.5f;
                                saltValueTensor[i] = 1.0f;
                                pepperValueTensor[i] = 0.0f;
                            }

                            startWallTime = omp_get_wtime();
                            startCpuTime = clock();
                            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                                rppt_salt_and_pepper_noise_host(input, srcDescPtr, output, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                            else
                                missingFuncFlag = 1;

                            break;
                        }
                        case 1:
                        {
                            Rpp32f meanTensor[batchSize];
                            Rpp32f stdDevTensor[batchSize];
                            Rpp32u seed = 1255459;
                            for (i = 0; i < batchSize; i++)
                            {
                                meanTensor[i] = 0.0f;
                                stdDevTensor[i] = 0.2f;
                            }

                            startWallTime = omp_get_wtime();
                            startCpuTime = clock();
                            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                                rppt_gaussian_noise_host(input, srcDescPtr, output, dstDescPtr, meanTensor, stdDevTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                            else
                                missingFuncFlag = 1;

                            break;
                        }
                        case 2:
                        {
                            Rpp32f shotNoiseFactorTensor[batchSize];
                            Rpp32u seed = 1255459;
                            for (i = 0; i < batchSize; i++)
                            {
                                shotNoiseFactorTensor[i] = 80.0f;
                            }

                            startWallTime = omp_get_wtime();
                            startCpuTime = clock();
                            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                                rppt_shot_noise_host(input, srcDescPtr, output, dstDescPtr, shotNoiseFactorTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                            else
                                missingFuncFlag = 1;

                            break;
                        }
                        default:
                        {
                            missingFuncFlag = 1;
                            break;
                        }
                    }

                    break;
                }
                case 13:
                {
                    testCaseName = "exposure";

                    Rpp32f exposureFactor[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        exposureFactor[i] = 1.4;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_exposure_host(input, srcDescPtr, output, dstDescPtr, exposureFactor, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 20:
                {
                    testCaseName = "flip";

                    Rpp32u horizontalFlag[batchSize];
                    Rpp32u verticalFlag[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        horizontalFlag[i] = 1;
                        verticalFlag[i] = 0;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_flip_host(input, srcDescPtr, output, dstDescPtr, horizontalFlag, verticalFlag, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 21:
                {
                    testCaseName = "resize";

                    for (i = 0; i < batchSize; i++)
                    {
                        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
                        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_resize_host(input, srcDescPtr, output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 23:
                {
                    testCaseName = "rotate";

                    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    Rpp32f angle[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        angle[i] = 50;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_rotate_host(input, srcDescPtr, output, dstDescPtr, angle, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 24:
                {
                    testCaseName = "warp_affine";

                    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    Rpp32f6 affineTensor_f6[batchSize];
                    Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
                    for (i = 0; i < batchSize; i++)
                    {
                        affineTensor_f6[i].data[0] = 1.23;
                        affineTensor_f6[i].data[1] = 0.5;
                        affineTensor_f6[i].data[2] = 0;
                        affineTensor_f6[i].data[3] = -0.8;
                        affineTensor_f6[i].data[4] = 0.83;
                        affineTensor_f6[i].data[5] = 0;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_warp_affine_host(input, srcDescPtr, output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 30:
                {
                    testCaseName = "non_linear_blend";

                    Rpp32f stdDev[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        stdDev[i] = 50.0;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_non_linear_blend_host(input, input_second, srcDescPtr, output, dstDescPtr, stdDev, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 31:
                {
                    testCaseName = "color_cast";

                    RpptRGB rgbTensor[batchSize];
                    Rpp32f alphaTensor[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        rgbTensor[i].R = 0;
                        rgbTensor[i].G = 0;
                        rgbTensor[i].B = 100;
                        alphaTensor[i] = 0.5;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_color_cast_host(input, srcDescPtr, output, dstDescPtr, rgbTensor, alphaTensor, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 36:
                {
                    testCaseName = "color_twist";

                    Rpp32f brightness[batchSize];
                    Rpp32f contrast[batchSize];
                    Rpp32f hue[batchSize];
                    Rpp32f saturation[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        brightness[i] = 1.4;
                        contrast[i] = 0.0;
                        hue[i] = 60.0;
                        saturation[i] = 1.9;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_color_twist_host(input, srcDescPtr, output, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 37:
                {
                    testCaseName = "crop";

                    for (i = 0; i < batchSize; i++)
                    {
                        roiTensorPtrSrc[i].xywhROI.xy.x = 10;
                        roiTensorPtrSrc[i].xywhROI.xy.y = 10;
                        dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                        dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_crop_host(input, srcDescPtr, output, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 38:
                {
                    testCaseName = "crop_mirror_normalize";
                    Rpp32f multiplier[batchSize * srcDescPtr->c];
                    Rpp32f offset[batchSize * srcDescPtr->c];
                    Rpp32u mirror[batchSize];
                    if (srcDescPtr->c == 3)
                    {
                        Rpp32f meanParam[3] = { 60.0f, 80.0f, 100.0f };
                        Rpp32f stdDevParam[3] = { 0.9f, 0.9f, 0.9f };
                        Rpp32f offsetParam[3] = { - meanParam[0] / stdDevParam[0], - meanParam[1] / stdDevParam[1], - meanParam[2] / stdDevParam[2] };
                        Rpp32f multiplierParam[3] = {  1.0f / stdDevParam[0], 1.0f / stdDevParam[1], 1.0f / stdDevParam[2] };

                        for (i = 0, j = 0; i < batchSize; i++, j += 3)
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

                        for (i = 0; i < batchSize; i++)
                        {
                            multiplier[i] = multiplierParam;
                            offset[i] = offsetParam;
                            mirror[i] = 1;
                        }
                    }

                    for (i = 0; i < batchSize; i++)
                    {
                        roiTensorPtrDst[i].xywhROI.xy.x = 10;
                        roiTensorPtrDst[i].xywhROI.xy.y = 10;
                        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 3 || inputBitDepth == 4 || inputBitDepth == 5)
                        rppt_crop_mirror_normalize_host(input, srcDescPtr, output, dstDescPtr, offset, multiplier, mirror, roiTensorPtrDst, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 39:
                {
                    testCaseName = "resize_crop_mirror";

                    if (interpolationType != RpptInterpolationType::BILINEAR)
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    Rpp32u mirror[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        mirror[i] = 1;
                    }

                    for (i = 0; i < batchSize; i++)
                    {
                        roiTensorPtrSrc[i].xywhROI.xy.x = 10;
                        roiTensorPtrSrc[i].xywhROI.xy.y = 10;
                        dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                        dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
                        roiTensorPtrSrc[i].xywhROI.roiWidth = 50;
                        roiTensorPtrSrc[i].xywhROI.roiHeight = 50;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 3 || inputBitDepth == 4 || inputBitDepth == 5)
                        rppt_resize_crop_mirror_host(input, srcDescPtr, output, dstDescPtr, dstImgSizes, interpolationType, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 70:
                {
                    testCaseName = "copy";

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_copy_host(input, srcDescPtr, output, dstDescPtr, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 80:
                {
                    testCaseName = "resize_mirror_normalize";

                    if (interpolationType != RpptInterpolationType::BILINEAR)
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    for (i = 0; i < batchSize; i++)
                    {
                        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                    }

                    Rpp32f mean[batchSize * 3];
                    Rpp32f stdDev[batchSize * 3];
                    Rpp32u mirror[batchSize];
                    for (i = 0, j = 0; i < batchSize; i++, j += 3)
                    {
                        mean[j] = 60.0;
                        stdDev[j] = 1.0;

                        mean[j + 1] = 80.0;
                        stdDev[j + 1] = 1.0;

                        mean[j + 2] = 100.0;
                        stdDev[j + 2] = 1.0;
                        mirror[i] = 1;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_resize_mirror_normalize_host(input, srcDescPtr, output, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 81:
                {
                    testCaseName = "color_jitter";

                    Rpp32f brightness[batchSize];
                    Rpp32f contrast[batchSize];
                    Rpp32f hue[batchSize];
                    Rpp32f saturation[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        brightness[i] = 1.02;
                        contrast[i] = 1.1;
                        hue[i] = 0.02;
                        saturation[i] = 1.3;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_color_jitter_host(input, srcDescPtr, output, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 83:
                {
                    testCaseName = "gridmask";

                    Rpp32u tileWidth = 40;
                    Rpp32f gridRatio = 0.6;
                    Rpp32f gridAngle = 0.5;
                    RpptUintVector2D translateVector;
                    translateVector.x = 0.0;
                    translateVector.y = 0.0;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_gridmask_host(input, srcDescPtr, output, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 84:
                {
                    testCaseName = "spatter";

                    RpptRGB spatterColor;

                    // Mud Spatter
                    spatterColor.R = 65;
                    spatterColor.G = 50;
                    spatterColor.B = 23;

                    // Blood Spatter
                    // spatterColor.R = 98;
                    // spatterColor.G = 3;
                    // spatterColor.B = 3;

                    // Ink Spatter
                    // spatterColor.R = 5;
                    // spatterColor.G = 20;
                    // spatterColor.B = 64;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_spatter_host(input, srcDescPtr, output, dstDescPtr, spatterColor, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 85:
                {
                    testCaseName = "swap_channels";

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_swap_channels_host(input, srcDescPtr, output, dstDescPtr, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 86:
                {
                    testCaseName = "color_to_greyscale";

                    RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_color_to_greyscale_host(input, srcDescPtr, output, dstDescPtr, srcSubpixelLayout, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                default:
                    missingFuncFlag = 1;
                    break;
            }

            endCpuTime = clock();
            endWallTime = omp_get_wtime();
            cpuTime = ((double)(endCpuTime - startCpuTime)) / CLOCKS_PER_SEC;
            wallTime = endWallTime - startWallTime;
            if (missingFuncFlag == 1)
            {
                printf("\nThe functionality %s doesn't yet exist in RPP\n", func.c_str());
                return -1;
            }

            maxWallTime = std::max(maxWallTime, wallTime);
            minWallTime = std::min(minWallTime, wallTime);
            avgWallTime += wallTime;
            cpuTime *= 1000;
            wallTime *= 1000;

            if (testType == 0)
            {
                cout <<"\n\n";
                cout <<"CPU Backend Clock Time: "<< cpuTime <<" ms/batch"<< endl;
                cout <<"CPU Backend Wall Time: "<< wallTime <<" ms/batch"<< endl;

                // Reconvert other bit depths to 8u for output display purposes
                convert_output_bitdepth_to_u8(output, outputu8, inputBitDepth, oBufferSize, outputBufferSize, dstDescPtr, invConversionFactor);

                // If DEBUG_MODE is set to 1 dump the outputs to csv files for debugging
                if(DEBUG_MODE && iterCount == 0)
                {
                    std::ofstream refFile;
                    refFile.open(func + ".csv");
                    for (int i = 0; i < oBufferSize; i++)
                        refFile << static_cast<int>(*(outputu8 + i)) << ",";
                    refFile.close();
                }

                /*Compare the output of the function with golden outputs only if
                1.QA Flag is set
                2.input bit depth 0 (U8)
                3.source and destination layout are the same*/
                if(qaFlag && inputBitDepth == 0 && (srcDescPtr->layout == dstDescPtr->layout))
                    compare_output<Rpp8u>(outputu8, testCaseName, srcDescPtr, dstDescPtr, dstImgSizes, batchSize, interpolationTypeName, noiseTypeName, testCase, dst);

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

                // OpenCV dump (if testType is unit test and QA mode is not set)
                if(!qaFlag)
                    write_image_batch_opencv(dst, outputu8, dstDescPtr, imageNamesStart, dstImgSizes, MAX_IMAGE_DUMP);
            }
        }
    }

    rppDestroyHost(handle);

    if(testType == 1)
    {
        // Display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= numIterations;
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }

    cout<<endl;

    // Free memory
    free(roiTensorPtrSrc);
    free(roiTensorPtrDst);
    free(dstImgSizes);
    free(input);
    free(inputu8);
    free(inputu8Second);
    free(outputu8);
    free(input_second);
    free(output);
    return 0;
}