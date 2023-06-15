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

typedef half Rpp16f;

using namespace cv;
using namespace std;

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
    int numRuns = atoi(argv[8]);
    int testType = atoi(argv[9]);     // 0 for unit and 1 for performance test
    int layoutType = atoi(argv[10]); // 0 for pkd3 / 1 for pln3 / 2 for pln1
    int qaFlag = atoi(argv[12]);
    int decoderType = atoi(argv[13]);
    int batchSize = atoi(argv[14]);

    bool additionalParamCase = (testCase == 8 || testCase == 21 || testCase == 23|| testCase == 24 || testCase == 40 || testCase == 41 || testCase == 49);
    bool dualInputCase = (testCase == 2);
    bool randomOutputCase = (testCase == 84);
    bool kernelSizeCase = (testCase == 40 || testCase == 41 || testCase == 49);
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
        printf("\ncase number (0:84) = %s", argv[6]);
        printf("\nnumber of times to run = %s", argv[8]);
        printf("\ntest type - (0 = unit tests / 1 = performance tests) = %s", argv[9]);
        printf("\nlayout type - (0 = PKD3/ 1 = PLN3/ 2 = PLN1) = %s", argv[10]);
        printf("\nqa mode - 0/1 = %s", argv[12]);
        printf("\ndecoder type - (0 = TurboJPEG / 1 = OpenCV) = %s", argv[13]);
        printf("\nbatch size = %s", argv[14]);
    }

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:84> <number of iterations > 0> <batch size > 1> <verbosity = 0/1>>\n");
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

    if(batchSize > MAX_BATCH_SIZE)
    {
        std::cerr << "\n Batchsize should be less than or equal to "<< MAX_BATCH_SIZE << " Aborting!";
        exit(0);
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
    std::string noiseTypeName = "";
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
            std::cerr << " \n The number of images in the input folders must be the same.";
            exit(0);
        }
    }
    noOfImages = imageNames.size();

    if(noOfImages < batchSize || ((noOfImages % batchSize) != 0))
    {
        replicate_last_image_to_fill_batch(imageNamesPath[noOfImages - 1], imageNamesPath, imageNames, imageNames[noOfImages - 1], noOfImages, batchSize);
        if(dualInputCase)
            replicate_last_image_to_fill_batch(imageNamesPathSecond[noOfImages - 1], imageNamesPathSecond, imageNamesSecond, imageNamesSecond[noOfImages - 1], noOfImages, batchSize);
        noOfImages = imageNames.size();
    }

    if(!noOfImages)
    {
        std::cerr << "Not able to find any images in the folder specified. Please check the input path";
        exit(0);
    }

    if(qaFlag)
    {
        sort(imageNames.begin(), imageNames.end());
        if(dualInputCase)
            sort(imageNamesSecond.begin(), imageNamesSecond.end());
    }

    // Initialize ROI tensors for src/dst
    RpptROI *roiTensorPtrSrc, *roiTensorPtrDst;
    hipHostMalloc(&roiTensorPtrSrc, batchSize * sizeof(RpptROI));
    hipHostMalloc(&roiTensorPtrDst, batchSize * sizeof(RpptROI));

    // Initialize the ImagePatch for dst
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, batchSize * sizeof(RpptImagePatch));

    // Set ROI tensors types for src/dst
    RpptRoiType roiTypeSrc, roiTypeDst;
    roiTypeSrc = RpptRoiType::XYWH;
    roiTypeDst = RpptRoiType::XYWH;

    Rpp32u outputChannels = inputChannels;
    if(pln1OutTypeCase)
        outputChannels = 1;
    Rpp32u offsetInBytes = 0;

    set_max_dimensions(imageNamesPath, maxHeight, maxWidth);

    // Set numDims, offset, n/c/h/w values, strides for src/dst
    set_descriptor_dims_and_strides(srcDescPtr, batchSize, maxHeight, maxWidth, inputChannels, offsetInBytes);
    set_descriptor_dims_and_strides(dstDescPtr, batchSize, maxHeight, maxWidth, outputChannels, offsetInBytes);

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
    void *d_input, *d_input_second, *d_output;

    input = static_cast<Rpp8u *>(calloc(inputBufferSize, 1));
    input_second = static_cast<Rpp8u *>(calloc(inputBufferSize, 1));
    output = static_cast<Rpp8u *>(calloc(outputBufferSize, 1));

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, batchSize);

    int noOfIterations = (int)imageNames.size() / batchSize;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double wallTime;
    string testCaseName;

    //Allocate hip memory for src/dst
    hipMalloc(&d_input, inputBufferSize);
    hipMalloc(&d_output, outputBufferSize);
    if(dualInputCase)
        hipMalloc(&d_input_second, inputBufferSize);

    // case-wise RPP API and measure time script for Unit and Performance test
    printf("\nRunning %s %d times (each time with a batch size of %d images) and computing mean statistics...", func.c_str(), numRuns, batchSize);
    for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
    {
        for(int iterCount = 0; iterCount < noOfIterations; iterCount++)
        {
            vector<string>::const_iterator imagesPathStart = imageNamesPath.begin() + (iterCount * batchSize);
            vector<string>::const_iterator imagesPathEnd = imagesPathStart + batchSize;
            vector<string>::const_iterator imageNamesStart = imageNames.begin() + (iterCount * batchSize);
            vector<string>::const_iterator imageNamesEnd = imageNamesStart + batchSize;
            vector<string>::const_iterator imagesPathSecondStart = imageNamesPathSecond.begin() + (iterCount * batchSize);
            vector<string>::const_iterator imagesPathSecondEnd = imagesPathSecondStart + batchSize;

            // Set ROIs for src/dst
            set_src_and_dst_roi(imagesPathStart, imagesPathEnd, roiTensorPtrSrc, roiTensorPtrDst, dstImgSizes);

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

            // Convert inputs to correponding bit depth specified by user
            convert_input_bitdepth(input, input_second, inputu8, inputu8Second, inputBitDepth, ioBufferSize, inputBufferSize, srcDescPtr, dualInputCase, conversionFactor);

            //copy decoded inputs to hip buffers
            hipMemcpy(d_input, input, inputBufferSize, hipMemcpyHostToDevice);
            hipMemcpy(d_output, output, outputBufferSize, hipMemcpyHostToDevice);
            if(dualInputCase)
                hipMemcpy(d_input_second, input_second, inputBufferSize, hipMemcpyHostToDevice);

            // Uncomment to run test case with an xywhROI override
            // roi.xywhROI = {0, 0, 25, 25};
            // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, batchSize);
            // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, batchSize);

            // Uncomment to run test case with an ltrbROI override
            // roiTypeSrc = RpptRoiType::LTRB;
            // roi.ltrbROI = {10, 10, 40, 40};
            // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, batchSize);
            // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, batchSize);

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
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_brightness_gpu(d_input, srcDescPtr, d_output, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 1:
            {
                testCaseName = "gamma_correction";

                Rpp32f gammaVal[batchSize];
                for (i = 0; i < batchSize; i++)
                    gammaVal[i] = 1.9;

                startWallTime = omp_get_wtime();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_gamma_correction_gpu(d_input, srcDescPtr, d_output, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 2:
            {
                testCaseName = "blend";

                Rpp32f alpha[batchSize];
                for (i = 0; i < batchSize; i++)
                    alpha[i] = 0.4;

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

                Rpp32f contrastFactor[batchSize];
                Rpp32f contrastCenter[batchSize];
                for (i = 0; i < batchSize; i++)
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

                Rpp32f exposureFactor[batchSize];
                for (i = 0; i < batchSize; i++)
                    exposureFactor[i] = 1.4;

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
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 37:
            {
                testCaseName = "crop";

                for (i = 0; i < batchSize; i++)
                {
                    roiTensorPtrDst[i].xywhROI.xy.x = 10;
                    roiTensorPtrDst[i].xywhROI.xy.y = 10;
                    dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                    dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
                }

                startWallTime = omp_get_wtime();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_crop_gpu(d_input, srcDescPtr, d_output, dstDescPtr, roiTensorPtrDst, roiTypeSrc, handle);
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
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 3 || inputBitDepth == 4 || inputBitDepth == 5)
                    rppt_crop_mirror_normalize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, offset, multiplier, mirror, roiTensorPtrDst, roiTypeSrc, handle);
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
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_spatter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, spatterColor, roiTensorPtrSrc, roiTypeSrc, handle);
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
            wallTime *= 1000;
            if (testType == 0)
            {
                cout << "\n\nGPU Backend Wall Time: " << wallTime <<" ms/batch"<< endl;
                hipMemcpy(output, d_output, outputBufferSize, hipMemcpyDeviceToHost);

                // Reconvert other bit depths to 8u for output display purposes
                convert_output_bitdepth_to_u8(output, outputu8, inputBitDepth, oBufferSize, outputBufferSize, dstDescPtr, invConversionFactor);

                // if DEBUG_MODE is set to 1, the output of the first iteration will be dumped to csv files for debugging purposes.
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
                2.input bit depth 0 (Input U8 && Output U8)
                3.source and destination layout are the same*/
                if(qaFlag && inputBitDepth == 0 && (srcDescPtr->layout == dstDescPtr->layout) && !(randomOutputCase))
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
    rppDestroyGPU(handle);
    if(testType == 1)
    {
        // Display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns * noOfIterations);
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
    hipFree(d_input);
    hipFree(d_input_second);
    hipFree(d_output);
    return 0;
}
