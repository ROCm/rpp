/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "rpp.h"
#include "../rpp_test_suite_single_image_utils.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <cstdlib>
#include <turbojpeg.h>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 19;

    char *src = argv[1];
    char *srcSecond = argv[2];
    string dst = argv[3];

    int BitDepthTestMode = atoi(argv[4]);
    unsigned int outputFormatToggle = atoi(argv[5]);
    int testCase = atoi(argv[6]);
    int numRuns = atoi(argv[8]);
    int testType = atoi(argv[9]);     // 0 for unit and 1 for performance test
    int layoutType = atoi(argv[10]); // 0 for pkd3 / 1 for pln3 / 2 for pln1
    int qaFlag = atoi(argv[12]);
    int decoderType = atoi(argv[13]);
    int batchSize = atoi(argv[14]);

    bool additionalParamCase = (additionalParamCases.find(testCase) != additionalParamCases.end());
    bool kernelSizeCase = (kernelSizeCases.find(testCase) != kernelSizeCases.end());
    bool dualInputCase = (dualInputCases.find(testCase) != dualInputCases.end());
    bool randomOutputCase = (randomOutputCases.find(testCase) != randomOutputCases.end());
    bool nonQACase = (nonQACases.find(testCase) != nonQACases.end());
    bool interpolationTypeCase = (interpolationTypeCases.find(testCase) != interpolationTypeCases.end());
    bool reductionTypeCase = (reductionTypeCases.find(testCase) != reductionTypeCases.end());
    bool noiseTypeCase = (noiseTypeCases.find(testCase) != noiseTypeCases.end());
    bool pln1OutTypeCase = (pln1OutTypeCases.find(testCase) != pln1OutTypeCases.end());

    unsigned int verbosity = atoi(argv[11]);
    unsigned int additionalParam = additionalParamCase ? atoi(argv[7]) : 1;
    int roiList[4] = {atoi(argv[15]), atoi(argv[16]), atoi(argv[17]), atoi(argv[18])};
    string scriptPath = argv[19];

    if (verbosity == 1)
    {
        cout << "\nInputs for this test case are:";
        cout << "\nsrc1 = " << argv[1];
        cout << "\nsrc2 = " << argv[2];
        if (testType == UNIT_TEST) // unit test mode
            cout << "\ndst = " << argv[3];
        cout << "\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = " << argv[4];
        cout << "\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = " << argv[5];
        cout << "\ncase number (0:91) = " << argv[6];
        cout << "\nnumber of times to run = " << argv[8];
        cout << "\ntest type - (0 = unit tests / 1 = performance tests) = " << argv[9];
        cout << "\nlayout type - (0 = PKD3/ 1 = PLN3/ 2 = PLN1) = " << argv[10];
        cout << "\nqa mode - 0/1 = " << argv[12];
        cout << "\ndecoder type - (0 = TurboJPEG / 1 = OpenCV) = " << argv[13];
        cout << "\nbatch size = " << argv[14];
    }

    if (argc < MIN_ARG_COUNT)
    {
        cout << "\nImproper Usage! Needs all arguments!\n";
        cout << "\nUsage: <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:87> <number of runs > 0> <layout type (0 = PKD3/ 1 = PLN3/ 2 = PLN1)> <qa mode (0/1)> <decoder type (0/1)> <batch size > 1> <roiList> <verbosity = 0/1>>\n";
        return -1;
    }

    if (layoutType == 2)
    {
        if(testCase == COLOR_TWIST || testCase == COLOR_CAST || testCase == GLITCH || testCase == COLOR_TEMPERATURE || testCase == COLOR_TO_GREYSCALE || testCase == HUE || testCase == SATURATION)
        {
            cout << "\ncase " << testCase << " does not exist for PLN1 layout\n";
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        else if (outputFormatToggle != 0)
        {
            cout << "\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = 0\n";
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
    }

    if(pln1OutTypeCase && outputFormatToggle != 0)
    {
        cout << "\ntest case " << testCase << " don't have outputFormatToggle! Please input outputFormatToggle = 0\n";
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
    else if(batchSize > MAX_BATCH_SIZE)
    {
        std::cerr << "\n Batchsize should be less than or equal to "<< MAX_BATCH_SIZE << " Aborting!";
        exit(0);
    }
    else if(testCase == RICAP && batchSize < 2)
    {
        std::cerr<<"\n RICAP only works with BatchSize > 1";
        exit(0);
    }

    // Get function name
    string funcName = augmentationMap[testCase];
    if (funcName.empty())
    {
        if (testType == UNIT_TEST) // unit test mode
            cout << "\ncase " << testCase << " is not supported\n";

        return -1;
    }

    // Determine the type of function to be used based on the specified layout type
    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    RpptInterpolationType interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
    string funcType = set_function_type(layoutType, pln1OutTypeCase, outputFormatToggle, "HIP");
    string func = funcName;
    std::string interpolationTypeName = "";
    set_descriptor_data_type_name(BitDepthTestMode, func);
    func += funcType;
    if (kernelSizeCase) func += "_kernelSize" + std::to_string(additionalParam);
    // else if (interpolationTypeCase)
    // {
    //     interpolationTypeName = get_interpolation_type(additionalParam, interpolationType);
    //     func += "_interpolationType";
    //     func += interpolationTypeName.c_str();
    // }
    if(!qaFlag) dst += "/" + func;
    Rpp32s additionalStride = 0;
    if (kernelSizeCase)
        additionalStride = additionalParam / 2;
    Rpp32u srcOffsetInBytes = (kernelSizeCase) ? (12 * (additionalParam / 2)) : 0;
    Rpp32u dstOffsetInBytes = 0;
    int noOfImages = 0, missingFuncFlag = 0;
    Rpp32f conversionFactor = 1.0f / 255.0;
    bool isColor = (layoutType != 2);
    vector<Mat> inputVec, inputVecSecond;
    if (decoderType == 0)
    {
        inputVec = loadBatchImages_jpegd(src, noOfImages, isColor);
        if (dualInputCase)
            inputVecSecond = loadBatchImages_jpegd(srcSecond, noOfImages, isColor);
    }
    else
    {
        inputVec = loadBatchImages_cv(src, noOfImages, isColor);
        if (dualInputCase)
            inputVecSecond = loadBatchImages_cv(srcSecond, noOfImages, isColor);
    }

    if (noOfImages == 0) { cerr << "No images found!"; return -1; }

    // Convert batch to user-specified bit depth
    convertBatchBitDepth(inputVec, BitDepthTestMode, conversionFactor);
    convertBatchBitDepth(inputVecSecond, BitDepthTestMode, conversionFactor);
    if (noOfImages < batchSize) {
        for (int i = noOfImages; i < batchSize; i++)
            inputVec.push_back(inputVec[noOfImages - 1]);
        noOfImages = batchSize;
    }
    vector<RpptDesc> srcDescPtr(noOfImages), dstDescPtr(noOfImages);
    RpptROI *roi = nullptr;
    CHECK_RETURN_STATUS(hipHostMalloc(&roi, noOfImages * sizeof(RpptROI)));
    memset(roi, 0, noOfImages * sizeof(RpptROI));
    RpptImagePatch dstImgSizes[noOfImages];

    int inputChannel = set_input_channels(layoutType);
    int outputChannel = inputChannel;
    if(pln1OutTypeCase)
        outputChannel = 1;
    set_descriptor_layout(srcDescPtr, dstDescPtr, layoutType, pln1OutTypeCase, outputFormatToggle, noOfImages);
    set_descriptor_data_type(BitDepthTestMode, srcDescPtr, dstDescPtr, noOfImages);

    initializeDescriptors(inputVec, srcDescPtr, inputChannel, true, additionalStride, srcOffsetInBytes);
    initializeDescriptors(inputVec, dstDescPtr, outputChannel, true, additionalStride, dstOffsetInBytes);
    int roiHeightList[noOfImages], roiWidthList[noOfImages];
    initializeROI(inputVec, roi, dstDescPtr, roiList, roiHeightList, roiWidthList);

    // For CROP, the output descriptor must match the crop ROI dimensions, not the input dimensions.
    // Re-initialize dstDescPtr using the crop output size so allocations and strides are correct.
    if (testCase == CROP)
    {
        for (int i = 0; i < noOfImages; i++)
        {
            Rpp32u cropW = roiWidthList[i];
            Rpp32u cropH = roiHeightList[i];
            // Pad width to next multiple of 8 (same rule as initializeDescriptors)
            dstDescPtr[i].w = (cropW % 8 == 0) ? cropW : (cropW / 8) * 8 + 8;
            dstDescPtr[i].h = cropH;
            if (dstDescPtr[i].layout == RpptLayout::NHWC)
            {
                dstDescPtr[i].strides.nStride = dstDescPtr[i].h * dstDescPtr[i].w * outputChannel;
                dstDescPtr[i].strides.hStride = dstDescPtr[i].w * outputChannel;
            }
            else
            {
                dstDescPtr[i].strides.nStride = dstDescPtr[i].h * dstDescPtr[i].w * outputChannel;
                dstDescPtr[i].strides.hStride = dstDescPtr[i].w;
                dstDescPtr[i].strides.cStride = dstDescPtr[i].h * dstDescPtr[i].w;
            }
        }
    }

    // Initialize output buffers with correct size
    vector<Mat> outputVec(noOfImages);
    vector<Rpp32u> actualInputWidth(noOfImages), actualInputHeight(noOfImages);
    for (int i = 0; i < noOfImages; i++)
    {
        actualInputWidth[i] = inputVec[i].cols;
        actualInputHeight[i] = inputVec[i].rows;

        // For CROP the output dimensions are the ROI dimensions, not the input image dimensions
        Rpp32u outW = (testCase == CROP) ? roiWidthList[i] : actualInputWidth[i];
        Rpp32u outH = (testCase == CROP) ? roiHeightList[i] : actualInputHeight[i];

        int cvType = get_cv_type(dstDescPtr[i].dataType, 1);

        if (dstDescPtr[i].layout == RpptLayout::NCHW && dstDescPtr[i].c == 3)
            outputVec[i] = Mat(outH * dstDescPtr[i].c, outW, cvType);
        else
            outputVec[i] = Mat(outH, outW, get_cv_type(dstDescPtr[i].dataType, dstDescPtr[i].c));
        
        // Prepare input buffers as contiguous memory for direct device copy
        if (srcDescPtr[i].layout == RpptLayout::NCHW && isColor)
        {
            // Convert PKD3 to PLN3 for NCHW layout
            inputVec[i] = convert_pkd3_to_pln3(inputVec[i]);
            if (dualInputCase)
                inputVecSecond[i] = convert_pkd3_to_pln3(inputVecSecond[i]);
        }
    }

    /* Initialize alpha and beta for blend and brightness kernels*/
    Rpp32f *alpha = nullptr;
    Rpp32f *beta = nullptr;

    if(testCase == BLEND)
        CHECK_RETURN_STATUS(hipHostMalloc(&alpha, sizeof(Rpp32f)));
        
    if(testCase == BRIGHTNESS)
    {
        CHECK_RETURN_STATUS(hipHostMalloc(&alpha, sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&beta, sizeof(Rpp32f)));
    }
    
    Rpp32u numThreads = 1;

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
    rppCreate(&handle, noOfImages, numThreads, nullptr, backend);
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double wallTime;
    string testCaseName;

    // case-wise RPP API and measure time script for Unit and Performance test
    cout << "\nRunning " << func << " " << numRuns << " times (each time with a batch size of " << batchSize << " images) and computing mean statistics...";

    for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
    {
        initializeROI(inputVec, roi, dstDescPtr, roiList, roiHeightList, roiWidthList);
        for (int i = 0; i < noOfImages; i++)
        {
            RppStatus errorCodeCapture = RPP_SUCCESS;
            double startWallTime, endWallTime;

            // Calculate element sizes
            size_t inElementSize = 1;
            if (srcDescPtr[i].dataType == RpptDataType::F16)
                inElementSize = 2;
            else if (srcDescPtr[i].dataType == RpptDataType::F32)
                inElementSize = 4;

            size_t outElementSize = 1;
            if (dstDescPtr[i].dataType == RpptDataType::F16)
                outElementSize = 2;
            else if (dstDescPtr[i].dataType == RpptDataType::F32)
                outElementSize = 4;

            // Allocate device memory for this image
            void *d_input, *d_inputSecond = nullptr, *d_output;
            size_t dataSizeInBytes = (size_t)srcDescPtr[i].strides.nStride * inElementSize;
            size_t actualInputCols = inputVec[i].cols;
            size_t inputInBytes = actualInputCols * inputVec[i].channels() * inElementSize;
            size_t inputSizeInBytes = dataSizeInBytes + srcDescPtr[i].offsetInBytes;
            size_t outputSizeInBytes = (size_t)dstDescPtr[i].strides.nStride * outElementSize + dstDescPtr[i].offsetInBytes;

            CHECK_RETURN_STATUS(hipMalloc(&d_input, inputSizeInBytes));
            CHECK_RETURN_STATUS(hipMemset(d_input, 0, inputSizeInBytes));
            if (dualInputCase)
            {
                CHECK_RETURN_STATUS(hipMalloc(&d_inputSecond, inputSizeInBytes));
                CHECK_RETURN_STATUS(hipMemset(d_inputSecond, 0, inputSizeInBytes));
            }
            CHECK_RETURN_STATUS(hipMalloc(&d_output, outputSizeInBytes));
            CHECK_RETURN_STATUS(hipMemset(d_output, 0, outputSizeInBytes));

            // Copy input data to device
            Rpp8u *inputTemp = (inputVec[i].data);
            Rpp8u *inputTempSecond;
            if (dualInputCase)
                inputTempSecond = (inputVecSecond[i].data);
            Rpp8u *d_input_offsetted = static_cast<Rpp8u*>(d_input) + srcDescPtr[i].offsetInBytes;
            Rpp8u *d_inputSecond_offsetted = static_cast<Rpp8u*>(d_inputSecond) + srcDescPtr[i].offsetInBytes;
            for(int j = 0; j < inputVec[i].rows; j++)
            {
                Rpp8u* d_inputRowTemp = d_input_offsetted + j * srcDescPtr[i].strides.hStride * inElementSize;
                Rpp8u* inputRowTemp = inputTemp + j * inputVec[i].step[0];
                CHECK_RETURN_STATUS(hipMemcpy(d_inputRowTemp, inputRowTemp, inputInBytes, hipMemcpyHostToDevice));
                if (dualInputCase)
                {
                    Rpp8u* d_inputSecondRowTemp = d_inputSecond_offsetted + j * srcDescPtr[i].strides.hStride * inElementSize;
                    Rpp8u* inputSecondRowTemp = inputTempSecond + j * inputVec[i].step[0];
                    CHECK_RETURN_STATUS(hipMemcpy(d_inputSecondRowTemp, inputSecondRowTemp, inputInBytes, hipMemcpyHostToDevice));
                }
            }

            switch (testCase)
            {
                case BRIGHTNESS:
                {
                    testCaseName = "brightness";
                    *alpha = 1.75f;
                    *beta = 50.0f;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_brightness(d_input, &srcDescPtr[i], d_output, &dstDescPtr[i], alpha, beta, &roi[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BLEND:
                {
                    testCaseName = "blend";

                    *alpha = 0.4f;
                    
                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                            errorCodeCapture = rppt_blend(d_input, d_inputSecond, &srcDescPtr[i], d_output, &dstDescPtr[i], alpha, &roi[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case FLIP:
                {
                    testCaseName = "flip";

                    Rpp32u horizontalFlag = 1;
                    Rpp32u verticalFlag = 0;

                    Rpp32u x = roi[i].xywhROI.xy.x;
                    Rpp32u y = roi[i].xywhROI.xy.y;
                    Rpp32u w = roi[i].xywhROI.roiWidth;
                    Rpp32u h = roi[i].xywhROI.roiHeight;

                    roi[i].ltrbROI.lt.x = x;
                    roi[i].ltrbROI.lt.y = y;
                    roi[i].ltrbROI.rb.x = x + w - 1;
                    roi[i].ltrbROI.rb.y = y + h - 1;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_flip(d_input, &srcDescPtr[i], d_output, &dstDescPtr[i], &horizontalFlag, &verticalFlag, &roi[i], RpptRoiType::LTRB, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RESIZE:
                {
                    testCaseName = "resize";

                    dstImgSizes[i].width = roi[i].xywhROI.roiWidth = roi[i].xywhROI.roiWidth / 2;
                    dstImgSizes[i].height = roi[i].xywhROI.roiHeight = roi[i].xywhROI.roiHeight / 2;

                    Rpp32u x = roi[i].xywhROI.xy.x;
                    Rpp32u y = roi[i].xywhROI.xy.y; 
                    Rpp32u w = roi[i].xywhROI.roiWidth;
                    Rpp32u h = roi[i].xywhROI.roiHeight;

                    roi[i].ltrbROI.lt.x = x;
                    roi[i].ltrbROI.lt.y = y;
                    roi[i].ltrbROI.rb.x = x + w - 1;
                    roi[i].ltrbROI.rb.y = y + h - 1;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_resize(d_input, &srcDescPtr[i], d_output, &dstDescPtr[i], &dstImgSizes[i], interpolationType, &roi[i], RpptRoiType::LTRB, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CROP:
                {
                    testCaseName = "crop";

                    roi[i].xywhROI.xy.x = roiList[0];
                    roi[i].xywhROI.xy.y = roiList[1];
                    dstImgSizes[i].width = roi[i].xywhROI.roiWidth = roiWidthList[i];
                    dstImgSizes[i].height = roi[i].xywhROI.roiHeight = roiHeightList[i];

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_crop(d_input, &srcDescPtr[i], d_output, &dstDescPtr[i],  &roi[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BOX_FILTER:
                {
                    testCaseName = "box_filter";
                    Rpp32u kernelSize = additionalParam;

                    if (borderType != RpptImageBorderType::REPLICATE)
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_box_filter(d_input, &srcDescPtr[i], d_output, &dstDescPtr[i], kernelSize, borderType, &roi[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case MEDIAN_FILTER:
                {
                    testCaseName = "median_filter";
                    Rpp32u kernelSize = additionalParam;
                    if (borderType != RpptImageBorderType::REPLICATE)
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                            errorCodeCapture = rppt_median_filter(d_input, &srcDescPtr[i], d_output, &dstDescPtr[i], kernelSize, borderType, &roi[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND);
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

            CHECK_RETURN_STATUS(hipDeviceSynchronize());
            endWallTime = omp_get_wtime();

            if (missingFuncFlag == 1)
            {
                cout << "\nThe functionality " << func << " doesn't yet exist in RPP\n";
                CHECK_RETURN_STATUS(hipFree(d_input));
                if (dualInputCase)
                    CHECK_RETURN_STATUS(hipFree(d_inputSecond));
                CHECK_RETURN_STATUS(hipFree(d_output));
                return RPP_ERROR_NOT_IMPLEMENTED;
            }
            if (errorCodeCapture != RPP_SUCCESS)
            {
                cout << "\nThe functionality " << func << " returned an error status " << rppStatusToString[errorCodeCapture] << " on run number " << perfRunCount + 1 << " of " << numRuns << " runs.\n";
                CHECK_RETURN_STATUS(hipFree(d_input));
                if (dualInputCase)
                    CHECK_RETURN_STATUS(hipFree(d_inputSecond));
                CHECK_RETURN_STATUS(hipFree(d_output));
                return errorCodeCapture;
            }

            // Copy output data from device to host
            Rpp8u *d_output_offsetted = static_cast<Rpp8u*>(d_output) + dstDescPtr[i].offsetInBytes;
            Rpp8u *outputTemp = (outputVec[i].data);
            
            // For resize/crop, use roi dimensions; for others, use input dimensions
            int outputWidth = (testCase == RESIZE || testCase == CROP) ? dstImgSizes[i].width : actualInputWidth[i];
            int outputHeight = (testCase == RESIZE || testCase == CROP) ? dstImgSizes[i].height : actualInputHeight[i];
            
            // Calculate correct row size based on layout
            size_t rowSizeInBytes;
            if (dstDescPtr[i].layout == RpptLayout::NCHW)
            {
                // For planar layout, each row is single channel width
                rowSizeInBytes = outputWidth * outElementSize;
            }
            else
            {
                // For packed layout, each row is width * channels
                rowSizeInBytes = outputWidth * outputChannel * outElementSize;
            }
            
            // Copy output based on layout
            if (dstDescPtr[i].layout == RpptLayout::NCHW && dstDescPtr[i].c == 3)
            {
                // For PLN3, copy each channel separately - channels are at actualInputHeight intervals in Mat
                for(int c = 0; c < 3; c++)
                {
                    for(int j = 0; j < outputHeight; j++)
                    {
                        Rpp8u *d_outputRowTemp = d_output_offsetted + (c * dstDescPtr[i].strides.cStride + j * dstDescPtr[i].strides.hStride) * outElementSize;
                        Rpp8u *outputRowTemp = outputTemp + (c * outputHeight + j) * outputVec[i].step[0];
                        CHECK_RETURN_STATUS(hipMemcpy(outputRowTemp, d_outputRowTemp, rowSizeInBytes, hipMemcpyDeviceToHost));
                    }
                }
            }
            else
            {
                // For PKD3 and PLN1, copy rows sequentially
                for(int j = 0; j < outputHeight; j++)
                {
                    Rpp8u *d_outputRowTemp = d_output_offsetted + j * dstDescPtr[i].strides.hStride * outElementSize;
                    Rpp8u *outputRowTemp = outputTemp + j * outputVec[i].step[0];
                    CHECK_RETURN_STATUS(hipMemcpy(outputRowTemp, d_outputRowTemp, rowSizeInBytes, hipMemcpyDeviceToHost));
                }
            }

            // Free device memory for this image
            CHECK_RETURN_STATUS(hipFree(d_input));
            if (dualInputCase)
                CHECK_RETURN_STATUS(hipFree(d_inputSecond));
            CHECK_RETURN_STATUS(hipFree(d_output));

            wallTime = endWallTime - startWallTime;
            maxWallTime = max(maxWallTime, wallTime);
            minWallTime = min(minWallTime, wallTime);
            avgWallTime += wallTime;
        }
    }

    if (testType == UNIT_TEST) // unit test mode
    {
        cout <<"\n\n";
        cout << "GPU Backend Wall Time: " << avgWallTime * 1000 / (numRuns * noOfImages) <<" ms/image";
        if(testCase != CROP && testCase != RESIZE)
        {
            for(int i = 0; i < noOfImages; i++)
            {
                // Use actual dimensions, not padded descriptor width
                dstImgSizes[i].width = actualInputWidth[i];
                dstImgSizes[i].height = actualInputHeight[i];
            }
        }

        // QA mode - compare outputs with golden outputs
        if (qaFlag && (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F32_TO_F32) && !randomOutputCase && !nonQACase)
        {
            string interpolationTypeName = "";
            string noiseTypeName = "";
            if (interpolationTypeCase)
                interpolationTypeName = get_interpolation_type(additionalParam, interpolationType);
            if (noiseTypeCase)
                noiseTypeName = get_noise_type(additionalParam);

            compare_output_single_image(outputVec, srcDescPtr, dstDescPtr, testCaseName, dstImgSizes, noOfImages, interpolationTypeName, noiseTypeName, additionalParam, testCase, dst, scriptPath);
        }

        saveBatchOutput(dst, noOfImages, outputVec, dstDescPtr, dstImgSizes);
    }

    if(testCase == BLEND)
        CHECK_RETURN_STATUS(hipHostFree(alpha));
    if(testCase == BRIGHTNESS)
    {
        CHECK_RETURN_STATUS(hipHostFree(alpha));
        CHECK_RETURN_STATUS(hipHostFree(beta));
    }
    CHECK_RETURN_STATUS(hipHostFree(roi));
    
    rppDestroy(handle, backend);
    if(testType == PERFORMANCE_TEST) // performance test mode
    {
        // Display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns);
        cout << fixed << "\n Running : "<< func << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }

    return 0;
}
