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
#include "../rpp_test_suite_image.h"
#include "../rpp_test_suite_single_image_utils.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <fstream>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    const int MIN_ARG_COUNT = 19;

    char *src = argv[1];
    char *srcSecond = argv[2];
    string dst = argv[3];

    int BitDepthTestMode = atoi(argv[4]);
    unsigned int outputFormatToggle = atoi(argv[5]);
    int testCase = atoi(argv[6]);
    int numRuns = atoi(argv[8]);
    int testType = atoi(argv[9]);
    int layoutType = atoi(argv[10]);
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

    if (argc < MIN_ARG_COUNT)
    {
        cout << "\nImproper Usage! Needs all arguments!\n";
        return 1;
    }

    string funcName = augmentationMap[testCase];
    if (funcName.empty())
    {
        if (testType == UNIT_TEST) cout << "\ncase " << testCase << " is not supported\n";
        return -1;
    }

    // Determine the type of function to be used based on the specified layout type
    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    RpptInterpolationType interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
    string funcType = set_function_type(layoutType, pln1OutTypeCase, outputFormatToggle, "HOST");
    string func = funcName;
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

    int noOfImages = 0, missingFuncFlag = 0;
    Rpp32f conversionFactor = 1.0f / 255.0;
    bool isColor = (layoutType != 2);
    RpptLayout srcLayoutEnum = (layoutType == 0) ? RpptLayout::NHWC : RpptLayout::NCHW;
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

    convertBatchBitDepth(inputVec, BitDepthTestMode, conversionFactor);
    convertBatchBitDepth(inputVecSecond, BitDepthTestMode, conversionFactor);
    if (noOfImages < batchSize) {
        for (int i = noOfImages; i < batchSize; i++)
            inputVec.push_back(inputVec[noOfImages - 1]);
            if (dualInputCase)
                inputVecSecond.push_back(inputVecSecond[noOfImages - 1]);
        noOfImages = batchSize;
    }

    vector<RpptDesc> srcDescPtr(noOfImages), dstDescPtr(noOfImages);
    vector<RpptROI> roi(noOfImages);
    memset(roi.data(), 0, noOfImages * sizeof(RpptROI));
    RpptImagePatch dstImgSizes[noOfImages];

    int inputChannel = set_input_channels(layoutType);
    int outputChannel = inputChannel;
    if(pln1OutTypeCase)
        outputChannel = 1;
    set_descriptor_layout(srcDescPtr, dstDescPtr, layoutType, pln1OutTypeCase, outputFormatToggle, noOfImages);
    set_descriptor_data_type(BitDepthTestMode, srcDescPtr, dstDescPtr, noOfImages);

    initializeDescriptors(inputVec, srcDescPtr, inputChannel);
    initializeDescriptors(inputVec, dstDescPtr, outputChannel);
    int roiHeightList[noOfImages], roiWidthList[noOfImages];
    initializeROI(inputVec, roi, dstDescPtr, roiList, roiHeightList, roiWidthList);

    // Track actual image dimensions (not padded descriptor width)
    vector<Mat> outputVec(noOfImages);
    vector<Rpp32u> actualInputWidth(noOfImages), actualInputHeight(noOfImages);
    for (int i = 0; i < noOfImages; i++)
    {
        // Store actual dimensions before any padding
        actualInputWidth[i] = inputVec[i].cols;
        actualInputHeight[i] = inputVec[i].rows;
        
        int channels = dstDescPtr[i].c;
        if (dstDescPtr[i].layout == RpptLayout::NCHW)
        {
            int planarCvType = get_cv_type(dstDescPtr[i].dataType, 1);
            if (planarCvType == -1) { cerr << "Unsupported type for Image " << i << endl; continue; }
            // Use actual dimensions, not padded descriptor width
            outputVec[i] = Mat(actualInputHeight[i] * channels, actualInputWidth[i], planarCvType);
        }
        else
        {
            int packedCvType = get_cv_type(dstDescPtr[i].dataType, channels);
            if (packedCvType == -1) { cerr << "Unsupported type for Image " << i << endl; continue; }
            // Use actual dimensions, not padded descriptor width
            outputVec[i] = Mat(actualInputHeight[i], actualInputWidth[i], packedCvType);
        }
    }
    if (isColor && srcDescPtr[0].layout == RpptLayout::NCHW)
    {
        for(int i = 0; i < noOfImages; i++)
        {
            inputVec[i] = convert_pkd3_to_pln3(inputVec[i]);
            if (dualInputCase)
                inputVecSecond[i] = convert_pkd3_to_pln3(inputVecSecond[i]);
        }
    }

    Rpp32u numThreads = noOfImages;
    rppHandle_t handle;
    RppBackend backend = RppBackend::RPP_HOST_BACKEND;
    rppCreate(&handle, noOfImages, numThreads, nullptr, backend);
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double startCpuTime, wallTime;
    string testCaseName;
    
    cout << "\nRunning " << func << " " << numRuns << " times (each time with a batch size of " << batchSize << " images) and computing mean statistics...";
    for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
    {
        RppStatus errorCodeCapture = RPP_SUCCESS;
        double startWallTime, endWallTime;
        switch (testCase)
        {
            case BRIGHTNESS:
            {
                testCaseName = "brightness";
                Rpp32f alpha = 1.75f;
                Rpp32f beta = 50.0f;

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                {
                    omp_set_dynamic(0);
                    #pragma omp parallel for num_threads(numThreads)
                    for (int i = 0; i < noOfImages; ++i)
                        errorCodeCapture = rppt_brightness(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], &alpha, &beta, &roi[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case BLEND:
            {
                testCaseName = "blend";
                Rpp32f alpha = 0.4;

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                {
                    omp_set_dynamic(0);
                    #pragma omp parallel for num_threads(numThreads)
                    for (int i = 0; i < noOfImages; ++i)
                        errorCodeCapture = rppt_blend(inputVec[i].data, inputVecSecond[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], &alpha, &roi[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case FLIP:
            {
                testCaseName = "flip";

                Rpp32u horizontalFlag = 1;
                Rpp32u verticalFlag = 0;

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                {
                    omp_set_dynamic(0);
                    #pragma omp parallel for num_threads(numThreads)
                    for (int i = 0; i < noOfImages; ++i)
                        errorCodeCapture = rppt_flip(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], &horizontalFlag, &verticalFlag, &roi[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case RESIZE:
            {
                testCaseName = "resize";

                for (int i = 0; i < noOfImages; i++)
                {
                    dstImgSizes[i].width = roi[i].xywhROI.roiWidth = roi[i].xywhROI.roiWidth / 2;
                    dstImgSizes[i].height = roi[i].xywhROI.roiHeight = roi[i].xywhROI.roiHeight / 2;
                }

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                {
                    omp_set_dynamic(0);
                    #pragma omp parallel for num_threads(numThreads)
                    for (int i = 0; i < noOfImages; ++i)
                        errorCodeCapture = rppt_resize(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], &dstImgSizes[i], interpolationType, &roi[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case CROP:
            {
                testCaseName = "crop";

                for(int i = 0; i < noOfImages ; i++)
                {
                    roi[i].xywhROI.xy.x = roiList[0];
                    roi[i].xywhROI.xy.y = roiList[1];
                    dstImgSizes[i].width = roi[i].xywhROI.roiWidth = roiWidthList[i];
                    dstImgSizes[i].height = roi[i].xywhROI.roiHeight = roiHeightList[i];

                }

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                {
                    omp_set_dynamic(0);
                    #pragma omp parallel for num_threads(numThreads)
                    for (int i = 0; i < noOfImages; ++i) 
                        errorCodeCapture = rppt_crop(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], &roi[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND);
                }
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
                startCpuTime = clock();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                {
                    omp_set_dynamic(0);
                    #pragma omp parallel for num_threads(numThreads)
                    for (int i = 0; i < noOfImages; ++i)
                        errorCodeCapture = rppt_box_filter(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], kernelSize, borderType, &roi[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND);
                }
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
                startCpuTime = clock();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                {
                    omp_set_dynamic(0);
                    #pragma omp parallel for num_threads(numThreads)
                    for (int i = 0; i < noOfImages; ++i)
                        errorCodeCapture = rppt_median_filter(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], kernelSize, borderType, &roi[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND);
                }
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
        endWallTime = omp_get_wtime();
        wallTime = endWallTime - startWallTime;

        if (missingFuncFlag == 1)
        {
            cout << "\nThe functionality " << " doesn't yet exist in RPP\n";
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        maxWallTime = std::max(maxWallTime, wallTime);
        minWallTime = std::min(minWallTime, wallTime);
        avgWallTime += wallTime;
    }

    wallTime *= 1000;

    if (testType == UNIT_TEST)
    {
        cout <<"\n\n";
        cout <<"CPU Backend Wall Time: "<< wallTime <<" ms/batch";

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

    rppDestroy(handle, backend);

    if(testType == PERFORMANCE_TEST)
    {
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns);
        cout << fixed << "\n Running : "<< func << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }

    cout << endl;

    return 0;
}
