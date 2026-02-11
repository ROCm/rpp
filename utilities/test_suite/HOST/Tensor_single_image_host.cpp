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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <fstream>

using namespace cv;
using namespace std;

cv::Mat convert_pkd3_to_pln3(const cv::Mat& srcPacked)
{
    int width = srcPacked.cols;
    int height = srcPacked.rows;
    
    cv::Mat dstPlanar(height * 3, width, CV_MAKETYPE(srcPacked.depth(), 1));

    if (srcPacked.depth() == CV_8U || srcPacked.depth() == CV_8S)
    {
        for (int y = 0; y < height; y++)
        {
            const uchar* srcRow = srcPacked.ptr<uchar>(y);
            uchar* dstR = dstPlanar.ptr<uchar>(y);
            uchar* dstG = dstPlanar.ptr<uchar>(y + height);
            uchar* dstB = dstPlanar.ptr<uchar>(y + 2 * height);

            for (int x = 0; x < width; x++)
            {
                uchar val0 = *srcRow++;
                uchar val1 = *srcRow++;
                uchar val2 = *srcRow++;

                *dstR++ = val0;
                *dstG++ = val1;
                *dstB++ = val2;
            }
        }
    }
    else if (srcPacked.depth() == CV_32F || srcPacked.depth() == CV_16F)
    {
        for (int y = 0; y < height; y++)
        {
            const float* srcRow = srcPacked.ptr<float>(y);

            float* dstR = dstPlanar.ptr<float>(y);
            float* dstG = dstPlanar.ptr<float>(y + height);
            float* dstB = dstPlanar.ptr<float>(y + 2 * height);

            for (int x = 0; x < width; x++)
            {
                float val0 = *srcRow++;
                float val1 = *srcRow++;
                float val2 = *srcRow++;

                *dstR++ = val0;
                *dstG++ = val1;
                *dstB++ = val2;
            }
        }
    }

    return dstPlanar;
}

cv::Mat convert_pln3_to_pkd3(const cv::Mat& srcPlanar, int height, int width)
{
    cv::Mat dstPacked(height, width, CV_MAKETYPE(srcPlanar.depth(), 3));

    if (srcPlanar.depth() == CV_8U)
    {
        for (int y = 0; y < height; y++)
        {
            const uchar* srcR = srcPlanar.ptr<uchar>(y);
            const uchar* srcG = srcPlanar.ptr<uchar>(y + height);
            const uchar* srcB = srcPlanar.ptr<uchar>(y + 2 * height);

            uchar* dstRow = dstPacked.ptr<uchar>(y);

            for (int x = 0; x < width; x++)
            {
                *dstRow++ = *srcR++;
                *dstRow++ = *srcG++;
                *dstRow++ = *srcB++;
            }
        }
    }
    else if (srcPlanar.depth() == CV_32F)
    {
        for (int y = 0; y < height; y++)
        {
            const float* srcR = srcPlanar.ptr<float>(y);
            const float* srcG = srcPlanar.ptr<float>(y + height);
            const float* srcB = srcPlanar.ptr<float>(y + 2 * height);

            float* dstRow = dstPacked.ptr<float>(y);

            for (int x = 0; x < width; x++)
            {
                *dstRow++ = *srcR++;
                *dstRow++ = *srcG++;
                *dstRow++ = *srcB++;
            }
        }
    }

    return dstPacked;
}

vector<Mat> loadBatchImages(const string& directory, int& noOfImages, RpptLayout layoutType, bool isColor, int bitDepthMode, float conversionFactor)
{
    vector<Mat> images;
    DIR* dir;
    struct dirent* entry;

    if ((dir = opendir(directory.c_str())) == NULL) {
        cerr << "Could not open directory: " << directory << endl;
        return images;
    }

    while ((entry = readdir(dir)) != NULL) {
        string filename = entry->d_name;
        if (filename == "." || filename == "..") continue;
        string ext = filename.substr(filename.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != "jpg" && ext != "jpeg" && ext != "png" && ext != "bmp") continue;

        string filePath = directory + "/" + filename;

        Mat processedImg = imread(filePath, isColor ? IMREAD_COLOR : IMREAD_GRAYSCALE);
        if (processedImg.empty()) continue;

        Mat finalImg;
        if (bitDepthMode == U8_TO_F32 || bitDepthMode == F32_TO_F32)
            processedImg.convertTo(finalImg, CV_32F, conversionFactor);
        else if (bitDepthMode == U8_TO_F16 || bitDepthMode == F16_TO_F16)
            processedImg.convertTo(finalImg, CV_16F, conversionFactor);
        else if (bitDepthMode == I8_TO_I8 || bitDepthMode == U8_TO_I8)
            processedImg.convertTo(finalImg, CV_8S, 1.0, -128.0);
        else
            finalImg = processedImg;

        images.push_back(finalImg);
    }
    closedir(dir);
    noOfImages = images.size();
    return images;
}

void initializeROI(const vector<Mat>& imgs, vector<RpptROI>& rois, vector<RpptDesc>& descPtr, int* roiList)
{
    int batchSize = imgs.size();
    bool useCustomROI = (roiList[2] != 0 && roiList[3] != 0);

    for (int i = 0; i < batchSize; ++i)
    {
        if (useCustomROI)
        {
            rois[i].xywhROI.xy.x = roiList[0];
            rois[i].xywhROI.xy.y = roiList[1];
            rois[i].xywhROI.roiWidth = roiList[2];
            rois[i].xywhROI.roiHeight = roiList[3];
        }
        else
        {
            rois[i].xywhROI.xy.x = 0;
            rois[i].xywhROI.xy.y = 0;
            rois[i].xywhROI.roiWidth = descPtr[i].w;
            rois[i].xywhROI.roiHeight = descPtr[i].h;
        }
    }
}

void initializeDescriptors(const vector<Mat>& imgs, vector<RpptDesc>& descPtr, int channel)
{
    int batchSize = imgs.size();

    for (int i = 0; i < batchSize; ++i)
    {
        const Mat& img = imgs[i];

        int realHeight = img.rows;
        int realWidth = img.cols;

        descPtr[i].h = img.rows;
        descPtr[i].w = img.cols;
        descPtr[i].c = channel;
        descPtr[i].n = 1;

        if (descPtr[i].layout == RpptLayout::NHWC)
        {
            descPtr[i].strides.nStride = realHeight * realWidth * channel;
            descPtr[i].strides.hStride = realWidth * channel;
            descPtr[i].strides.wStride = channel;
            descPtr[i].strides.cStride = 1;
        }
        else
        {
            descPtr[i].strides.nStride = realHeight * realWidth * channel;
            descPtr[i].strides.hStride = realWidth;
            descPtr[i].strides.wStride = 1;
            descPtr[i].strides.cStride = realHeight * realWidth;
        }
    }
}

inline void set_descriptor_data_type_name(int BitDepthTestMode, string &funcName)
{
    if (BitDepthTestMode == U8_TO_U8) funcName += "_u8_";
    else if (BitDepthTestMode == F16_TO_F16) funcName += "_f16_";
    else if (BitDepthTestMode == F32_TO_F32) funcName += "_f32_";
    else if (BitDepthTestMode == U8_TO_F16) funcName += "_u8_f16_";
    else if (BitDepthTestMode == U8_TO_F32) funcName += "_u8_f32_";
    else if (BitDepthTestMode == I8_TO_I8) funcName += "_i8_";
    else if (BitDepthTestMode == U8_TO_I8) funcName += "_u8_i8_";
}

inline void set_descriptor_layout(vector<RpptDesc>& srcDescs, vector<RpptDesc>& dstDescs, int layoutType, bool pln1OutTypeCase, int outputFormatToggle, int noOfImages)
{
    for(int i = 0; i < noOfImages; i++)
    {
        if(layoutType == 0) srcDescs[i].layout = RpptLayout::NHWC;
        else srcDescs[i].layout = RpptLayout::NCHW;

        RpptLayout dstLayout;

        if (layoutType == 0) {
            if (pln1OutTypeCase) { dstLayout = RpptLayout::NCHW;}
            else { dstLayout = (outputFormatToggle == 0) ? RpptLayout::NHWC : RpptLayout::NCHW; }
        } else if (layoutType == 1) {
            if (pln1OutTypeCase) { dstLayout = RpptLayout::NCHW; }
            else { dstLayout = (outputFormatToggle == 0) ? RpptLayout::NCHW : RpptLayout::NHWC; }
        } else {
            dstLayout = RpptLayout::NCHW;
        }

        dstDescs[i].layout = dstLayout;
    }
}

inline void set_descriptor_data_type(int BitDepthTestMode, vector<RpptDesc>& srcDescPtr, vector<RpptDesc>& dstDescPtr, int noOfImages)
{
    for(int i = 0; i < noOfImages; i++)
    {
        if (BitDepthTestMode == U8_TO_U8) {
            srcDescPtr[i].dataType = RpptDataType::U8;
            dstDescPtr[i].dataType = RpptDataType::U8;
        } else if (BitDepthTestMode == F16_TO_F16) {
            srcDescPtr[i].dataType = RpptDataType::F16;
            dstDescPtr[i].dataType = RpptDataType::F16;
        } else if (BitDepthTestMode == F32_TO_F32) {
            srcDescPtr[i].dataType = RpptDataType::F32;
            dstDescPtr[i].dataType = RpptDataType::F32;
        } else if (BitDepthTestMode == U8_TO_F16) {
            srcDescPtr[i].dataType = RpptDataType::U8;
            dstDescPtr[i].dataType = RpptDataType::F16;
        } else if (BitDepthTestMode == U8_TO_F32) {
            srcDescPtr[i].dataType = RpptDataType::U8;
            dstDescPtr[i].dataType = RpptDataType::F32;
        } else if (BitDepthTestMode == I8_TO_I8) {
            srcDescPtr[i].dataType = RpptDataType::I8;
            dstDescPtr[i].dataType = RpptDataType::I8;
        } else if (BitDepthTestMode == U8_TO_I8) {
            srcDescPtr[i].dataType = RpptDataType::U8;
            dstDescPtr[i].dataType = RpptDataType::I8;
        }
    }
}

int get_cv_type(RpptDataType dataType, int channels)
{
    switch (dataType)
    {
        case RpptDataType::U8:  return CV_MAKETYPE(CV_8U, channels);
        case RpptDataType::I8:  return CV_MAKETYPE(CV_8S, channels);
        case RpptDataType::F16: return CV_MAKETYPE(CV_16F, channels);
        case RpptDataType::F32: return CV_MAKETYPE(CV_32F, channels);
        default: return -1;
    }
}

void saveBatchOutput(const string& dstDir, int noOfImages, const vector<Mat>& outputVec, const vector<RpptDesc>& dstDescPtr, RpptImagePatch *dstImgSizes)
{
    mkdir(dstDir.c_str(), 0700);

    for (int i = 0; i < noOfImages; i++)
    {
        string separator = (dstDir.back() == '/') ? "" : "/";
        string currentFileName = dstDir + separator + to_string(i) + ".jpg";

        // Skip empty or invalid output
        if (outputVec[i].empty())
        {
            cerr << "\n[Error] Output image " << i << " is empty";
            continue;
        }

        // Validate and clamp dimensions to Mat bounds
        int matCols = outputVec[i].cols;
        int matRows = outputVec[i].rows;
        int extractWidth = std::min((int)dstImgSizes[i].width, matCols);
        int extractHeight = std::min((int)dstImgSizes[i].height, matRows);
        
        // Extract the actual output region based on dstImgSizes
        Mat tempImg;
        if ((dstDescPtr[i].c == 3) && (dstDescPtr[i].layout == RpptLayout::NCHW))
        {
            // For planar layout, extract region for all 3 channels
            int planarHeight = std::min(extractHeight * 3, matRows);
            tempImg = outputVec[i](Rect(0, 0, extractWidth, planarHeight));
        }
        else
        {
            // For packed layout or single channel
            tempImg = outputVec[i](Rect(0, 0, extractWidth, extractHeight));
        }

        Mat saveImg;
        if (tempImg.depth() == CV_32F || tempImg.depth() == CV_16F)
        {
            tempImg.convertTo(saveImg, CV_8U, 255.0);
        }
        else if (tempImg.depth() == CV_8S)
        {
            tempImg.convertTo(saveImg, CV_8U, 1.0, 128.0);
        }
        else
        {
            saveImg = tempImg;
        }

        if (!saveImg.empty())
        {
            imwrite(currentFileName, saveImg);
            cout << "\nSaved: " << currentFileName;
        }
        else
        {
            cerr << "\n[Error] Failed to create output image " << i;
        }
    }
}

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

    string funcType = set_function_type(layoutType, pln1OutTypeCase, outputFormatToggle, "HOST");
    string func = funcName;
    set_descriptor_data_type_name(BitDepthTestMode, func);
    func += funcType;
    if (kernelSizeCase) func += "_kernelSize" + std::to_string(additionalParam);
    if(!qaFlag) dst += "/" + func;

    int noOfImages = 0, missingFuncFlag = 0;
    Rpp32f conversionFactor = 1.0f / 255.0;
    bool isColor = (layoutType != 2);
    RpptLayout srcLayoutEnum = (layoutType == 0) ? RpptLayout::NHWC : RpptLayout::NCHW;
    vector<Mat> inputVec = loadBatchImages(src, noOfImages, srcLayoutEnum, isColor, BitDepthTestMode, conversionFactor);
    vector<Mat> inputVecSecond;
    if (dualInputCase)
        inputVecSecond = loadBatchImages(srcSecond, noOfImages, srcLayoutEnum, isColor, BitDepthTestMode, conversionFactor);

    if (noOfImages == 0) { cerr << "No images found!"; return -1; }
    if (noOfImages < batchSize) {
        for (int i = noOfImages; i < batchSize; i++)
        {
            inputVec.push_back(inputVec[noOfImages - 1]);
            if (dualInputCase)
                inputVecSecond.push_back(inputVecSecond[noOfImages - 1]);
        }
        noOfImages = batchSize;
    }

    vector<RpptDesc> srcDescPtr(noOfImages), dstDescPtr(noOfImages);
    vector<RpptROI> roi(noOfImages);
    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    RpptInterpolationType interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
    int inputChannel = set_input_channels(layoutType);
    int outputChannel = inputChannel;
    if(pln1OutTypeCase)
        outputChannel = 1;
    set_descriptor_layout(srcDescPtr, dstDescPtr, layoutType, pln1OutTypeCase, outputFormatToggle, noOfImages);
    initializeDescriptors(inputVec, srcDescPtr, inputChannel);
    initializeDescriptors(inputVec, dstDescPtr, outputChannel);
    set_descriptor_data_type(BitDepthTestMode, srcDescPtr, dstDescPtr, noOfImages);
    initializeROI(inputVec, roi, srcDescPtr, roiList);

    vector<Mat> outputVec(noOfImages);
    for (int i = 0; i < noOfImages; i++)
    {
        int channels = dstDescPtr[i].c;
        if (dstDescPtr[i].layout == RpptLayout::NCHW)
        {
            int planarCvType = get_cv_type(dstDescPtr[i].dataType, 1);
            if (planarCvType == -1) { cerr << "Unsupported type for Image " << i << endl; continue; }
            outputVec[i] = Mat(dstDescPtr[i].h * channels, dstDescPtr[i].w, planarCvType);
        }
        else
        {
            int packedCvType = get_cv_type(dstDescPtr[i].dataType, channels);
            if (packedCvType == -1) { cerr << "Unsupported type for Image " << i << endl; continue; }
            outputVec[i] = Mat(dstDescPtr[i].h, dstDescPtr[i].w, packedCvType);
        }
    }
    if (isColor && srcDescPtr[0].layout == RpptLayout::NCHW)
    {
        for(int i = 0; i < noOfImages; i++)
            inputVec[i] = convert_pkd3_to_pln3(inputVec[i]);
        if (dualInputCase)
            for(int i = 0; i < noOfImages; i++)
                inputVecSecond[i] = convert_pkd3_to_pln3(inputVecSecond[i]);
    }

    Rpp32u numThreads = noOfImages;
    rppHandle_t handle;
    RppBackend backend = RppBackend::RPP_HOST_BACKEND;
    rppCreate(&handle, noOfImages, numThreads, nullptr, backend);
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double startCpuTime, wallTime;
    string testCaseName;
    
    // Initialize dstImgSizes for all images to their original dimensions
    RpptImagePatch dstImgSizes[noOfImages];
    for (int i = 0; i < noOfImages; i++)
    {
        dstImgSizes[i].width = dstDescPtr[i].w;
        dstImgSizes[i].height = dstDescPtr[i].h;
    }
    
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
                    for (int i = 0; i < noOfImages; ++i) {
                        errorCodeCapture = rppt_brightness_host(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], &alpha, &beta, &roi[i], RpptRoiType::XYWH, handle);
                    }
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
                    for (int i = 0; i < noOfImages; ++i) {
                        errorCodeCapture = rppt_box_filter_host(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], kernelSize, borderType, &roi[i], RpptRoiType::XYWH, handle);
                    }
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
                    for (int i = 0; i < noOfImages; ++i) {
                        errorCodeCapture = rppt_flip_host(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], &horizontalFlag, &verticalFlag, &roi[i], RpptRoiType::XYWH, handle);
                    }
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
                    dstImgSizes[i].width = roi[i].xywhROI.roiWidth / 2;
                    dstImgSizes[i].height = roi[i].xywhROI.roiHeight / 2;
                }

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                {
                    omp_set_dynamic(0);
                    #pragma omp parallel for num_threads(numThreads)
                    for (int i = 0; i < noOfImages; ++i) {
                        errorCodeCapture = rppt_resize_host(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], &dstImgSizes[i], interpolationType, &roi[i], RpptRoiType::XYWH, handle);
                    }
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case CROP:
            {
                testCaseName = "crop";
                int roiHeightList[noOfImages], roiWidthList[noOfImages];
                bool invalidROI = (roiList[0] == 0 && roiList[1] == 0 && roiList[2] == 0 && roiList[3] == 0);

                for(int i = 0; i < noOfImages ; i++)
                {
                    if(invalidROI)
                    {
                        roiList[0] = 10;
                        roiList[1] = 10;
                        roiWidthList[i] = roi[i].xywhROI.roiWidth / 2;
                        roiHeightList[i] = roi[i].xywhROI.roiHeight / 2;
                    }
                    else
                    {
                        roiWidthList[i] = roiList[2];
                        roiHeightList[i] = roiList[3];
                    }
                }

                for (int i = 0; i < noOfImages; i++)
                {
                    roi[i].xywhROI.xy.x = roiList[0];
                    roi[i].xywhROI.xy.y = roiList[1];
                    roi[i].xywhROI.roiWidth = roiWidthList[i];
                    roi[i].xywhROI.roiHeight = roiHeightList[i];
                    // Update dstImgSizes for crop
                    dstImgSizes[i].width = roiWidthList[i];
                    dstImgSizes[i].height = roiHeightList[i];
                }

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                {
                    omp_set_dynamic(0);
                    #pragma omp parallel for num_threads(numThreads)
                    for (int i = 0; i < noOfImages; ++i) 
                        errorCodeCapture = rppt_crop_host(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], &roi[i], RpptRoiType::XYWH, handle);
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

        if ((dstDescPtr[0].c == 3) && (dstDescPtr[0].layout == RpptLayout::NCHW))
        {
            vector<Mat> outputVecPkd3(noOfImages);
            for(int i = 0; i < noOfImages; i++)
                outputVecPkd3[i] = convert_pln3_to_pkd3(outputVec[i], dstImgSizes[i].height, dstImgSizes[i].width);
            saveBatchOutput(dst, noOfImages, outputVecPkd3, dstDescPtr, dstImgSizes);
        }
        else
        {
            saveBatchOutput(dst, noOfImages, outputVec, dstDescPtr, dstImgSizes);
        }
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
