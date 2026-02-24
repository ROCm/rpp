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
    else if (srcPacked.depth() == CV_32F)
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
    else if (srcPacked.depth() == CV_16F)
    {
        for (int y = 0; y < height; y++)
        {
            const cv::float16_t* srcRow = srcPacked.ptr<cv::float16_t>(y);

            cv::float16_t* dstR = dstPlanar.ptr<cv::float16_t>(y);
            cv::float16_t* dstG = dstPlanar.ptr<cv::float16_t>(y + height);
            cv::float16_t* dstB = dstPlanar.ptr<cv::float16_t>(y + 2 * height);

            for (int x = 0; x < width; x++)
            {
                cv::float16_t val0 = *srcRow++;
                cv::float16_t val1 = *srcRow++;
                cv::float16_t val2 = *srcRow++;

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

    if (srcPlanar.depth() == CV_8U || srcPlanar.depth() == CV_8S)
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
    else if (srcPlanar.depth() == CV_16F)
    {
        for (int y = 0; y < height; y++)
        {
            const cv::float16_t* srcR = srcPlanar.ptr<cv::float16_t>(y);
            const cv::float16_t* srcG = srcPlanar.ptr<cv::float16_t>(y + height);
            const cv::float16_t* srcB = srcPlanar.ptr<cv::float16_t>(y + 2 * height);

            cv::float16_t* dstRow = dstPacked.ptr<cv::float16_t>(y);

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

vector<Mat> loadBatchImages_jpegd(const string& directory, int& noOfImages, bool isColor)
{
    vector<Mat> images;
    DIR* dir;
    struct dirent* entry;
    tjhandle m_jpegDecompressor = tjInitDecompress();

    if ((dir = opendir(directory.c_str())) == NULL) {
        cerr << "Could not open directory: " << directory << endl;
        return images;
    }

    vector<string> image_names;
    while ((entry = readdir(dir)) != NULL) {
        string filename = entry->d_name;
        if (filename == "." || filename == "..") continue;
        string ext = filename.substr(filename.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != "jpg" && ext != "jpeg") continue;
        image_names.push_back(directory + "/" + filename);
    }
    closedir(dir);
    std::sort(image_names.begin(), image_names.end());

    for (const auto& inputImagePath : image_names)
    {
        FILE* fp = fopen(inputImagePath.c_str(), "rb");
        if(!fp) {
            std::cerr << "\n unable to open file : "<< inputImagePath;
            continue;
        }
        fseek(fp, 0, SEEK_END);
        long jpegSize = ftell(fp);
        rewind(fp);
        
        unsigned char* jpegBuf = (unsigned char*)malloc(jpegSize);
        if (fread(jpegBuf, 1, jpegSize, fp) != jpegSize) {
             std::cerr << "\n File read incomplete: " << inputImagePath;
             free(jpegBuf);
             fclose(fp);
             continue;
        }
        fclose(fp);

        int width, height, subsamp, color_space;
        if(tjDecompressHeader3(m_jpegDecompressor, jpegBuf, jpegSize, &width, &height, &subsamp, &color_space) != 0) {
            std::cerr << "\n Jpeg image decode failed in tjDecompressHeader3 for " << inputImagePath;
            free(jpegBuf);
            continue;
        }

        Mat decodedImage;
        if(isColor)
        {
            decodedImage = Mat::zeros(height, width, CV_8UC3);
            if(tjDecompress2(m_jpegDecompressor, jpegBuf, jpegSize, decodedImage.data, width, 
                             (int)decodedImage.step, height, TJPF_RGB, TJFLAG_ACCURATEDCT) != 0) {
                std::cerr << "\n Jpeg image decode failed for " << inputImagePath;
                free(jpegBuf);
                continue;
            }
        }
        else
        {
            decodedImage = Mat::zeros(height, width, CV_8UC1);
            if(tjDecompress2(m_jpegDecompressor, jpegBuf, jpegSize, decodedImage.data, width, 
                             (int)decodedImage.step, height, TJPF_GRAY, 0) != 0) {
                std::cerr << "\n Jpeg image decode failed for " << inputImagePath;
                free(jpegBuf);
                continue;
            }
        }
        images.push_back(decodedImage);
        free(jpegBuf);
    }

    tjDestroy(m_jpegDecompressor);
    noOfImages = images.size();
    return images;
}

vector<Mat> loadBatchImages_cv(const string& directory, int& noOfImages, bool isColor)
{
    vector<Mat> images;
    DIR* dir;
    struct dirent* entry;

    if ((dir = opendir(directory.c_str())) == NULL) {
        cerr << "Could not open directory: " << directory << endl;
        return images;
    }

    vector<string> filenames;
    while ((entry = readdir(dir)) != NULL) {
        string filename = entry->d_name;
        if (filename == "." || filename == "..") continue;
        string ext = filename.substr(filename.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != "jpg" && ext != "jpeg" && ext != "png" && ext != "bmp") continue;
        filenames.push_back(directory + "/" + filename);
    }
    closedir(dir);
    std::sort(filenames.begin(), filenames.end());

    for (const auto& filePath : filenames) {
        Mat temp = imread(filePath, isColor ? IMREAD_COLOR : IMREAD_GRAYSCALE);
        if (temp.empty()) continue;
        Mat img;
        if (isColor && temp.depth() == CV_8U)
            cvtColor(temp, img, COLOR_BGR2RGB);
        else
            img = temp;
        if (!img.isContinuous())
            img = img.clone();
        images.push_back(img);
    }
    noOfImages = images.size();
    return images;
}

void convertBatchBitDepth(vector<Mat>& images, int bitDepthMode, float conversionFactor)
{
    if (bitDepthMode == U8_TO_U8) 
        return;

    for (size_t i = 0; i < images.size(); i++)
    {
        Mat finalImg;
        
        if (bitDepthMode == U8_TO_F32 || bitDepthMode == F32_TO_F32)
            images[i].convertTo(finalImg, CV_32F, conversionFactor);
        else if (bitDepthMode == U8_TO_F16 || bitDepthMode == F16_TO_F16)
            images[i].convertTo(finalImg, CV_16F, conversionFactor);
        else if (bitDepthMode == I8_TO_I8 || bitDepthMode == U8_TO_I8)
            images[i].convertTo(finalImg, CV_8S, 1.0, -128.0);
        else
            finalImg = images[i];

        images[i] = finalImg;
    }
}

void initializeROI(const vector<Mat>& imgs, vector<RpptROI>& rois, vector<RpptDesc>& descPtr, int* roiList, int* roiHeightList, int* roiWidthList)
{
    int batchSize = imgs.size();
    bool invalidROI = (roiList[0] == 0 && roiList[1] == 0 && roiList[2] == 0 && roiList[3] == 0);

    for (int i = 0; i < batchSize; ++i)
    {
        rois[i].xywhROI.xy.x = 0;
        rois[i].xywhROI.xy.y = 0;
        rois[i].xywhROI.roiWidth = imgs[i].cols;
        rois[i].xywhROI.roiHeight = imgs[i].rows;
        if (invalidROI)
        {
            roiList[0] = 10;
            roiList[1] = 10;
            roiWidthList[i] = rois[i].xywhROI.roiWidth / 2;
            roiHeightList[i] = rois[i].xywhROI.roiHeight / 2;
        }
        else
        {
            roiWidthList[i] = roiList[2];
            roiHeightList[i] = roiList[3];
        }
    }
}

void initializeDescriptors(const vector<Mat>& imgs, vector<RpptDesc>& descPtr, int channel)
{
    int batchSize = imgs.size();

    for (int i = 0; i < batchSize; ++i)
    {
        const Mat& img = imgs[i];

        descPtr[i].h = img.rows;
        descPtr[i].w = img.cols;
        descPtr[i].c = channel;
        descPtr[i].n = 1;

        if (descPtr[i].layout == RpptLayout::NHWC)
        {
            descPtr[i].strides.nStride = descPtr[i].h * descPtr[i].w * channel;
            descPtr[i].strides.hStride = descPtr[i].w * channel;
            descPtr[i].strides.wStride = channel;
            descPtr[i].strides.cStride = 1;
        }
        else
        {
            descPtr[i].strides.nStride = descPtr[i].h * descPtr[i].w * channel;
            descPtr[i].strides.hStride = descPtr[i].w;
            descPtr[i].strides.wStride = 1;
            descPtr[i].strides.cStride = descPtr[i].h * descPtr[i].w;
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
    else if (BitDepthTestMode == I8_TO_F32) funcName += "_i8_f32_";
    else if (BitDepthTestMode == I16_TO_F32) funcName += "_i16_f32_";
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
    string outputFolder = dstDir;
    if (outputFolder.back() != '/')
        outputFolder += "/";
    int cnt = 1;

    for (int i = 0; i < noOfImages; i++)
    {
        string baseName = to_string(i);
        string ext = ".jpg";
        string outputImagePath = outputFolder + baseName + ext;

        Rpp32u height = dstImgSizes[i].height;
        Rpp32u width = dstImgSizes[i].width;

        Mat tempImg;
        if ((dstDescPtr[i].c == 3) && (dstDescPtr[i].layout == RpptLayout::NCHW))
        {
            // For PLN3, extract each channel separately and reconstruct
            int matChannelStride = outputVec[i].rows / 3;
            Mat planarImg(height * 3, width, CV_MAKETYPE(outputVec[i].depth(), 1));
            
            for(int c = 0; c < 3; c++)
            {
                Mat channelSrc = outputVec[i](Rect(0, c * matChannelStride, width, height));
                Mat channelDst = planarImg(Rect(0, c * height, width, height));
                channelSrc.copyTo(channelDst);
            }
            
            tempImg = convert_pln3_to_pkd3(planarImg, height, width);
        }
        else
        {
            tempImg = outputVec[i](Rect(0, 0, width, height));
        }


        Mat saveImg;
        if (tempImg.depth() == CV_32F || tempImg.depth() == CV_16F)
            tempImg.convertTo(saveImg, CV_8U, 255.0);
        else if (tempImg.depth() == CV_8S)
            tempImg.convertTo(saveImg, CV_8U, 1.0, 128.0);
        else
            saveImg = tempImg;

        if (saveImg.channels() == 3)
            cvtColor(saveImg, saveImg, COLOR_RGB2BGR);

        if (!saveImg.empty())
        {
            if (std::filesystem::exists(outputImagePath))
            {
                std::string outPath = outputFolder + baseName + "_" + to_string(cnt) + ext;
                imwrite(outPath, saveImg);
                cnt++;
            }
            else
                imwrite(outputImagePath, saveImg);
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
                printf("hello");
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
