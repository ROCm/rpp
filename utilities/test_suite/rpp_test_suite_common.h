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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <turbojpeg.h>

using namespace cv;
using namespace std;

#define CUTOFF 1

std::map<int, string> augmentationMap =
{
    {0, "brightness"},
    {2, "blend"},
    {4, "contrast"},
    {13, "exposure"},
    {31, "color_cast"},
    {34, "lut"},
    {36, "color_twist"},
    {38, "crop_mirror_normalize"},
    {88, "image_min_max"},
    {89, "image_min"},
    {90, "image_max"},
};

template <typename T>
inline T validate_pixel_range(T pixel)
{
    pixel = (pixel < static_cast<Rpp32f>(0)) ? (static_cast<Rpp32f>(0)) : ((pixel < static_cast<Rpp32f>(255)) ? pixel : (static_cast<Rpp32f>(255)));
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

inline int set_input_channels(int layoutType)
{
    if(layoutType == 0 || layoutType == 1)
        return 3;
    else
        return 1;
}

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
                *inputTempG = *inputCopyTemp;
                inputCopyTemp++;
                *inputTempB = *inputCopyTemp;
                inputCopyTemp++;

                // std::cerr<<"i, j, R, G, B: "<<i<<", "<<j<<", "<<(int)*inputTempR<<", "<<(int)*inputTempG<<", "<<(int)*inputTempB<<std::endl;
                inputTempR++;
                inputTempB++;
                inputTempG++;
            }
        }
    }

    free(inputCopy);
}

inline void read_image_batch_opencv(Rpp8u *input, RpptDescPtr descPtr, string imageNames[])
{
    for(int i = 0; i < descPtr->n; i++)
    {
        Rpp8u *inputTemp = input + (i * descPtr->strides.nStride);
        string inputImagePath = imageNames[i];
        cv::Mat image, imageBgr;
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
            inputTemp += descPtr->w * descPtr->c;
        }
    }
}

inline void read_image_batch_turbojpeg(Rpp8u *input, RpptDescPtr descPtr, string imageNames[])
{
    tjhandle m_jpegDecompressor = tjInitDecompress();

    // Loop through the input images
    for (int i = 0; i < descPtr->n; i++)
    {
        // Read the JPEG compressed data from a file
        std::string inputImagePath = imageNames[i];
        FILE* fp = fopen(inputImagePath.c_str(), "rb");
        if(!fp)
            std::cerr<<"\n unable to open file : "<<inputImagePath;
        fseek(fp, 0, SEEK_END);
        long jpegSize = ftell(fp);
        rewind(fp);
        unsigned char* jpegBuf = (unsigned char*)malloc(jpegSize);
        fread(jpegBuf, 1, jpegSize, fp);
        fclose(fp);

        // Decompress the JPEG data into an RGB image buffer
        int width, height, subsamp, color_space;
        if(tjDecompressHeader2(m_jpegDecompressor, jpegBuf, jpegSize, &width, &height, &color_space) != 0)
            std::cerr<<"\n Jpeg image decode failed in tjDecompressHeader2";
        Rpp8u* rgbBuf;
        int elementsInRow;
        if(descPtr->c == 3)
        {
            elementsInRow = width * descPtr->c;
            rgbBuf= (Rpp8u*)malloc(width * height * 3);
            if(tjDecompress2(m_jpegDecompressor, jpegBuf, jpegSize, rgbBuf, width, width * 3, height, TJPF_RGB, TJFLAG_FASTDCT) != 0)
                std::cerr<<"\n Jpeg image decode failed ";
        }
        else
        {
            elementsInRow = width;
            rgbBuf= (Rpp8u*)malloc(width * height);
            if(tjDecompress2(m_jpegDecompressor, jpegBuf, jpegSize, rgbBuf, width, width, height, TJPF_GRAY, 0) != 0)
                std::cerr<<"\n Jpeg image decode failed ";
        }
        // Copy the decompressed image buffer to the RPP input buffer
        Rpp8u *inputTemp = input + (i * descPtr->strides.nStride);
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

inline void write_image_batch_opencv(string outputFolder, Rpp8u *output, RpptDescPtr dstDescPtr, vector<string> imageNames, RpptImagePatch *dstImgSizes)
{
    // create output folder
    mkdir(outputFolder.c_str(), 0700);
    outputFolder += "/";

    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
    Rpp8u *offsettedOutput = output + dstDescPtr->offsetInBytes;
    for (int j = 0; j < dstDescPtr->n; j++)
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

        string outputImagePath = outputFolder + imageNames[j];
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

template <typename T>
inline void compare_output(T* output, string funcName, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, RpptImagePatch *dstImgSizes, int noOfImages, string interpolationTypeName, int testCase, string dst)
{
    string func = funcName;
    string refPath = get_current_dir_name();
    string pattern = "/build";
    string refFile = "";
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
    else
        refFile = refPath + "/../REFERENCE_OUTPUT/" + funcName + "/"+ func + ".csv";

    ifstream file(refFile);
    Rpp8u *refOutput;
    refOutput = (Rpp8u *)malloc(noOfImages * dstDescPtr->strides.nStride * sizeof(Rpp8u));
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
    Rpp8u *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef;
    for(int c = 0; c < noOfImages; c++)
    {
        outputTemp = output + c * dstDescPtr->strides.nStride;
        outputTempRef = refOutput + c * dstDescPtr->strides.nStride;
        int height = dstImgSizes[c].height;
        int width = dstImgSizes[c].width;
        int matched_idx = 0;

        if(dstDescPtr->layout == RpptLayout::NHWC)
            width = dstImgSizes[c].width * dstDescPtr->c;
        else
        {
            if (dstDescPtr->c == 3)
                height = dstImgSizes[c].height * dstDescPtr->c;
        }

        for(int i = 0; i < height; i++)
        {
            rowTemp = outputTemp + i * dstDescPtr->strides.hStride;
            rowTempRef = outputTempRef + i * dstDescPtr->strides.hStride;
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
    std::cerr<<std::endl<<"Results for "<<func<<" :"<<std::endl;
    std::string status = func + ": ";
    if(fileMatch == dstDescPtr->n)
    {
        std::cerr<<"PASSED!"<<std::endl;
        status += "PASSED";
    }
    else
    {
        std::cerr<<"FAILED! "<<fileMatch<<"/"<<dstDescPtr->n<<" outputs are matching with reference outputs"<<std::endl;
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
