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

#include "rpp_test_suite_common.h"

using namespace std;

std::map<int, string> augmentationMiscMap =
{
    {1, "normalize"}
};

void compute_strides(RpptGenericDescPtr descriptorPtr)
{
    if (descriptorPtr->numDims > 0)
    {
        uint64_t v = 1;
        for (int i = descriptorPtr->numDims - 1; i > 0; i--)
        {
            descriptorPtr->strides[i] = v;
            v *= descriptorPtr->dims[i];
        }
        descriptorPtr->strides[0] = v;
    }
}

string get_path(Rpp32u nDim, Rpp32u readType, string scriptPath, bool isMeanStd = false)
{
    string folderName, suffix;
    if(readType == 0)
    {
        folderName = "input";
        if(isMeanStd)
            suffix = "mean_std";
        else
            suffix = "input";
    }
    else if(readType == 1)
    {
        folderName = "output";
        suffix = "output";
    }

    string fileName = std::to_string(nDim) + "d_" + suffix + ".bin";
    string finalPath = scriptPath + "/../NORMALIZE/" + folderName + "/" + fileName;
    return finalPath;
}

void read_data(Rpp32f *data, Rpp32u nDim, Rpp32u readType, string scriptPath, bool isMeanStd = false)
{
    if(nDim != 2 && nDim != 3)
    {
        std::cout<<"\nGolden Inputs / Outputs are generated only for 2D/3D data"<<std::endl;
        exit(0);
    }
    string refPath = get_path(nDim, readType, scriptPath, isMeanStd);
    read_bin_file(refPath, data);
}

// Fill the starting indices and length of ROI values
void fill_roi_values(Rpp32u nDim, Rpp32u batchSize, Rpp32u *roiTensor, bool qaMode)
{
    if(qaMode)
    {
        switch(nDim)
        {
            case 2:
            {
                std::array<Rpp32u, 4> roi = {0, 0, 100, 100};
                for(int i = 0, j = 0; i < batchSize ; i++, j += 4)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
            }
            case 3:
            {
                std::array<Rpp32u, 6> roi = {0, 0, 0, 50, 50, 8};
                for(int i = 0, j = 0; i < batchSize ; i++, j += 6)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
            }
            default:
            {
                cout << "Error! QA mode is supported only for 2D/3D inputs" << endl;
                exit(0);
            }
        }
    }
    else
    {
        switch(nDim)
        {
            case 2:
            {
                std::array<Rpp32u, 4> roi = {0, 0, 1920, 1080};
                for(int i = 0, j = 0; i < batchSize ; i++, j += 4)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
            }
            case 3:
            {
                std::array<Rpp32u, 6> roi = {0, 0, 0, 1920, 1080, 3};
                for(int i = 0, j = 0; i < batchSize ; i++, j += 6)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
            }
            case 4:
            {
                std::array<Rpp32u, 8> roi = {0, 0, 0, 0, 1, 128, 128, 128};
                for(int i = 0, j = 0; i < batchSize ; i++, j += 8)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
            }
            default:
            {
                // if nDim is not 2/3/4 and mode choosen is not QA
                for(int i = 0; i < batchSize; i++)
                {
                    int startIndex = i * nDim * 2;
                    int lengthIndex = startIndex + nDim;
                    for(int j = 0; j < nDim; j++)
                    {
                        roiTensor[startIndex + j] = 0;
                        roiTensor[lengthIndex + j] = std::rand() % 10;  // limiting max value in a dimension to 10 for testing purposes
                    }
                }
                break;
            }
        }
    }
}

// Set layout for generic descriptor
void set_generic_descriptor_layout(RpptGenericDescPtr srcDescriptorPtrND, RpptGenericDescPtr dstDescriptorPtrND, Rpp32u nDim, int toggle, int qaMode)
{
    if(qaMode && !toggle)
    {
        switch(nDim)
        {
            case 2:
            {
                srcDescriptorPtrND->layout = RpptLayout::NHWC;
                dstDescriptorPtrND->layout = RpptLayout::NHWC;
                break;
            }
            case 3:
            {
                srcDescriptorPtrND->layout = RpptLayout::NHWC;
                dstDescriptorPtrND->layout = RpptLayout::NHWC;
                break;
            }
            case 4:
            {
                srcDescriptorPtrND->layout = RpptLayout::NDHWC;
                dstDescriptorPtrND->layout = RpptLayout::NDHWC;
                break;
            }
            default:
            {
                cout << "Error! QA mode is supported only for 2D/3D inputs" << endl;
                exit(0);
            }
        }
    }
    else if(nDim == 3)
    {
        if(toggle)
        {
            srcDescriptorPtrND->layout = RpptLayout::NHWC;
            dstDescriptorPtrND->layout = RpptLayout::NCHW;
        }
    }
    else
    {
        srcDescriptorPtrND->layout = RpptLayout::NDHWC;
        dstDescriptorPtrND->layout = RpptLayout::NDHWC;
    }
}

// sets generic descriptor numDims, offsetInBytes,  bitdepth, dims and strides
inline void set_generic_descriptor(RpptGenericDescPtr descriptorPtr3D, int nDim, int offsetInBytes, int bitDepth, int batchSize, Rpp32u *roiTensor)
{
    descriptorPtr3D->numDims = nDim + 1;
    descriptorPtr3D->offsetInBytes = offsetInBytes;
    if (bitDepth == 0)
        descriptorPtr3D->dataType = RpptDataType::U8;
    else if (bitDepth == 1)
        descriptorPtr3D->dataType = RpptDataType::F16;
    else if (bitDepth == 2)
        descriptorPtr3D->dataType = RpptDataType::F32;
    else if (bitDepth == 5)
        descriptorPtr3D->dataType = RpptDataType::I8;
    descriptorPtr3D->dims[0] = batchSize;
    for(int i = 1; i <= nDim; i++)
        descriptorPtr3D->dims[i] = roiTensor[nDim + i - 1];
    compute_strides(descriptorPtr3D);
}

// strides used for jumping to corresponding axisMask mean and stddev
std::map<Rpp32s, Rpp32u> paramStrideMap2D =
{
    {1, 0},
    {2, 100},
    {3, 200}
};

// strides used for jumping to corresponding axisMask mean and stddev
std::map<Rpp32s, Rpp32u> paramStrideMap3D =
{
    {1, 0},
    {2, 400},
    {3, 800},
    {4, 808},
    {5, 3308},
    {6, 3358},
    {7, 3408}
};

// fill the mean and stddev values used for normalize
void fill_mean_stddev_values(Rpp32u nDim, Rpp32u size, Rpp32f *meanTensor,
                             Rpp32f *stdDevTensor, bool qaMode, int axisMask, string scriptPath)
{
    if(qaMode)
    {
        Rpp32u numValues, paramStride;
        switch(nDim)
        {
            case 2:
            {
                numValues = 100 + 100 + 1;
                paramStride = paramStrideMap2D[axisMask];
                break;
            }
            case 3:
            {
                numValues = 400 + 400 + 8 + 2500 + 50 + 50 + 1;
                paramStride = paramStrideMap3D[axisMask];
                break;
            }
            default:
            {
                cout << "Error! QA mode is supported only for 2D/3D inputs" << endl;
                exit(0);
            }
        }
        std::vector<Rpp32f> paramBuf(numValues * 2);
        Rpp32f *data = paramBuf.data();
        read_data(data, nDim, 0, scriptPath, true);
        memcpy(meanTensor, data + paramStride, size * sizeof(Rpp32f));
        memcpy(stdDevTensor, data + numValues + paramStride, size * sizeof(Rpp32f));
    }
    else
    {
        for(int j = 0; j < size; j++)
        {
            meanTensor[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            stdDevTensor[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }
}

Rpp32u get_bin_size(Rpp32u nDim, Rpp32u readType, string scriptPath)
{
    string refFile = get_path(nDim, readType, scriptPath);
    std::ifstream filestream(refFile, ios_base::in | ios_base::binary);
    filestream.seekg(0, ios_base::end);
    Rpp32u filesize = filestream.tellg();
    return filesize;
}

void compare_output(Rpp32f *outputF32, Rpp32u nDim, Rpp32u batchSize, Rpp32u bufferLength,
                    string dst, string funcName, int axisMask, string scriptPath, bool isMeanStd = false)
{
    Rpp32u goldenOutputLength = get_bin_size(nDim, 1, scriptPath);
    Rpp32f *refOutput = static_cast<Rpp32f *>(calloc(goldenOutputLength, 1));
    read_data(refOutput, nDim, 1, scriptPath);
    int meanStdDevOutputStride = 0;
    if(isMeanStd)
        meanStdDevOutputStride = goldenOutputLength / (2 * sizeof(Rpp32f));
    int axisMaskStride = (axisMask - 1) * bufferLength;
    int sampleLength = bufferLength / batchSize;
    int fileMatch = 0;
    for(int i = 0; i < batchSize; i++)
    {
        Rpp32f *ref = refOutput + meanStdDevOutputStride + axisMaskStride + i * sampleLength;
        Rpp32f *out = outputF32 + i * sampleLength;
        int cnt = 0;
        for(int j = 0; j < sampleLength; j++)
        {
            bool invalid_comparision = ((out[j] == 0.0f) && (ref[j] != 0.0f));
            if(!invalid_comparision && abs(out[j] - ref[j]) < 1e-4)
                cnt++;
        }
        if (cnt == sampleLength)
            fileMatch++;
    }

    std::string status = funcName + ": ";
    cout << std::endl << "Results for Test case: " << funcName << std::endl;
    if (fileMatch == batchSize)
    {
        std::cout << "\nPASSED!"<<std::endl;
        status += "PASSED";
    }
    else
    {
        std::cout << "\nFAILED! " << fileMatch << "/" << batchSize << " outputs are matching with reference outputs" << std::endl;
        status += "FAILED";
    }
    free(refOutput);

    // Append the QA results to file
    std::string qaResultsPath = dst + "/QA_results.txt";
    std:: ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open())
    {
        qaResults << status << std::endl;
        qaResults.close();
    }
}
