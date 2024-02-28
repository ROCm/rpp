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

string get_path(Rpp32u nDim, Rpp32u readType, string scriptPath)
{
    string type = "input";
    if(readType == 1)
        type = "output";
    string fileName = std::to_string(nDim) + "d_" + type + ".bin";
    string finalPath = scriptPath + "/../NORMALIZE/" + type + "/" + fileName;
    return finalPath;
}

void read_data(Rpp32f *data, Rpp32u nDim, Rpp32u readType, string scriptPath)
{
    if(nDim != 2 && nDim != 3 && nDim != 4)
    {
        std::cout<<"\nGolden Inputs / Outputs are generated only for 2D/3D/4D data"<<std::endl;
        exit(0);
    }
    string refPath = get_path(nDim, readType, scriptPath);
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
                std::array<Rpp32u, 6> roi = {0, 0, 0, 3, 4, 16};
                for(int i = 0, j = 0; i < batchSize ; i++, j += 6)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
            }
            case 4:
            {
                std::array<Rpp32u, 8> roi = {0, 0, 0, 0, 2, 3, 4, 5};
                for(int i = 0, j = 0; i < batchSize ; i++, j += 8)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
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
                std::array<Rpp32u, 8> roi = {0, 0, 0, 0, 45, 20, 25, 10};
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
                cout << "Error! QA mode is supported only for 3/4 Dimension inputs" << endl;
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

// fill the mean and stddev values used for normalize
void fill_mean_stddev_values(Rpp32u nDim, Rpp32u batchSize, Rpp32u size, Rpp32f *meanTensor, Rpp32f *stdDevTensor, bool qaMode)
{
    if(qaMode)
    {
        switch(nDim)
        {
            case 3:
            {
                for(int i = 0; i < batchSize * 16; i += 16)
                {
                    meanTensor[i] = 0.10044092854408704;
                    meanTensor[i + 1] = 0.9923954479926445;
                    meanTensor[i + 2] = 0.1463966240511576;
                    meanTensor[i + 3] = 0.8511748753528452;
                    meanTensor[i + 4] = 0.241989919160714;
                    meanTensor[i + 5] = 0.724488856565572;
                    meanTensor[i + 6] = 0.42082916847069873;
                    meanTensor[i + 7] = 0.46859982127051925;
                    meanTensor[i + 8] = 0.3775650937841545;
                    meanTensor[i + 9] = 0.4495086677760334;
                    meanTensor[i + 10] = 0.8382375156517684;
                    meanTensor[i + 11] = 0.4477761580072823;
                    meanTensor[i + 12] = 0.32061482730987134;
                    meanTensor[i + 13] = 0.3844935131563223;
                    meanTensor[i + 14] = 0.7987222326619818;
                    meanTensor[i + 15] = 0.10494099481214858;

                    stdDevTensor[i] = 0.23043620850364177;
                    stdDevTensor[i + 1] = 0.1455208174769702;
                    stdDevTensor[i + 2] = 0.8719780160981172;
                    stdDevTensor[i + 3] = 0.414600410599096;
                    stdDevTensor[i + 4] = 0.6735379720722622;
                    stdDevTensor[i + 5] = 0.6898490355115773;
                    stdDevTensor[i + 6] = 0.928227311970384;
                    stdDevTensor[i + 7] = 0.2256026577060809;
                    stdDevTensor[i + 8] = 0.06284357739269342;
                    stdDevTensor[i + 9] = 0.5563155411432268;
                    stdDevTensor[i + 10] = 0.21911684022872935;
                    stdDevTensor[i + 11] = 0.3947508370853534;
                    stdDevTensor[i + 12] = 0.7577237777839925;
                    stdDevTensor[i + 13] = 0.8079874528633991;
                    stdDevTensor[i + 14] = 0.21589143239793473;
                    stdDevTensor[i + 15] = 0.7972578943669427;
                }
                break;
            }
            default:
            {
                cout << "Error! QA mode is supported only for 3 Dimension inputs with mean and stddev read from user" << endl;
                exit(0);
            }
        }
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

void compare_output(Rpp32f *outputF32, Rpp32u nDim, Rpp32u batchSize, Rpp32u bufferLength, string dst, string funcName, int axisMask, string scriptPath)
{
    Rpp32u goldenOutputLength = get_bin_size(nDim, 1, scriptPath);
    Rpp32f *refOutput = static_cast<Rpp32f *>(calloc(goldenOutputLength, 1));
    read_data(refOutput, nDim, 1, scriptPath);
    int axisMaskStride = (axisMask - 1) * bufferLength;
    int sampleLength = bufferLength / batchSize;
    int fileMatch = 0;
    for(int i = 0; i < batchSize; i++)
    {
        Rpp32f *ref = refOutput + axisMaskStride + i * sampleLength;
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