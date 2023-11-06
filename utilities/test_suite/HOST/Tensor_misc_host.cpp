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
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <unistd.h>
#include <dirent.h>
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iterator>
#include "rpp.h"
#include "../rpp_test_suite_common.h"

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
        for (int i = descriptorPtr->numDims; i > 0; i--)
        {
            descriptorPtr->strides[i] = v;
            v *= descriptorPtr->dims[i];
        }
        descriptorPtr->strides[0] = v;
    }
}

string get_path(Rpp32u nDim, Rpp32u readType)
{
    string refPath = get_current_dir_name();
    string pattern = "HOST/build";
    string finalPath = "";
    remove_substring(refPath, pattern);
    string dim = std::to_string(nDim) + "d";

    if (readType == 0 || readType == 2 || readType == 3)
        finalPath = refPath + "NORMALIZE/input/" + dim;
    else
        finalPath = refPath + "NORMALIZE/output/" + dim;

    return finalPath;
}

Rpp32u get_buffer_length(Rpp32u nDim, int axisMask)
{
    string dimSpecifier = std::to_string(nDim) + "d";
    string refPath = get_path(nDim, 0);
    string refFile;
    if(axisMask != 6)
        refFile = refPath + "/" + dimSpecifier + "_" + "input" + "_3x4x16.txt";
    else
        refFile = refPath + "/" + dimSpecifier + "_" + "input" + "_4x5x7.txt";
    ifstream file(refFile);
    Rpp32u bufferLength = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
    return bufferLength;
}

void read_data(Rpp32f *data, Rpp32u nDim, Rpp32u readType, Rpp32u bufferLength, Rpp32u batchSize, Rpp32u axisMask)
{
    Rpp32u sampleLength = bufferLength / batchSize;
    if(nDim != 3 && nDim != 4)
    {
        std::cout<<"\nGolden Inputs / Outputs are generated only for 3D/4D data"<<std::endl;
        exit(0);
    }

    string refPath = get_path(nDim, readType);
    string dimSpecifier = std::to_string(nDim) + "d";
    string type = "input";
    if (readType == 1)
        type = "output";

    string refFilePath;
    if (readType == 1)
        refFilePath = refPath + "/" + dimSpecifier + "_axisMask" + std::to_string(axisMask) + ".txt";
    else
    {
        if(axisMask != 6)
            refFilePath = refPath + "/" + dimSpecifier + "_" + type + "_3x4x16.txt";
        else
            refFilePath = refPath + "/" + dimSpecifier + "_" + type + "_4x5x7.txt";
    }
    fstream refFile;
    refFile.open(refFilePath, ios::in);
    if(!refFile.is_open())
        cout<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;

    for(int i = 0; i < batchSize; i++)
    {
        Rpp32f *curData = data + i * sampleLength;
        for(int j = 0; j < sampleLength; j++)
            refFile >> curData[j];
    }
}

// Fill the starting indices and length of ROI values
void fill_roi_values(Rpp32u nDim, Rpp32u batchSize, Rpp32u *roiTensor, bool qaMode)
{
    switch(nDim)
    {
        case 3:
        {
            if (qaMode)
            {
                for(int i = 0; i < batchSize * 6; i += 6)
                {
                    roiTensor[i] = 0;
                    roiTensor[i + 1] = 0;
                    roiTensor[i + 2] = 0;
                    roiTensor[i + 3] = 3;
                    roiTensor[i + 4] = 4;
                    roiTensor[i + 5] = 16;
                }
            }
            else
            {
                for(int i = 0; i < batchSize * 6; i += 6)
                {
                    roiTensor[i] = 0;
                    roiTensor[i + 1] = 0;
                    roiTensor[i + 2] = 0;
                    roiTensor[i + 3] = 1152;
                    roiTensor[i + 4] = 768;
                    roiTensor[i + 5] = 16;
                }
            }
            break;
        }
        case 4:
        {
            if (qaMode)
            {
                for(int i = 0; i < batchSize * 8; i += 8)
                {
                    roiTensor[i] = 0;
                    roiTensor[i + 1] = 0;
                    roiTensor[i + 2] = 0;
                    roiTensor[i + 3] = 0;
                    roiTensor[i + 4] = 2;
                    roiTensor[i + 5] = 3;
                    roiTensor[i + 6] = 4;
                    roiTensor[i + 7] = 5;
                }
            }
            else
            {
                for(int i = 0; i < batchSize * 8; i += 8)
                {
                    roiTensor[i] = 0;
                    roiTensor[i + 1] = 0;
                    roiTensor[i + 2] = 0;
                    roiTensor[i + 3] = 0;
                    roiTensor[i + 4] = 45;
                    roiTensor[i + 5] = 20;
                    roiTensor[i + 6] = 25;
                    roiTensor[i + 7] = 10;
                }
            }
            break;
        }

        default:
        {
            // if nDim is not 3/4 and mode choosen is not QA
            if(!qaMode)
            {
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
            }
            break;
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

void compare_output(Rpp32f *outputF32, Rpp32u nDim, Rpp32u batchSize, string dst, string funcName, int axisMask)
{
    Rpp32u bufferLength = get_buffer_length(nDim, axisMask);
    Rpp32f *refOutput = (Rpp32f *)calloc(bufferLength, sizeof(Rpp32f));
    read_data(refOutput, nDim, 1, bufferLength, batchSize, axisMask);
    int sampleLength = bufferLength / batchSize;
    int fileMatch = 0;
    for(int i = 0; i < batchSize; i++)
    {
        Rpp32f *ref = refOutput + i * sampleLength;
        Rpp32f *out = outputF32 + i * sampleLength;
        int cnt = 0;
        for(int j = 0; j < sampleLength; j++)
        {
            bool invalid_comparision = ((out[j] == 0.0f) && (ref[j] != 0.0f));
            if(!invalid_comparision && abs(out[j] - ref[j]) < 1e-6)
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

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 7;
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_normalize_host <case number = 0:0> <test type 0/1> <toggle 0/1> <number of dimensions> <batch size> <num runs> <dst path>\n");
        return -1;
    }
    Rpp32u testCase, testType, nDim, batchSize, numRuns, toggle;
    bool qaMode;

    testCase = atoi(argv[1]);
    testType = atoi(argv[2]);
    toggle = atoi(argv[3]);
    nDim = atoi(argv[4]);
    batchSize = atoi(argv[5]);
    numRuns = atoi(argv[6]);
    string dst = argv[7];
    qaMode = (testType == 0);

    if (qaMode && batchSize != 3)
    {
        std::cout<<"QA mode can only run with batchsize 3"<<std::endl;
        return -1;
    }

    string funcName = augmentationMiscMap[testCase];
    if (funcName.empty())
    {
        printf("\ncase %d is not supported\n", testCase);
        return -1;
    }

    // fill roi based on mode and number of dimensions
    Rpp32u *roiTensor = (Rpp32u *)calloc(nDim * 2 * batchSize, sizeof(Rpp32u));
    fill_roi_values(nDim, batchSize, roiTensor, qaMode);

    // set src/dst generic tensor descriptors
    RpptGenericDesc srcDescriptor, dstDescriptor;
    RpptGenericDescPtr srcDescriptorPtrND, dstDescriptorPtrND;
    srcDescriptorPtrND  = &srcDescriptor;
    srcDescriptorPtrND->numDims = nDim;
    srcDescriptorPtrND->offsetInBytes = 0;
    srcDescriptorPtrND->dataType = RpptDataType::F32;

    dstDescriptorPtrND  = &dstDescriptor;
    dstDescriptorPtrND->numDims = nDim;
    dstDescriptorPtrND->offsetInBytes = 0;
    dstDescriptorPtrND->dataType = RpptDataType::F32;

    // set dims and compute strides
    srcDescriptorPtrND->dims[0] = batchSize;
    dstDescriptorPtrND->dims[0] = batchSize;
    for(int i = 1; i <= nDim; i++)
        srcDescriptorPtrND->dims[i] = roiTensor[nDim + i - 1];
    compute_strides(srcDescriptorPtrND);

    // if testCase is not normalize, then copy dims and strides from src to dst
    if(testCase != 0)
    {
        memcpy(dstDescriptorPtrND->dims, srcDescriptorPtrND->dims, nDim * sizeof(Rpp32u));
        memcpy(dstDescriptorPtrND->strides, srcDescriptorPtrND->strides, nDim * sizeof(Rpp32u));
    }

    set_generic_descriptor_layout(srcDescriptorPtrND, dstDescriptorPtrND, nDim, toggle, qaMode);

    Rpp32u numValues = 1;
    for(int i = 0; i <= nDim; i++)
        numValues *= srcDescriptorPtrND->dims[i];

    // allocate memory for input / output
    Rpp32f *inputF32 = NULL, *outputF32 = NULL;
    inputF32 = (Rpp32f *)calloc(numValues, sizeof(Rpp32f));
    outputF32 = (Rpp32f *)calloc(numValues, sizeof(Rpp32f));

    // case-wise RPP API and measure time script for Unit and Performance test
    printf("\nRunning normalize %d times (each time with a batch size of %d) and computing mean statistics...", numRuns, batchSize);

    // Set the number of threads to be used by OpenMP pragma for RPP batch processing on host.
    // If numThreads value passed is 0, number of OpenMP threads used by RPP will be set to batch size
    Rpp32u numThreads = 0;
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, batchSize, numThreads);

    double startWallTime, endWallTime;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0, wallTime = 0;
    for(int perfCount = 0; perfCount < numRuns; perfCount++)
    {
        switch(testCase)
        {
            case 1:
            {
                // Modify ROI to 4x5x7 when checking QA for axisMask = 6 alone(calls direct c code internally)
                int axisMask = 3; // 3D HWC Channel normalize axes(0,1)
                float scale = 1.0;
                float shift = 0.0;
                bool computeMean, computeStddev;
                computeMean = computeStddev = 0;

                // read input data
                if(qaMode)
                    read_data(inputF32, nDim, 0,  numValues, batchSize, axisMask);
                else
                {
                    std::srand(0);
                    for(int i = 0; i < numValues; i++)
                        inputF32[i] = (float)(std::rand() % 255);
                }

                if (qaMode && nDim == 3 && axisMask == 3 && (computeMean || computeStddev))
                {
                    std::cout<<"QA mode can only run with mean and stddev input from user when nDim is 3"<<std::endl;
                    return -1;
                }
                else if(qaMode && nDim == 3 && axisMask != 3 && (!computeMean || !computeStddev))
                {
                    std::cout<<"QA mode can only run with internal mean and stddev when nDim is 3"<<std::endl;
                    return -1;
                }

                if (qaMode && nDim == 4 && (!computeMean && !computeStddev))
                {
                    std::cout<<"QA mode can only run with internal mean and stddev when nDim is 4"<<std::endl;
                    return -1;
                }

                Rpp32u size = 1; // length of input tensors differ based on axisMask and nDim
                Rpp32u maxSize = 1;
                for(int batch = 0; batch < batchSize; batch++)
                {
                    size = 1;
                    for(int i = 0; i < nDim; i++)
                        size *= ((axisMask & (int)(pow(2,i))) >= 1) ? 1 : roiTensor[(nDim * 2 * batch) + nDim + i];
                    maxSize = max(maxSize, size);
                }

                Rpp32f *meanTensor = (Rpp32f *)calloc(maxSize * batchSize, sizeof(Rpp32f));
                Rpp32f *stdDevTensor = (Rpp32f *)calloc(maxSize * batchSize, sizeof(Rpp32f));

                if(!(computeMean && computeStddev))
                    fill_mean_stddev_values(nDim, batchSize, size, meanTensor, stdDevTensor, qaMode);

                startWallTime = omp_get_wtime();
                rppt_normalize_generic_host(inputF32, srcDescriptorPtrND, outputF32, dstDescriptorPtrND, axisMask, meanTensor, stdDevTensor, computeMean, computeStddev, scale, shift, roiTensor, handle);
                free(meanTensor);
                free(stdDevTensor);

                // compare outputs if qaMode is true
                if(qaMode)
                    compare_output(outputF32, nDim, batchSize, dst, funcName, axisMask);
                break;
            }
            default:
            {
                cout << "functionality is not supported" <<std::endl;
                exit(0);
            }
        }
        endWallTime = omp_get_wtime();

        wallTime = endWallTime - startWallTime;
        maxWallTime = std::max(maxWallTime, wallTime);
        minWallTime = std::min(minWallTime, wallTime);
        avgWallTime += wallTime;
    }

    if(!qaMode)
    {
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= numRuns;
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }

    free(inputF32);
    free(outputF32);
    free(roiTensor);
    return 0;
}