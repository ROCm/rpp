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
namespace fs = boost::filesystem;

std::map<int, string> augmentationMiscMap =
{
    {0, "transpose"}
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

    if (readType == 0)
        finalPath = refPath + "TRANSPOSE/input/" + dim;
    else
        finalPath = refPath + "TRANSPOSE/output/" + dim;

    return finalPath;
}

Rpp32u get_buffer_length(Rpp32u nDim)
{
    string dimSpecifier = std::to_string(nDim) + "d";
    string refPath = get_path(nDim, 0);
    string refFile = refPath + "/" + dimSpecifier + "_" + "input" + std::to_string(0) + ".txt";
    ifstream file(refFile);
    Rpp32u bufferLength = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
    return bufferLength;
}

void read_data(Rpp32f *data, Rpp32u nDim, Rpp32u readType, Rpp32u bufferLength, Rpp32u batchSize)
{
    Rpp32u sampleLength = bufferLength / batchSize;
    if(nDim != 2 && nDim != 3 && nDim != 4)
    {
        std::cout << "\nGolden Inputs / Outputs are generated only for 2D / 3D / 4D data"<<std::endl;
        exit(0);
    }

    string refPath = get_path(nDim, readType);
    string dimSpecifier = std::to_string(nDim) + "d";
    string type = "input";
    if (readType == 1)
        type = "output";

    for(int i = 0; i < batchSize; i++)
    {
        string refFile = refPath + "/" + dimSpecifier + "_" + type + std::to_string(i) + ".txt";
        Rpp32f *curData = data + i * sampleLength;
        fstream fileStream;
        fileStream.open(refFile, ios::in);
        if(!fileStream.is_open())
        {
            cout << "Unable to open the file specified! Please check the path of the file given as input" << endl;
            break;
        }
        for(int j = 0; j < sampleLength; j++)
        {
            Rpp32f val;
            fileStream>>val;
            curData[j] = val;
        }
    }
}

void fill_roi_values(Rpp32u nDim, Rpp32u batchSize, Rpp32u *roiTensor, bool qaMode)
{
    switch(nDim)
    {
        case 2:
        {
            if (qaMode)
            {
                for(int i = 0; i < batchSize * 4; i += 4)
                {
                    roiTensor[i] = 0;
                    roiTensor[i + 1] = 0;
                    roiTensor[i + 2] = 125;
                    roiTensor[i + 3] = 125;
                }
            }
            else
            {
                for(int i = 0; i < batchSize * 4; i += 4)
                {
                    roiTensor[i] = 0;
                    roiTensor[i + 1] = 0;
                    roiTensor[i + 2] = 1920;
                    roiTensor[i + 3] = 1080;
                }
            }
            break;
        }
        case 3:
        {
            if (qaMode)
            {
                for(int i = 0; i < batchSize * 6; i += 6)
                {
                    roiTensor[i] = 0;
                    roiTensor[i + 1] = 0;
                    roiTensor[i + 2] = 0;
                    roiTensor[i + 3] = 100;
                    roiTensor[i + 4] = 100;
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
                    roiTensor[i + 4] = 75;
                    roiTensor[i + 5] = 75;
                    roiTensor[i + 6] = 4;
                    roiTensor[i + 7] = 3;
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
                    roiTensor[i + 4] = 1;
                    roiTensor[i + 5] = 128;
                    roiTensor[i + 6] = 128;
                    roiTensor[i + 7] = 128;
                }
            }
            break;
        }
        default:
        {
            // if nDim is not 2/3/4 and mode choosen is not QA
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

// fill the permuation values used for transpose
void fill_perm_values(Rpp32u nDim, Rpp32u *permTensor, bool qaMode)
{
    if(qaMode)
    {
        switch(nDim)
        {
            case 2:
            {
                permTensor[0] = 1;
                permTensor[1] = 0;
                break;
            }
            case 3:
            {
                permTensor[0] = 2;
                permTensor[1] = 0;
                permTensor[2] = 1;
                break;
            }
            case 4:
            {
                permTensor[0] = 1;
                permTensor[1] = 2;
                permTensor[2] = 3;
                permTensor[3] = 0;
                break;
            }
            default:
            {
                cout << "Error! QA mode is supported only for 2 / 3 / 4 Dimension inputs" << endl;
                exit(0);
            }
        }
    }
    else
    {
        for(int i = 0; i < nDim; i++)
            permTensor[i] = nDim - 1 - i;
    }
}

void compare_output(Rpp32f *outputF32, Rpp32u nDim, Rpp32u batchSize, string dst, string funcName)
{
    Rpp32u bufferLength = get_buffer_length(nDim);
    Rpp32f *refOutput = (Rpp32f *)calloc(bufferLength * batchSize, sizeof(Rpp32f));
    read_data(refOutput, nDim, 1, bufferLength * batchSize, batchSize);
    int fileMatch = 0;
    for(int i = 0; i < batchSize; i++)
    {
        Rpp32f *ref = refOutput + i * bufferLength;
        Rpp32f *out = outputF32 + i * bufferLength;
        int cnt = 0;
        for(int j = 0; j < bufferLength; j++)
        {
            if (abs(out[j] - ref[j]) < 1e-20)
                cnt++;
        }
        if (cnt == bufferLength)
            fileMatch++;
    }

    std::string status = funcName + ": ";
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
    const int MIN_ARG_COUNT = 6;
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_transpose_host <case number = 0:0> <test type 0/1> <number of dimensions> <batch size> <num runs>\n");
        return -1;
    }

    Rpp32u testCase, testType, nDim, batchSize, numRuns;
    bool qaMode;

    testCase = atoi(argv[1]);
    testType = atoi(argv[2]);
    nDim = atoi(argv[3]);
    batchSize = atoi(argv[4]);
    numRuns = atoi(argv[5]);
    string dst = argv[6];
    qaMode = (testType == 0);

    if (qaMode && batchSize != 3)
    {
        std::cout << "QA mode can only run with batchsize 3"<<std::endl;
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
    srcDescriptorPtrND->layout = RpptLayout::NDHWC;

    dstDescriptorPtrND  = &dstDescriptor;
    dstDescriptorPtrND->numDims = nDim;
    dstDescriptorPtrND->offsetInBytes = 0;
    dstDescriptorPtrND->dataType = RpptDataType::F32;
    dstDescriptorPtrND->layout = RpptLayout::NDHWC;

    // set dims and compute strides for src
    srcDescriptorPtrND->dims[0] = batchSize;
    dstDescriptorPtrND->dims[0] = batchSize;
    for(int i = 1; i <= nDim; i++)
        srcDescriptorPtrND->dims[i] = roiTensor[nDim + i - 1];
    compute_strides(srcDescriptorPtrND);

    // if testCase is not transpose, then copy dims and strides from src to dst
    if(testCase != 0)
    {
        memcpy(dstDescriptorPtrND->dims, srcDescriptorPtrND->dims, nDim * sizeof(Rpp32u));
        memcpy(dstDescriptorPtrND->strides, srcDescriptorPtrND->strides, nDim * sizeof(Rpp32u));
    }

    Rpp32u numValues = 1;
    for(int i = 0; i <= nDim; i++)
        numValues *= srcDescriptorPtrND->dims[i];

    // allocate memory for input / output
    Rpp32f *inputF32 = NULL, *outputF32 = NULL;
    inputF32 = (Rpp32f *)calloc(numValues, sizeof(Rpp32f));
    outputF32 = (Rpp32f *)calloc(numValues, sizeof(Rpp32f));

    // read input data
    if(qaMode)
        read_data(inputF32, nDim, 0,  numValues, batchSize);
    else
    {
        std::srand(0);
        for(int i = 0; i < numValues; i++)
            inputF32[i] = (float)(std::rand() % 255);
    }

    // case-wise RPP API and measure time script for Unit and Performance test
    printf("\nRunning transpose %d times (each time with a batch size of %d) and computing mean statistics...", numRuns, batchSize);

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
            case 0:
            {
                Rpp32u permTensor[nDim];
                fill_perm_values(nDim, permTensor, qaMode);

                for(int i = 1; i <= nDim; i++)
                    dstDescriptorPtrND->dims[i] = roiTensor[nDim + permTensor[i - 1]];
                compute_strides(dstDescriptorPtrND);

                startWallTime = omp_get_wtime();
                rppt_transpose_generic_host(inputF32, srcDescriptorPtrND, outputF32, dstDescriptorPtrND, permTensor, roiTensor, handle);
                break;
            }
            default:
            {
                cout << "functionality is not supported" << endl;
                exit(0);
            }
        }
        endWallTime = omp_get_wtime();
        wallTime = endWallTime - startWallTime;
        maxWallTime = std::max(maxWallTime, wallTime);
        minWallTime = std::min(minWallTime, wallTime);
        avgWallTime += wallTime;
    }

    // compare outputs if qaMode is true
    if(qaMode)
        compare_output(outputF32, nDim, batchSize, dst, funcName);
    else
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