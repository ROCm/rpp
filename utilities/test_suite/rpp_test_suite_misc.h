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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <unistd.h>
#include <dirent.h>
#include "filesystem.h"
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iterator>
#include <map>
#include "rpp.h"

using namespace std;

#define CHECK(x) do { \
  int retval = (x); \
  if (retval != 0) { \
    fprintf(stderr, "Runtime error: %s returned %d at %s:%d", #x, retval, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)

std::map<int, string> augmentationMiscMap =
{
    {0, "transpose"}
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

inline void remove_substring(string &str, string &pattern)
{
    std::string::size_type i = str.find(pattern);
    while (i != std::string::npos)
    {
        str.erase(i, pattern.length());
        i = str.find(pattern, i);
   }
}

string get_path(Rpp32u nDim, Rpp32u readType, string pattern, string funcName)
{
    string refPath = get_current_dir_name();
    string finalPath = "";
    remove_substring(refPath, pattern);
    string dim = std::to_string(nDim) + "d";

    if (readType == 0)
        finalPath = refPath + funcName + "/input/" + dim;
    else
        finalPath = refPath + funcName + "/output/" + dim;

    return finalPath;
}

Rpp32u get_buffer_length(Rpp32u nDim, string pattern, string funcName)
{
    string dimSpecifier = std::to_string(nDim) + "d";
    string refPath = get_path(nDim, 0, pattern, funcName);
    string refFile = refPath + "/" + dimSpecifier + "_" + "input" + std::to_string(0) + ".txt";
    ifstream file(refFile);
    Rpp32u bufferLength = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
    return bufferLength;
}

void read_data(Rpp32f *data, Rpp32u nDim, Rpp32u readType, Rpp32u bufferLength, Rpp32u batchSize, string pattern, string funcName)
{
    Rpp32u sampleLength = bufferLength / batchSize;
    if(nDim != 2 && nDim != 3 && nDim != 4)
    {
        std::cout << "\nGolden Inputs / Outputs are generated only for 2D / 3D / 4D data"<<std::endl;
        exit(0);
    }

    string refPath = get_path(nDim, readType, pattern + "/build", funcName);
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
            exit(0);
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

// Fills specific roi values for 2D/3D/4D inputs for golden output based QA tests, or performance tests.
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

void compare_output(Rpp32f *outputF32, Rpp32u nDim, Rpp32u batchSize, string dst, string pattern, string funcName)
{
    Rpp32u bufferLength = get_buffer_length(nDim, pattern + "/build", funcName);
    Rpp32f *refOutput = (Rpp32f *)calloc(bufferLength * batchSize, sizeof(Rpp32f));
    read_data(refOutput, nDim, 1, bufferLength * batchSize, batchSize, pattern, funcName);
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