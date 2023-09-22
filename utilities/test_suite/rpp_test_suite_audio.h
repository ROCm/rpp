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

#include "rpp_test_suite_common.h"
#include <fstream>
#include <iomanip>
#include <vector>
#include <half/half.hpp>

namespace fs = boost::filesystem;
using half_float::half;
using namespace std;
typedef half Rpp16f;

// Include this header file to use functions from libsndfile
#include <sndfile.h>

std::map<int, string> audioAugmentationMap =
{
    {0, "non_silent_region_detection"},
};


void verify_output(Rpp32f *dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstDims, string testCase, vector<string> audioNames, string dst)
{
    fstream refFile;
    string refPath = get_current_dir_name();
    string pattern = "HOST/build";
    remove_substring(refPath, pattern);
    refPath = refPath + "REFERENCE_OUTPUTS_AUDIO/";
    int fileMatch = 0;
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        string currentFileName = audioNames[batchCount];
        size_t lastIndex = currentFileName.find_last_of(".");
        currentFileName = currentFileName.substr(0, lastIndex);  // Remove extension from file name
        string outFile = refPath + testCase + "/" + testCase + "_ref_" + currentFileName + ".txt";
        refFile.open(outFile, ios::in);
        if (!refFile.is_open())
        {
            cout << "\n Unable to open the file specified! Please check the path of the file given as input" << endl;
            break;
        }
        int matchedIndices = 0;
        Rpp32f refVal, outVal;
        Rpp32f *dstPtrCurrent = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32f *dstPtrRow = dstPtrCurrent;
        for (int i = 0; i < dstDims[batchCount].height; i++)
        {
            Rpp32f *dstPtrTemp = dstPtrRow;
            for (int j = 0; j < dstDims[batchCount].width; j++)
            {
                refFile >> refVal;
                outVal = dstPtrTemp[j];
                bool invalidComparision = ((outVal == 0.0f) && (refVal != 0.0f));
                if (!invalidComparision && abs(outVal - refVal) < 1e-20)
                    matchedIndices += 1;
            }
            dstPtrRow += dstDescPtr->strides.hStride;
        }
        refFile.close();
        if (matchedIndices == (dstDims[batchCount].width * dstDims[batchCount].height) && matchedIndices !=0)
            fileMatch++;
    }
    std::string status = testCase + ": ";
    cout << std::endl << "Results for Test case: " << testCase << std::endl;
    if (fileMatch == dstDescPtr->n)
    {
        cout << "PASSED!" << std::endl;
        status += "PASSED";
    }
    else
    {
        cout << "FAILED! " << fileMatch << "/" << dstDescPtr->n << " outputs are matching with reference outputs" << std::endl;
        status += "FAILED";
    }
    std::string qaResultsPath = dst + "/QA_results.txt";
    std:: ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open())
    {
        qaResults << status << std::endl;
        qaResults.close();
    }
}

void verify_non_silent_region_detection(float *detectedIndex, float *detectionLength, string testCase, int bs, vector<string> audioNames, string dst)
{
    fstream refFile;
    string refPath = get_current_dir_name();
    string pattern = "HOST/build";
    remove_substring(refPath, pattern);
    refPath = refPath + "REFERENCE_OUTPUTS_AUDIO/";
    int fileMatch = 0;
    for (int i = 0; i < bs; i++)
    {
        string currentFileName = audioNames[i];
        size_t lastIndex = currentFileName.find_last_of(".");
        currentFileName = currentFileName.substr(0, lastIndex);  // Remove extension from file name
        string outFile = refPath + testCase + "/" + testCase + "_ref_" + currentFileName + ".txt";
        refFile.open(outFile, ios::in);
        if (!refFile.is_open())
        {
            cout << "\n Unable to open the file specified! Please check the path of the file given as input" << endl;
            break;
        }

        Rpp32s refIndex, refLength;
        Rpp32s outIndex, outLength;
        refFile >> refIndex;
        refFile >> refLength;
        outIndex = detectedIndex[i];
        outLength = detectionLength[i];

        if ((outIndex == refIndex) && (outLength == refLength))
            fileMatch += 1;
        refFile.close();
    }
    std::string status = testCase + ": ";
    cout << std::endl << "Results for Test case: " << testCase << std::endl;
    if (fileMatch == bs)
    {
        cout << "PASSED!" << std::endl;
        status += "PASSED";
    }
    else
    {
        cout << "FAILED! "<< fileMatch << "/" << bs << " outputs are matching with reference outputs" << std::endl;
        status += "FAILED";
    }

    std::string qaResultsPath = dst + "/QA_results.txt";
    std:: ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open())
    {
        qaResults << status << std::endl;
        qaResults.close();
    }
}