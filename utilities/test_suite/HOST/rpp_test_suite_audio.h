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
    {8, "normalize"},
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

// sets descriptor dimensions and strides of src/dst
inline void set_audio_descriptor_dims_and_strides(RpptGenericDescPtr descPtr, int batchSize, int maxHeight, int maxWidth, int maxChannels, int offsetInBytes)
{
    descPtr->numDims = 4;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->dims[0] = batchSize;
    descPtr->dims[1] = maxHeight;
    descPtr->dims[2] = maxWidth;
    descPtr->dims[3] = maxChannels;

    // Optionally set w stride as a multiple of 8 for src/dst
    descPtr->dims[2] = ((descPtr->dims[2] / 8) * 8) + 8;
    compute_strides(descPtr);
}

// sets values of maxHeight and maxWidth
inline void set_audio_max_dimensions(vector<string> audioFilesPath, int& maxWidth, int& maxChannels)
{
    for (const std::string& audioPath : audioFilesPath)
    {
        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;

        // The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (!(infile = sf_open (audioPath.c_str(), SFM_READ, &sfinfo)))
        {
            sf_close (infile);
            continue;
        }

        maxWidth = std::max(maxWidth, static_cast<int>(sfinfo.frames));
        maxChannels = std::max(maxChannels, static_cast<int>(sfinfo.channels));

        // Close input
        sf_close (infile);
    }
}

void read_audio_batch_and_fill_dims(RpptGenericDescPtr descPtr, Rpp32f *inputf32, vector<string> audioFilesPath, int iterCount, Rpp32u *roiTensor)
{
    auto fileIndex = iterCount * descPtr->dims[0];
    for (int i = 0, j = fileIndex; i < descPtr->dims[0], j < fileIndex + descPtr->dims[0]; i++, j++)
    {
        Rpp32f *inputTempF32;
        inputTempF32 = inputf32 + (i * descPtr->strides[0]);

        // Read and decode data
        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;

        // The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (!(infile = sf_open (audioFilesPath[j].c_str(), SFM_READ, &sfinfo)))
        {
            sf_close (infile);
            continue;
        }

        roiTensor[i * 2] = sfinfo.frames;
        roiTensor[(i * 2) + 1] = sfinfo.channels;

        int bufferLength = sfinfo.frames * sfinfo.channels;
        readcount = (int) sf_read_float (infile, inputTempF32, bufferLength);
        if (readcount != bufferLength)
        {
            std::cout << "Unable to read audio file: "<<audioFilesPath[j].c_str() << std::endl;
            exit(0);
        }

        // Close input
        sf_close (infile);
    }
}

void verify_output(Rpp32f *dstPtr, RpptGenericDescPtr dstDescPtr, Rpp32u roiTensor, string testCase, vector<string> audioNames, string dst)
{
    //Considering height dim as 1 always
    fstream refFile;
    string refPath = get_current_dir_name();
    string pattern = "HOST/build";
    remove_substring(refPath, pattern);
    refPath = refPath + "REFERENCE_OUTPUTS_AUDIO/";
    int fileMatch = 0;
    for (int batchCount = 0; batchCount < dstDescPtr->dims[0]; batchCount++)
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
        Rpp32f *dstPtrCurrent = dstPtr + batchCount * dstDescPtr->strides[0];
        Rpp32f *dstPtrRow = dstPtrCurrent;
        Rpp32u hStride = dstDescPtr->strides[1];
        if (roiTensor[(batchCount * 2) + 1] == 1)
            hStride = 1;

        Rpp32f *dstPtrTemp = dstPtrRow;
        for (int j = 0; j < roiTensor[(batchCount * 2) + 1]; j++)
        {
            refFile >> refVal;
            outVal = dstPtrTemp[j];
            bool invalidComparision = ((outVal == 0.0f) && (refVal != 0.0f));
            if (!invalidComparision && abs(outVal - refVal) < 1e-20)
                matchedIndices += 1;
        }
        dstPtrRow += hStride;

        refFile.close();
        if (matchedIndices == (roiTensor[(batchCount * 2) + 1]) && matchedIndices !=0)
            fileMatch++;
    }
    std::string status = testCase + ": ";
    cout << std::endl << "Results for Test case: " << testCase << std::endl;
    if (fileMatch == dstDescPtr->dims[0])
    {
        cout << "PASSED!" << std::endl;
        status += "PASSED";
    }
    else
    {
        cout << "FAILED! " << fileMatch << "/" << dstDescPtr->dims[0] << " outputs are matching with reference outputs" << std::endl;
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