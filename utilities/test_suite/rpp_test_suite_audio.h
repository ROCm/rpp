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
#include <iomanip>
#include <vector>
#include <half/half.hpp>

using half_float::half;
using namespace std;
typedef half Rpp16f;

// Include this header file to use functions from libsndfile
#include <sndfile.h>

std::map<int, string> audioAugmentationMap =
{
    {0, "non_silent_region_detection"},
    {1, "to_decibels"},
    {2, "pre_emphasis_filter"},
    {3, "down_mixing"},
    {6, "resample"}
};

// Golden outputs for Non Silent Region Detection
std::map<string, std::vector<int>> NonSilentRegionReferenceOutputs =
{
    {"sample1", {0, 35840}},
    {"sample2", {0, 33680}},
    {"sample3", {0, 34160}}
};

// sets descriptor dimensions and strides of src/dst
inline void set_audio_descriptor_dims_and_strides(RpptDescPtr descPtr, int batchSize, int maxHeight, int maxWidth, int maxChannels, int offsetInBytes)
{
    descPtr->numDims = 4;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->n = batchSize;
    descPtr->h = maxHeight;
    descPtr->w = maxWidth;
    descPtr->c = maxChannels;

    // Optionally set w stride as a multiple of 8 for src/dst
    descPtr->w = ((descPtr->w / 8) * 8) + 8;
    descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
    descPtr->strides.hStride = descPtr->c * descPtr->w;
    descPtr->strides.wStride = descPtr->c;
    descPtr->strides.cStride = 1;
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

void read_audio_batch_and_fill_dims(RpptDescPtr descPtr, Rpp32f *inputf32, vector<string> audioFilesPath, int iterCount, Rpp32s *srcLengthTensor, Rpp32s *channelsTensor)
{
    auto fileIndex = iterCount * descPtr->n;
    for (int i = 0, j = fileIndex; i < descPtr->n; i++, j++)
    {
        Rpp32f *inputTempF32;
        inputTempF32 = inputf32 + (i * descPtr->strides.nStride);

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

        srcLengthTensor[i] = sfinfo.frames;
        channelsTensor[i] = sfinfo.channels;

        int bufferLength = sfinfo.frames * sfinfo.channels;
        readcount = (int) sf_read_float (infile, inputTempF32, bufferLength);
        if (readcount != bufferLength)
        {
            std::cout << "Unable to read audio file: "<< audioFilesPath[j].c_str() << std::endl;
            exit(0);
        }

        // Close input
        sf_close (infile);
    }
}

void verify_output(Rpp32f *dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstDims, string testCase, string dst, string scriptPath)
{
    fstream refFile;
    int fileMatch = 0;

    // read data from golden outputs
    Rpp64u oBufferSize = dstDescPtr->n * dstDescPtr->strides.nStride;
    Rpp32f *refOutput = static_cast<Rpp32f *>(malloc(oBufferSize * sizeof(float)));
    string outFile = scriptPath + "/../REFERENCE_OUTPUTS_AUDIO/" + testCase + "/" + testCase + ".bin";
    std::fstream fin(outFile, std::ios::in | std::ios::binary);
    if(fin.is_open())
    {
        for(Rpp64u i = 0; i < oBufferSize; i++)
        {
            if(!fin.eof())
                fin.read(reinterpret_cast<char*>(&refOutput[i]), sizeof(float));
            else
            {
                std::cout<<"\nUnable to read all data from golden outputs\n";
                return;
            }
        }
    }
    else
    {
        std::cout<<"\nCould not open the reference output. Please check the path specified\n";
        return;
    }

    // iterate over all samples in a batch and compare with reference outputs
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        Rpp32f *dstPtrCurrent = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32f *refPtrCurrent = refOutput + batchCount * dstDescPtr->strides.nStride;
        Rpp32f *dstPtrRow = dstPtrCurrent;
        Rpp32f *refPtrRow = refPtrCurrent;
        Rpp32u hStride = dstDescPtr->strides.hStride;
        if (dstDims[batchCount].width == 1)
            hStride = 1;

        int matchedIndices = 0;
        for (int i = 0; i < dstDims[batchCount].height; i++)
        {
            Rpp32f *dstPtrTemp = dstPtrRow;
            Rpp32f *refPtrTemp = refPtrRow;
            for (int j = 0; j < dstDims[batchCount].width; j++)
            {
                Rpp32f refVal, outVal;
                refVal = refPtrTemp[j];
                outVal = dstPtrTemp[j];
                bool invalidComparision = ((outVal == 0.0f) && (refVal != 0.0f));
                if (!invalidComparision && abs(outVal - refVal) < 1e-20)
                    matchedIndices += 1;
            }
            dstPtrRow += hStride;
            refPtrRow += hStride;
        }
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

    free(refOutput);
}

void verify_non_silent_region_detection(float *detectedIndex, float *detectionLength, string testCase, int bs, vector<string> audioNames, string dst)
{
    int fileMatch = 0;
    for (int i = 0; i < bs; i++)
    {
        string currentFileName = audioNames[i];
        size_t lastIndex = currentFileName.find_last_of(".");
        currentFileName = currentFileName.substr(0, lastIndex);  // Remove extension from file name
        std::vector<int> referenceOutput = NonSilentRegionReferenceOutputs[currentFileName];
        if(referenceOutput.empty())
        {
            cout << "\nUnable to get the reference outputs for the file specified!" << endl;
            break;
        }
        Rpp32s outBegin = detectedIndex[i];
        Rpp32s outLength = detectionLength[i];
        Rpp32s refBegin = referenceOutput[0];
        Rpp32s refLength = referenceOutput[1];

        if ((outBegin == refBegin) && (outLength == refLength))
            fileMatch += 1;
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

inline Rpp32f sinc(Rpp32f x)
{
    x *= M_PI;
    return (std::abs(x) < 1e-5f) ? (1.0f - x * x * (1.0f / 6)) : std::sin(x) / x;
}

inline Rpp64f hann(Rpp64f x)
{
    return 0.5 * (1 + std::cos(x * M_PI));
}

// initialization function used for filling the values in Resampling window (RpptResamplingWindow)
// using the coeffs and lobes value this function generates a LUT (look up table) which is further used in Resample audio augmentation
inline void windowed_sinc(RpptResamplingWindow &window, Rpp32s coeffs, Rpp32s lobes)
{
    Rpp32f scale = 2.0f * lobes / (coeffs - 1);
    Rpp32f scale_envelope = 2.0f / coeffs;
    window.coeffs = coeffs;
    window.lobes = lobes;
    window.lookup.clear();
    window.lookup.resize(coeffs + 5);
    window.lookupSize = window.lookup.size();
    Rpp32s center = (coeffs - 1) * 0.5f;
    for (int i = 0; i < coeffs; i++) {
        Rpp32f x = (i - center) * scale;
        Rpp32f y = (i - center) * scale_envelope;
        Rpp32f w = sinc(x) * hann(y);
        window.lookup[i + 1] = w;
    }
    window.center = center + 1;
    window.scale = 1 / scale;
    window.pCenter = _mm_set1_ps(window.center);
    window.pScale = _mm_set1_ps(window.scale);
}