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
    {4, "spectrogram"},
    {5, "mel_filter_bank"}
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

// sets descriptor dimensions and strides of src/dst
inline void set_audio_descriptor_dims_and_strides_nostriding(RpptDescPtr descPtr, int batchSize, int maxHeight, int maxWidth, int maxChannels, int offsetInBytes)
{
    descPtr->numDims = 4;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->n = batchSize;
    descPtr->h = maxHeight;
    descPtr->w = maxWidth;
    descPtr->c = maxChannels;

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

void read_from_text_files(Rpp32f *srcPtr, RpptDescPtr srcDescPtr, RpptImagePatch *srcDims, string test_case, int read_type, vector<std::string> audioNames)
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "HOST/build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "REFERENCE_OUTPUTS_AUDIO/";

    string read_type_str;
    if(read_type == 0)
        read_type_str = "_ref_";
    else
        read_type_str = "_info_";

    for (int batchcount = 0; batchcount < srcDescPtr->n; batchcount++)
    {
        string current_file_name = audioNames[batchcount];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + test_case + "/" + test_case + read_type_str + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);
        if(!ref_file.is_open())
        {
            cerr<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;
            break;
        }

        if(read_type == 0)
        {
            Rpp32f ref_val;
            Rpp32f *srcPtrCurrent = srcPtr + batchcount * srcDescPtr->strides.nStride;
            Rpp32f *srcPtrRow = srcPtrCurrent;
            for(int i = 0; i < srcDims[batchcount].height; i++)
            {
                Rpp32f *srcPtrTemp = srcPtrRow;
                for(int j = 0; j < srcDims[batchcount].width; j++)
                {
                    ref_file>>ref_val;
                    srcPtrTemp[j] = ref_val;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
        }
        else
        {
            Rpp32s ref_height, ref_width;
            ref_file>>ref_height;
            ref_file>>ref_width;
            srcDims[batchcount].height = ref_height;
            srcDims[batchcount].width = ref_width;
        }
        ref_file.close();
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