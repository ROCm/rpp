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
#include <iostream>
#include "rpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half/half.hpp>
#include <fstream>
#include <iomanip>

// Include this header file to use functions from libsndfile
#include <sndfile.h>

using namespace std;
using half_float::half;

typedef half Rpp16f;

void remove_substring(string &str, string &pattern)
{
    std::string::size_type i = str.find(pattern);
    while (i != std::string::npos)
    {
        str.erase(i, pattern.length());
        i = str.find(pattern, i);
   }
}

void verify_non_silent_region_detection(float *detectedIndex, float *detectionLength, string testCase, int bs, char audioNames[][1000])
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "HOST/audio/build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "REFERENCE_OUTPUTS_AUDIO/";
    int file_match = 0;
    for (int i = 0; i < bs; i++)
    {
        string current_file_name = audioNames[i];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + testCase + "/" + testCase + "_ref_" + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);
        if(!ref_file.is_open())
        {
            cerr<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;
            break;
        }

        Rpp32s ref_index, ref_length;
        Rpp32s out_index, out_length;
        ref_file>>ref_index;
        ref_file>>ref_length;
        out_index = detectedIndex[i];
        out_length = detectionLength[i];

        if((out_index == ref_index) && (out_length == ref_length))
            file_match += 1;
        ref_file.close();
    }
    std::cerr<<std::endl<<"Results for Test case: "<<testCase<<std::endl;
    if(file_match == bs)
        std::cerr<<"PASSED!"<<std::endl;
    else
        std::cerr<<"FAILED! "<<file_match<<"/"<<bs<<" outputs are matching with reference outputs"<<std::endl;
}

void verify_output(Rpp32f *dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstDims, string test_case, char audioNames[][1000])
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "HOST/audio/build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "REFERENCE_OUTPUTS_AUDIO/";
    int file_match = 0;
    for (int batchcount = 0; batchcount < dstDescPtr->n; batchcount++)
    {
        string current_file_name = audioNames[batchcount];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + test_case + "/" + test_case + "_ref_" + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);
        if(!ref_file.is_open())
        {
            cerr<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;
            break;
        }
        int matched_indices = 0;
        Rpp32f ref_val, out_val;
        Rpp32f *dstPtrCurrent = dstPtr + batchcount * dstDescPtr->strides.nStride;
        Rpp32f *dstPtrRow = dstPtrCurrent;
        for(int i = 0; i < dstDims[batchcount].height; i++)
        {
            Rpp32f *dstPtrTemp = dstPtrRow;
            for(int j = 0; j < dstDims[batchcount].width; j++)
            {
                ref_file>>ref_val;
                out_val = dstPtrTemp[j];
                bool invalid_comparision = ((out_val == 0.0f) && (ref_val != 0.0f));
                if(!invalid_comparision && abs(out_val - ref_val) < 1e-20)
                    matched_indices += 1;
            }
            dstPtrRow += dstDescPtr->strides.hStride;
        }
        ref_file.close();
        if(matched_indices == (dstDims[batchcount].width * dstDims[batchcount].height) && matched_indices !=0)
            file_match++;
    }

    std::cerr<<std::endl<<"Results for Test case: "<<test_case<<std::endl;
    if(file_match == dstDescPtr->n)
        std::cerr<<"PASSED!"<<std::endl;
    else
        std::cerr<<"FAILED! "<<file_match<<"/"<<dstDescPtr->n<<" outputs are matching with reference outputs"<<std::endl;
}

void read_from_text_files(Rpp32f *srcPtr, RpptDescPtr srcDescPtr, RpptImagePatch *srcDims, string test_case, int read_type, char audioNames[][1000])
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "HOST/audio/build";
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

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 3;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_host_audio <src folder> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:9>\n");
        return -1;
    }

    char *src = argv[1];
    int inputBitDepth = atoi(argv[2]);
    int testCase = atoi(argv[3]);

    // Set case names
    char funcName[1000];
    switch (testCase)
    {
        case 0:
            strcpy(funcName, "non_silent_region_detection");
            break;
        case 1:
            strcpy(funcName, "to_decibels");
            break;
        case 2:
            strcpy(funcName, "pre_emphasis_filter");
            break;
        case 3:
            strcpy(funcName, "down_mixing");
            break;
        case 4:
            strcpy(funcName, "slice_audio");
            break;
        case 5:
            strcpy(funcName, "mel_filter_bank");
            break;
        case 6:
            strcpy(funcName, "spectrogram");
            break;
        default:
            strcpy(funcName, "testCase");
            break;
    }

    // Initialize tensor descriptors
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    // Set src/dst data types in tensor descriptors
    if (inputBitDepth == 2)
    {
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    }

    // Other initializations
    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxChannels = 0;
    int maxSrcWidth = 0, maxSrcHeight = 0;
    int maxDstWidth = 0, maxDstHeight = 0;
    Rpp64u count = 0;
    Rpp64u iBufferSize = 0;
    Rpp64u oBufferSize = 0;
    static int noOfAudioFiles = 0;

    // String ops on function name
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");

    char func[1000];
    strcpy(func, funcName);
    printf("\nRunning %s...", func);

    // Get number of audio files
    struct dirent *de;
    DIR *dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfAudioFiles += 1;
    }
    closedir(dr);

    // Initialize the AudioPatch for source
    Rpp32s *srcLengthTensor = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));
    Rpp32s *channelsTensor = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));
    RpptImagePatch *srcDims = (RpptImagePatch *) calloc(noOfAudioFiles, sizeof(RpptImagePatch));
    RpptImagePatch *dstDims = (RpptImagePatch *) calloc(noOfAudioFiles, sizeof(RpptImagePatch));

    // Set maxLength
    char audioNames[noOfAudioFiles][1000];

    // Set Height as 1 for src, dst
    maxSrcHeight = 1;
    maxDstHeight = 1;

    dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        strcpy(audioNames[count], de->d_name);
        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, audioNames[count]);

        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;

        //The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (!(infile = sf_open (temp, SFM_READ, &sfinfo)))
        {
            sf_close (infile);
            continue;
        }

        srcLengthTensor[count] = sfinfo.frames;
        channelsTensor[count] = sfinfo.channels;

        srcDims[count].width = sfinfo.frames;
        dstDims[count].width = sfinfo.frames;
        srcDims[count].height = 1;
        dstDims[count].height = 1;

        maxSrcWidth = std::max(maxSrcWidth, srcLengthTensor[count]);
        maxDstWidth = std::max(maxDstWidth, srcLengthTensor[count]);
        maxChannels = std::max(maxChannels, channelsTensor[count]);

        // Close input
        sf_close (infile);
        count++;
    }
    closedir(dr);

    // Set numDims, offset, n/c/h/w values for src/dst
    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    srcDescPtr->offsetInBytes = 0;
    dstDescPtr->offsetInBytes = 0;

    srcDescPtr->n = noOfAudioFiles;
    dstDescPtr->n = noOfAudioFiles;

    srcDescPtr->h = maxSrcHeight;
    dstDescPtr->h = maxDstHeight;

    srcDescPtr->w = maxSrcWidth;
    dstDescPtr->w = maxDstWidth;

    srcDescPtr->c = maxChannels;
    //if(testCase == 3)
        //dstDescPtr->c = 1;
    //else
        dstDescPtr->c = maxChannels;

    // Optionally set w stride as a multiple of 8 for src
    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
    srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
    srcDescPtr->strides.wStride = srcDescPtr->c;
    srcDescPtr->strides.cStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
    dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
    dstDescPtr->strides.wStride = dstDescPtr->c;
    dstDescPtr->strides.cStride = 1;

    // Set buffer sizes for src/dst
    iBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)srcDescPtr->n;
    oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)dstDescPtr->n;

    // Initialize host buffers for input & output
    Rpp32f *inputf32 = (Rpp32f *)calloc(iBufferSize, sizeof(Rpp32f));
    Rpp32f *outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));

    i = 0;
    count = 0;
    dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        Rpp32f *input_temp_f32;
        input_temp_f32 = inputf32 + (i * srcDescPtr->strides.nStride);

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        strcpy(audioNames[count], de->d_name);
        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, audioNames[count]);

        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;

        // The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (!(infile = sf_open (temp, SFM_READ, &sfinfo)))
        {
            sf_close (infile);
            continue;
        }

        int bufferLength = sfinfo.frames * sfinfo.channels;
        if(inputBitDepth == 2)
        {
            readcount = (int) sf_read_float (infile, input_temp_f32, bufferLength);
            if(readcount != bufferLength)
                std::cerr<<"F32 Unable to read audio file completely"<<std::endl;
        }
        i++;
        count++;

        // Close input
        sf_close (infile);
    }
    closedir(dr);

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, srcDescPtr->n, 8);
    clock_t startCpuTime, endCpuTime;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double startWallTime, endWallTime;
    double cpuTime, wallTime;

    string testCaseName;
    switch (testCase)
    {
        case 0:
        {
            testCaseName = "non_silent_region_detection";
            Rpp32f detectedIndex[noOfAudioFiles];
            Rpp32f detectionLength[noOfAudioFiles];
            Rpp32f cutOffDB = -60.0;
            Rpp32s windowLength = 2048;
            Rpp32f referencePower = 0.0f;
            Rpp32s resetInterval = 8192;

            startWallTime = omp_get_wtime();
            startCpuTime = clock();
            if (inputBitDepth == 2)
            {
                rppt_non_silent_region_detection_host(inputf32, srcDescPtr, srcLengthTensor, detectedIndex, detectionLength, cutOffDB, windowLength, referencePower, resetInterval, handle);
            }
            else
                missingFuncFlag = 1;

            verify_non_silent_region_detection(detectedIndex, detectionLength, testCaseName, noOfAudioFiles, audioNames);
            break;
        }
        case 1:
        {
            testCaseName = "to_decibels";
            Rpp32f cutOffDB = std::log(1e-20);
            Rpp32f multiplier = std::log(10);
            Rpp32f referenceMagnitude = 1.0f;

            for (i = 0; i < noOfAudioFiles; i++)
            {
                srcDims[i].height = srcLengthTensor[i];
                srcDims[i].width = 1;
            }

            startWallTime = omp_get_wtime();
            startCpuTime = clock();
            if (inputBitDepth == 2)
            {
                rppt_to_decibels_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcDims, cutOffDB, multiplier, referenceMagnitude, handle);
            }
            else
                missingFuncFlag = 1;

            verify_output(outputf32, dstDescPtr, dstDims, testCaseName, audioNames);
            break;
        }
        case 2:
        {
            testCaseName = "pre_emphasis_filter";
            Rpp32f coeff[noOfAudioFiles];
            for (i = 0; i < noOfAudioFiles; i++)
                coeff[i] = 0.97;
            RpptAudioBorderType borderType = RpptAudioBorderType::CLAMP;

            startWallTime = omp_get_wtime();
            startCpuTime = clock();
            if (inputBitDepth == 2)
            {
                rppt_pre_emphasis_filter_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, coeff, borderType, handle);
            }
            else
                missingFuncFlag = 1;

            verify_output(outputf32, dstDescPtr, dstDims, testCaseName, audioNames);
            break;
        }
        case 3:
        {
            testCaseName = "down_mixing";
            bool normalizeWeights = false;

            startWallTime = omp_get_wtime();
            startCpuTime = clock();
            if (inputBitDepth == 2)
            {
                rppt_down_mixing_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, channelsTensor, normalizeWeights, handle);
            }
            else
                missingFuncFlag = 1;

            verify_output(outputf32, dstDescPtr, dstDims, testCaseName, audioNames);
            break;
        }
        case 4:
        {
            testCaseName = "slice_audio";

            Rpp32f fillValues[noOfAudioFiles];
            Rpp32s srcDimsTensor[noOfAudioFiles * 2];
            Rpp32f anchor[noOfAudioFiles];
            Rpp32f shape[noOfAudioFiles];

            // 1D slice arguments
            for (i = 0, j = i * 2; i < noOfAudioFiles; i++, j += 2)
            {
                srcDimsTensor[j] = srcLengthTensor[i];
                srcDimsTensor[j + 1] = 1;
                shape[i] =  dstDims[i].width = 200;
                anchor[i] = 100;
            }
            fillValues[0] = 0.0f;

            startWallTime = omp_get_wtime();
            startCpuTime = clock();
            if (inputBitDepth == 2)
            {
                rppt_slice_audio_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcDimsTensor, anchor, shape, fillValues, handle);
            }
            else
                missingFuncFlag = 1;

            verify_output(outputf32, dstDescPtr, dstDims, testCaseName, audioNames);
            break;
        }
        case 5:
        {
            testCaseName = "mel_filter_bank";

            Rpp32f sampleRate = 16000;
            Rpp32f minFreq = 0.0;
            Rpp32f maxFreq = sampleRate / 2;
            RpptMelScaleFormula melFormula = RpptMelScaleFormula::SLANEY;
            Rpp32s numFilter = 80;
            bool normalize = true;

            // Read source dimension
            read_from_text_files(inputf32, srcDescPtr, srcDims, "spectrogram", 1, audioNames);

            maxDstHeight = 0;
            maxDstWidth = 0;
            maxSrcHeight = 0;
            maxSrcWidth = 0;
            for(int i = 0; i < noOfAudioFiles; i++)
            {
                maxSrcHeight = std::max(maxSrcHeight, (int)srcDims[i].height);
                maxSrcWidth = std::max(maxSrcWidth, (int)srcDims[i].width);
                dstDims[i].height = numFilter;
                dstDims[i].width = srcDims[i].width;
                maxDstHeight = std::max(maxDstHeight, (int)dstDims[i].height);
                maxDstWidth = std::max(maxDstWidth, (int)dstDims[i].width);
            }

            srcDescPtr->h = maxSrcHeight;
            srcDescPtr->w = maxSrcWidth;
            dstDescPtr->h = maxDstHeight;
            dstDescPtr->w = maxDstWidth;

            srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
            srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
            srcDescPtr->strides.wStride = srcDescPtr->c;
            srcDescPtr->strides.cStride = 1;

            dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
            dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
            dstDescPtr->strides.wStride = dstDescPtr->c;
            dstDescPtr->strides.cStride = 1;

            // Set buffer sizes for src/dst
            unsigned long long spectrogramBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)srcDescPtr->n;
            unsigned long long melFilterBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
            inputf32 = (Rpp32f *)realloc(inputf32, spectrogramBufferSize * sizeof(Rpp32f));
            outputf32 = (Rpp32f *)realloc(outputf32, melFilterBufferSize * sizeof(Rpp32f));

            // Read source data
            read_from_text_files(inputf32, srcDescPtr, srcDims, "spectrogram", 0, audioNames);

            startWallTime = omp_get_wtime();
            startCpuTime = clock();
            if (inputBitDepth == 2)
            {
                rppt_mel_filter_bank_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcDims, maxFreq, minFreq, melFormula, numFilter, sampleRate, normalize, handle);
            }
            else
                missingFuncFlag = 1;

            verify_output(outputf32, dstDescPtr, dstDims, testCaseName, audioNames);
            break;
        }
        case 6:
        {
            testCaseName = "spectrogram";

            bool centerWindows = true;
            bool reflectPadding = true;
            Rpp32f *windowFn = NULL;
            Rpp32s power = 2;
            Rpp32s windowLength = 320;
            Rpp32s windowStep = 160;
            Rpp32s nfft = 512;
            RpptSpectrogramLayout layout = RpptSpectrogramLayout::FT;

            int windowOffset = 0;
            if(!centerWindows)
                windowOffset = windowLength;

            maxDstWidth = 0;
            maxDstHeight = 0;
            if(layout == RpptSpectrogramLayout::FT)
            {
                for(int i = 0; i < noOfAudioFiles; i++)
                {
                    dstDims[i].height = nfft / 2 + 1;
                    dstDims[i].width = ((srcLengthTensor[i] - windowOffset) / windowStep) + 1;
                    maxDstHeight = std::max(maxDstHeight, (int)dstDims[i].height);
                    maxDstWidth = std::max(maxDstWidth, (int)dstDims[i].width);
                }
            }
            else
            {
                for(int i = 0; i < noOfAudioFiles; i++)
                {
                    dstDims[i].height = ((srcLengthTensor[i] - windowOffset) / windowStep) + 1;
                    dstDims[i].width = nfft / 2 + 1;
                    maxDstHeight = std::max(maxDstHeight, (int)dstDims[i].height);
                    maxDstWidth = std::max(maxDstWidth, (int)dstDims[i].width);
                }
            }

            dstDescPtr->w = maxDstWidth;
            dstDescPtr->h = maxDstHeight;

            dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
            dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
            dstDescPtr->strides.wStride = dstDescPtr->c;
            dstDescPtr->strides.cStride = 1;

            // Set buffer sizes for src/dst
            unsigned long long spectrogramBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
            outputf32 = (Rpp32f *)realloc(outputf32, spectrogramBufferSize * sizeof(Rpp32f));

            startWallTime = omp_get_wtime();
            startCpuTime = clock();
            if (inputBitDepth == 2)
            {
                rppt_spectrogram_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, centerWindows, reflectPadding, windowFn, nfft, power, windowLength, windowStep, layout, handle);
            }
            else
                missingFuncFlag = 1;

            verify_output(outputf32, dstDescPtr, dstDims, testCaseName, audioNames);
            break;
        }
        default:
        {
            missingFuncFlag = 1;
            break;
        }
    }

    endCpuTime = clock();
    endWallTime = omp_get_wtime();

    if (missingFuncFlag == 1)
    {
        printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
        return -1;
    }

    // Display measured times
    cpuTime = ((double)(endCpuTime - startCpuTime)) / CLOCKS_PER_SEC;
    wallTime = endWallTime - startWallTime;
    cout <<"CPU Backend Clock Time: "<< cpuTime <<" ms/batch"<< endl;
    cout <<"CPU Backend Wall Time: "<< wallTime <<" ms/batch"<< endl;
    printf("\n");

    rppDestroyHost(handle);

    // Free memory
    free(srcLengthTensor);
    free(channelsTensor);
    free(srcDims);
    free(dstDims);
    free(inputf32);
    free(outputf32);

    return 0;
}