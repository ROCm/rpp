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

#include "../rpp_test_suite_audio.h"

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 6;
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_host_audio <src folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:0> <test type 0/1> <numRuns> <batchSize> <dst folder>\n");
        return -1;
    }

    char *src = argv[1];
    int inputBitDepth = atoi(argv[2]);
    int testCase = atoi(argv[3]);
    int testType = atoi(argv[4]);
    int numRuns = atoi(argv[5]);
    int batchSize = atoi(argv[6]);
    char *dst = argv[7];

    // Set case names
    string funcName = audioAugmentationMap[testCase];
    if (funcName.empty())
    {
        if (testType == 0)
            printf("\ncase %d is not supported\n", testCase);

        return -1;
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
    int i = 0, j = 0, fileCnt = 0;
    int maxChannels = 0;
    int maxSrcWidth = 0, maxSrcHeight = 0;
    int maxDstWidth = 0, maxDstHeight = 0;
    unsigned long long iBufferSize = 0;
    unsigned long long oBufferSize = 0;
    static int noOfAudioFiles = 0;

    // String ops on function name
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");
    string func = funcName;

    // Get number of audio files
    vector<string> audioNames, audioFilePath;
    search_files_recursive(src, audioNames, audioFilePath, ".wav");
    noOfAudioFiles = audioNames.size();
    if (noOfAudioFiles < batchSize || ((noOfAudioFiles % batchSize) != 0))
    {
        replicate_last_file_to_fill_batch(audioFilePath[noOfAudioFiles - 1], audioFilePath, audioNames, audioNames[noOfAudioFiles - 1], noOfAudioFiles, batchSize);
        noOfAudioFiles = audioNames.size();
    }

    // Initialize the AudioPatch for source
    Rpp32s *srcLengthTensor = (Rpp32s *) calloc(batchSize, sizeof(Rpp32s));
    Rpp32s *channelsTensor = (Rpp32s *) calloc(batchSize, sizeof(Rpp32s));
    RpptImagePatch *srcDims = (RpptImagePatch *) calloc(batchSize, sizeof(RpptImagePatch));
    RpptImagePatch *dstDims = (RpptImagePatch *) calloc(batchSize, sizeof(RpptImagePatch));

    // Find max audio dimensions in the input dataset
    maxSrcHeight = 1;
    maxDstHeight = 1;
    for (int cnt = 0; cnt < noOfAudioFiles ; cnt++)
    {
        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;

        // The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (!(infile = sf_open (audioFilePath[cnt].c_str(), SFM_READ, &sfinfo)))
        {
            sf_close (infile);
            continue;
        }

        maxSrcWidth = std::max(maxSrcWidth, static_cast<int>(sfinfo.frames));
        maxChannels = std::max(maxChannels, static_cast<int>(sfinfo.channels));

        // Close input
        sf_close (infile);
    }
    maxDstWidth = maxSrcWidth;

    // Set numDims, offset, n/c/h/w values for src/dst
    srcDescPtr->numDims = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n = batchSize;
    srcDescPtr->h = maxSrcHeight;
    srcDescPtr->w = maxSrcWidth;
    srcDescPtr->c = maxChannels;

    dstDescPtr->numDims = 4;
    dstDescPtr->offsetInBytes = 0;
    dstDescPtr->n = batchSize;
    dstDescPtr->h = maxDstHeight;
    dstDescPtr->w = maxDstWidth;
    if (testCase == 3)
        dstDescPtr->c = 1;
    else
        dstDescPtr->c = maxChannels;

    // Optionally set w stride as a multiple of 8 for src/dst
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
    iBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)srcDescPtr->n;
    oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;

    // Initialize host buffers for input & output
    Rpp32f *inputf32 = (Rpp32f *)calloc(iBufferSize, sizeof(Rpp32f));
    Rpp32f *outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, srcDescPtr->n, 8);
    int noOfIterations = (int)audioNames.size() / batchSize;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double cpuTime, wallTime;
    string testCaseName;
    printf("\nRunning %s %d times (each time with a batch size of %d images) and computing mean statistics...", func.c_str(), numRuns, batchSize);
    for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
    {
        for (int iterCount = 0; iterCount < noOfIterations; iterCount++)
        {
            for (int cnt = 0; cnt < batchSize; cnt++)
            {
                Rpp32f *inputTempF32;
                inputTempF32 = inputf32 + (cnt * srcDescPtr->strides.nStride);

                SNDFILE	*infile;
                SF_INFO sfinfo;
                int	readcount;

                // The SF_INFO struct must be initialized before using it
                memset (&sfinfo, 0, sizeof (sfinfo));
                if (!(infile = sf_open (audioFilePath[fileCnt].c_str(), SFM_READ, &sfinfo)))
                {
                    sf_close (infile);
                    continue;
                }

                srcLengthTensor[cnt] = sfinfo.frames;
                channelsTensor[cnt] = sfinfo.channels;
                srcDims[cnt].width = sfinfo.frames;
                dstDims[cnt].width = sfinfo.frames;
                srcDims[cnt].height = 1;
                dstDims[cnt].height = 1;

                int bufferLength = sfinfo.frames * sfinfo.channels;
                if (inputBitDepth == 2)
                {
                    readcount = (int) sf_read_float (infile, inputTempF32, bufferLength);
                    if (readcount != bufferLength)
                        cout << "F32 Unable to read audio file completely " << std::endl;
                }
                fileCnt++;

                // Close input
                sf_close (infile);
            }
            clock_t startCpuTime, endCpuTime;
            double startWallTime, endWallTime;
            switch (testCase)
            {
                case 0:
                {
                    testCaseName = "non_silent_region_detection";
                    Rpp32f detectedIndex[batchSize];
                    Rpp32f detectionLength[batchSize];
                    Rpp32f cutOffDB = -60.0;
                    Rpp32s windowLength = 2048;
                    Rpp32f referencePower = 0.0f;
                    Rpp32s resetInterval = 8192;

                    startWallTime = omp_get_wtime();
                    startCpuTime= clock();
                    if (inputBitDepth == 2)
                        rppt_non_silent_region_detection_host(inputf32, srcDescPtr, srcLengthTensor, detectedIndex, detectionLength, cutOffDB, windowLength, referencePower, resetInterval, handle);
                    else
                        missingFuncFlag = 1;

                    if (testType == 0)
                        verify_non_silent_region_detection(detectedIndex, detectionLength, testCaseName, batchSize, audioNames, dst);

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
            cpuTime = ((double)(endCpuTime - startCpuTime)) / CLOCKS_PER_SEC;
            wallTime = endWallTime - startWallTime;
            if (missingFuncFlag == 1)
            {
                printf("\nThe functionality %s doesn't yet exist in RPP\n", func.c_str());
                return -1;
            }
            maxWallTime = std::max(maxWallTime, wallTime);
            minWallTime = std::min(minWallTime, wallTime);
            avgWallTime += wallTime;
            cpuTime *= 1000;
            wallTime *= 1000;

            if (testType == 0)
            {
                if (batchSize == 8 && testCase != 0)
                    verify_output(outputf32, dstDescPtr, dstDims, testCaseName, audioNames, dst);

                cout <<"\n\n";
                cout <<"CPU Backend Clock Time: "<< cpuTime <<" ms/batch"<< endl;
                cout <<"CPU Backend Wall Time: "<< wallTime <<" ms/batch"<< endl;

                // If DEBUG_MODE is set to 1 dump the outputs to csv files for debugging
                if (DEBUG_MODE && iterCount == 0 && testCase != 0)
                {
                    std::ofstream refFile;
                    refFile.open(func + ".csv");
                    for (int i = 0; i < oBufferSize; i++)
                        refFile << *(outputf32 + i) << "\n";
                    refFile.close();
                }
            }
        }

        // Reset fileIndex to 0 for next run
        fileCnt = 0;
    }
    rppDestroyHost(handle);

    if (testType == 1)
    {
        // Display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns * noOfIterations);
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }

    cout<<endl;

    // Free memory
    free(srcLengthTensor);
    free(channelsTensor);
    free(srcDims);
    free(dstDims);
    free(inputf32);
    free(outputf32);
    return 0;
}