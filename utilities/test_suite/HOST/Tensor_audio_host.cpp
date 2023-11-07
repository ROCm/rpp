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
        printf("\nUsage: ./Tensor_audio_host <src folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:0> <test type 0/1> <numRuns> <batchSize> <dst folder>\n");
        return -1;
    }

    char *src = argv[1];
    int inputBitDepth = atoi(argv[2]);
    int testCase = atoi(argv[3]);
    int testType = atoi(argv[4]);
    int numRuns = atoi(argv[5]);
    int batchSize = atoi(argv[6]);
    char *dst = argv[7];

    if (testType == 0 && batchSize != 8)
    {
        cout << "Error! QA Mode only runs with batchsize 8" << endl;
        return -1;
    }

    // Set case names
    string funcName = audioAugmentationMap[testCase];
    if (funcName.empty())
    {
        if (testType == 0)
            printf("\ncase %d is not supported\n", testCase);

        return -1;
    }

    // Initialize tensor descriptors
    RpptGenericDesc srcDescriptor, dstDescriptor;
    RpptGenericDescPtr srcDescriptorPtrND, dstDescriptorPtrND;
    srcDescriptorPtrND = &srcDescriptor;
    dstDescriptorPtrND = &dstDescriptor;
    int nDim = 2;

    // Set src/dst data types in tensor descriptors
    if (inputBitDepth == 2)
    {
        srcDescriptorPtrND->dataType = RpptDataType::F32;
        dstDescriptorPtrND->dataType = RpptDataType::F32;
    }

    srcDescriptorPtrND->layout = RpptLayout::NHWC;
    dstDescriptorPtrND->layout = RpptLayout::NHWC;

    // Other initializations
    int missingFuncFlag = 0;
    int maxSrcChannels = 0;
    int maxSrcWidth = 0, maxSrcHeight = 0;
    int maxDstWidth = 0, maxDstHeight = 0;
    static int noOfAudioFiles = 0;

    // String ops on function name
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");
    string func = funcName;

    // Get number of audio files
    vector<string> audioNames, audioFilesPath;
    search_files_recursive(src, audioNames, audioFilesPath, ".wav");
    noOfAudioFiles = audioNames.size();
    if (noOfAudioFiles < batchSize || ((noOfAudioFiles % batchSize) != 0))
    {
        replicate_last_file_to_fill_batch(audioFilesPath[noOfAudioFiles - 1], audioFilesPath, audioNames, audioNames[noOfAudioFiles - 1], noOfAudioFiles, batchSize);
        noOfAudioFiles = audioNames.size();
    }

    // Initialize roi Tensor for audio
    Rpp32u *roiTensor = (Rpp32u *)calloc(4 * batchSize, sizeof(Rpp32u));

    // Find max audio dimensions in the input dataset
    set_audio_max_dimensions(audioFilesPath, maxSrcWidth, maxSrcChannels);
    maxDstWidth = maxSrcWidth;

    // Set numDims, offset, n/c/h/w values for src/dst
    Rpp32u offsetInBytes = 0;
    set_audio_descriptor_dims_and_strides(srcDescriptorPtrND, batchSize, maxSrcWidth, maxSrcChannels, offsetInBytes);
    int maxDstChannels = maxSrcChannels;
    if(testCase == 3)
        maxDstChannels = 1;
    set_audio_descriptor_dims_and_strides(dstDescriptorPtrND, batchSize, maxDstWidth, maxDstChannels, offsetInBytes);

    Rpp64u numValues = 1;
    for(int i = 0; i <= nDim; i++)
        numValues *= srcDescriptorPtrND->dims[i];

    // Initialize host buffers for input & output
    Rpp32f *inputF32 = (Rpp32f *)calloc(numValues, sizeof(Rpp32f));
    Rpp32f *outputF32 = (Rpp32f *)calloc(numValues, sizeof(Rpp32f));

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, srcDescriptorPtrND->dims[0], 8);
    int noOfIterations = (int)audioNames.size() / batchSize;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    string testCaseName;
    printf("\nRunning %s %d times (each time with a batch size of %d images) and computing mean statistics...", func.c_str(), numRuns, batchSize);
    for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
    {
        for (int iterCount = 0; iterCount < noOfIterations; iterCount++)
        {
            // Read and decode audio and fill the audio dim values
            if (inputBitDepth == 2)
                read_audio_batch_and_fill_dims(srcDescriptorPtrND, inputF32, audioFilesPath, iterCount, roiTensor);

            double startWallTime, endWallTime;
            double wallTime;
            switch (testCase)
            {
                case 8:
                {
                    testCaseName = "normalize";
                    int axisMask = 1;
                    float scale = 1.0;
                    float shift = 0.0;
                    Rpp32u size = 1; // length of input tensors differ based on axisMask and nDim
                    Rpp32u maxSize = 1;
                    for(int batch = 0; batch < batchSize; batch++)
                    {
                        size = 1;
                        for(int i = 0; i < nDim; i++)
                            size *= ((axisMask & (int)(pow(2,i))) >= 1) ? 1 : roiTensor[(nDim * 2 * batch) + nDim + i];
                        maxSize = max(maxSize, size);
                    }
                    bool computeMean, computeStddev;
                    computeMean = computeStddev = 1;

                    Rpp32f *meanTensor = (Rpp32f *)calloc(maxSize * batchSize, sizeof(Rpp32f));
                    Rpp32f *stdDevTensor = (Rpp32f *)calloc(maxSize * batchSize, sizeof(Rpp32f));

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 2)
                    {
                        rppt_normalize_generic_host(inputF32, srcDescriptorPtrND, outputF32, dstDescriptorPtrND, axisMask, meanTensor, stdDevTensor, computeMean, computeStddev, scale, shift, roiTensor, handle);
                        free(meanTensor);
                        free(stdDevTensor);
                    }
                    else
                        missingFuncFlag = 1;

                    // QA mode - verify outputs with golden outputs. Below code doesn’t run for performance tests
                    if (testType == 0)
                        verify_output(outputF32, dstDescriptorPtrND, roiTensor, testCaseName, audioNames, dst);
                    break;
                }
                default:
                {
                    missingFuncFlag = 1;
                    break;
                }
            }

            endWallTime = omp_get_wtime();
            if (missingFuncFlag == 1)
            {
                printf("\nThe functionality %s doesn't yet exist in RPP\n", func.c_str());
                return -1;
            }

            wallTime = endWallTime - startWallTime;
            maxWallTime = std::max(maxWallTime, wallTime);
            minWallTime = std::min(minWallTime, wallTime);
            avgWallTime += wallTime;
        }
    }
    rppDestroyHost(handle);

    // performance test mode
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
    free(roiTensor);
    free(inputF32);
    free(outputF32);
    return 0;
}