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

#include "../rpp_test_suite_audio.h"

int main(int argc, char **argv)
{
    // handle inputs
    const int MIN_ARG_COUNT = 7;
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_hip_audio <src folder> <case number = 0:0> <test type 0/1> <numRuns> <batchSize> <dst folder>\n");
        return -1;
    }

    char *src = argv[1];
    int testCase = atoi(argv[2]);
    int testType = atoi(argv[3]);
    int numRuns = atoi(argv[4]);
    int batchSize = atoi(argv[5]);
    char *dst = argv[6];
    string scriptPath = argv[7];

    // validation checks
    if (testType == 0 && batchSize != 3)
    {
        cout << "Error! QA Mode only runs with batchsize 3" << endl;
        return -1;
    }

    // set case names
    string funcName = audioAugmentationMap[testCase];
    if (funcName.empty())
    {
        if (testType == 0)
            printf("\ncase %d is not supported\n", testCase);

        return -1;
    }

    // initialize tensor descriptors
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    // set src/dst data types in tensor descriptors
    srcDescPtr->dataType = RpptDataType::F32;
    dstDescPtr->dataType = RpptDataType::F32;

    // other initializations
    int missingFuncFlag = 0;
    int maxSrcChannels = 0;
    int maxSrcWidth = 0, maxSrcHeight = 0;
    int maxDstWidth = 0, maxDstHeight = 0;
    Rpp64u iBufferSize = 0;
    Rpp64u oBufferSize = 0;
    static int noOfAudioFiles = 0;

    // string ops on function name
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");
    string func = funcName;

    // get number of audio files
    vector<string> audioNames, audioFilesPath;
    search_files_recursive(src, audioNames, audioFilesPath, ".wav");
    noOfAudioFiles = audioNames.size();
    if (noOfAudioFiles < batchSize || ((noOfAudioFiles % batchSize) != 0))
    {
        replicate_last_file_to_fill_batch(audioFilesPath[noOfAudioFiles - 1], audioFilesPath, audioNames, audioNames[noOfAudioFiles - 1], noOfAudioFiles, batchSize);
        noOfAudioFiles = audioNames.size();
    }

    // find max audio dimensions in the input dataset
    maxSrcHeight = 1;
    maxDstHeight = 1;
    set_audio_max_dimensions(audioFilesPath, maxSrcWidth, maxSrcChannels);
    maxDstWidth = maxSrcWidth;

    // set numDims, offset, n/c/h/w values for src/dst
    Rpp32u offsetInBytes = 0;
    set_audio_descriptor_dims_and_strides(srcDescPtr, batchSize, maxSrcHeight, maxSrcWidth, maxSrcChannels, offsetInBytes);
    int maxDstChannels = maxSrcChannels;
    if(testCase == 3)
        maxDstChannels = 1;
    set_audio_descriptor_dims_and_strides(dstDescPtr, batchSize, maxDstHeight, maxDstWidth, maxDstChannels, offsetInBytes);
    srcDescPtr->numDims = 2;
    dstDescPtr->numDims = 2;
    // set buffer sizes for src/dst
    iBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)srcDescPtr->n;
    oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)dstDescPtr->n;

    // allocate hip buffers for input & output
    Rpp32f *inputf32 = (Rpp32f *)calloc(iBufferSize, sizeof(Rpp32f));
    Rpp32f *outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));

    void *d_inputf32, *d_outputf32;
    CHECK(hipMalloc(&d_inputf32, iBufferSize * sizeof(Rpp32f)));
    CHECK(hipMalloc(&d_outputf32, oBufferSize * sizeof(Rpp32f)));

    // allocate the buffers for audio length and channels
    Rpp32s *srcLengthTensor, *channelsTensor;
    CHECK(hipHostMalloc(&srcLengthTensor, batchSize * sizeof(Rpp32s)));
    CHECK(hipHostMalloc(&channelsTensor, batchSize * sizeof(Rpp32s)));

    // allocate the buffers for src/dst dimensions for each element in batch
    RpptImagePatch *srcDims = (RpptImagePatch *) calloc(batchSize, sizeof(RpptImagePatch));
    RpptImagePatch *dstDims = (RpptImagePatch *) calloc(batchSize, sizeof(RpptImagePatch));

    // allocate the buffer for srcDimsTensor
    Rpp32s *srcDimsTensor;
    CHECK(hipHostMalloc(&srcDimsTensor, batchSize * 2 * sizeof(Rpp32s)));

    Rpp32f *detectedIndex = nullptr, *detectionLength = nullptr;
    if(testCase == 0)
    {
        CHECK(hipHostMalloc(&detectedIndex, batchSize * sizeof(Rpp32f)));
        CHECK(hipHostMalloc(&detectionLength, batchSize * sizeof(Rpp32f)));
    }

    // run case-wise RPP API and measure time
    rppHandle_t handle;
    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));
    rppCreateWithStreamAndBatchSize(&handle, stream, batchSize);

    int noOfIterations = (int)audioNames.size() / batchSize;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    string testCaseName;
    printf("\nRunning %s %d times (each time with a batch size of %d images) and computing mean statistics...", func.c_str(), numRuns, batchSize);
    for (int iterCount = 0; iterCount < 1; iterCount++)
    {
        // read and decode audio and fill the audio dim values
        read_audio_batch_and_fill_dims(srcDescPtr, inputf32, audioFilesPath, iterCount, srcLengthTensor, channelsTensor);
        CHECK(hipMemcpy(d_inputf32, inputf32, iBufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
        for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
        {
            double startWallTime, endWallTime;
            double wallTime;
            switch (testCase)
            {
                case 6:
                {
                    testCaseName = "resample";
                    Rpp32f inRateTensor[batchSize];
                    Rpp32f outRateTensor[batchSize];
                    Rpp32s srcDimsTensor[batchSize * 2];

                    maxDstWidth = 0;
                    for(int i = 0, j = 0; i < batchSize; i++, j += 2)
                    {
                        inRateTensor[i] = 16000;
                        outRateTensor[i] = 16000 * 1.15f;
                        Rpp32f scaleRatio = outRateTensor[i] / inRateTensor[i];
                        srcDimsTensor[j] = srcLengthTensor[i];
                        srcDimsTensor[j + 1] = channelsTensor[i];
                        dstDims[i].width = static_cast<int>(std::ceil(scaleRatio * srcLengthTensor[i]));
                        dstDims[i].height = 1;
                        maxDstWidth = std::max(maxDstWidth, static_cast<int>(dstDims[i].width));
                    }
                    Rpp32f quality = 50.0f;
                    Rpp32s lobes = std::round(0.007 * quality * quality - 0.09 * quality + 3);
                    Rpp32s lookupSize = lobes * 64 + 1;
                    RpptResamplingWindow *window;
                    CHECK(hipHostMalloc(&window, sizeof(RpptResamplingWindow)));
                    CHECK(hipHostMalloc(&window->lookup, (lookupSize + 5) * sizeof(Rpp32f)));
                    windowed_sinc(*window, lookupSize, lobes);

                    dstDescPtr->w = maxDstWidth;
                    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;

                    // Set buffer sizes for dst
                    Rpp64u resampleBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)dstDescPtr->n;

                    // Reinitialize host buffers for output
                    outputf32 = (Rpp32f *)realloc(outputf32, sizeof(Rpp32f) * resampleBufferSize);
                    if(!outputf32)
                    {
                        std::cout << "Unable to reallocate memory for output" << std::endl;
                        break;
                    }

                    startWallTime = omp_get_wtime();
                    rppt_resample_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, inRateTensor, outRateTensor, srcDimsTensor, *window, handle);

                    CHECK(hipHostFree(window->lookup));
                    CHECK(hipHostFree(window));
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

        // QA mode - verify outputs with golden outputs. Below code doesn’t run for performance tests
        if (testType == 0)
        {
            CHECK(hipMemcpy(outputf32, d_outputf32, oBufferSize * sizeof(Rpp32f), hipMemcpyDeviceToHost));

            /* Run only if testCase is not 0
            For testCase 0 verify_non_silent_region_detection function is used for QA testing */
            if (testCase != 0)
                verify_output(outputf32, dstDescPtr, dstDims, testCaseName, dst, scriptPath);

            /* Dump the outputs to csv files for debugging
            Runs only if
            1. DEBUG_MODE is enabled
            2. Current iteration is 1st iteration
            3. Test case is not 0 */
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
    rppDestroyGPU(handle);

    // performance test mode
    if (testType == 1)
    {
        // display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns * noOfIterations);
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }

    cout << endl;

    // free memory
    free(srcDims);
    free(dstDims);
    free(inputf32);
    free(outputf32);
    CHECK(hipFree(d_inputf32));
    CHECK(hipFree(d_outputf32));
    CHECK(hipHostFree(srcLengthTensor));
    CHECK(hipHostFree(channelsTensor));
    CHECK(hipHostFree(srcDimsTensor));
    if (detectedIndex != nullptr)
        CHECK(hipHostFree(detectedIndex));
    if (detectionLength != nullptr)
        CHECK(hipHostFree(detectionLength));
    return 0;
}