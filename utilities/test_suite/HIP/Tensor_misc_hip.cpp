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

#include "../rpp_test_suite_misc.h"

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 8;
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_misc_hip <case number = 0:0> <test type 0/1> <toggle 0/1> <number of dimensions> <batch size> <num runs> <additional param> <dst path> <script path>\n");
        return -1;
    }
    Rpp32u testCase, testType, nDim, batchSize, numRuns, toggle;
    bool qaMode;

    testCase = atoi(argv[1]);
    testType = atoi(argv[2]);
    toggle = atoi(argv[3]);
    nDim = atoi(argv[4]);
    batchSize = atoi(argv[5]);
    numRuns = atoi(argv[6]);
    string dst = argv[7];
    string scriptPath = argv[8];
    qaMode = (testType == 0);
    int additionalParam = 1;

    if (qaMode && batchSize != 3)
    {
        std::cout<<"QA mode can only run with batchsize 3"<<std::endl;
        return -1;
    }

    string funcName = augmentationMiscMap[testCase];
    if (funcName.empty())
    {
        printf("\ncase %d is not supported\n", testCase);
        return -1;
    }

    // fill roi based on mode and number of dimensions
    Rpp32u *roiTensor;
    CHECK(hipHostMalloc(&roiTensor, nDim * 2 * batchSize, sizeof(Rpp32u)));
    fill_roi_values(nDim, batchSize, roiTensor, qaMode);

    // set src/dst generic tensor descriptors
    RpptGenericDescPtr srcDescriptorPtrND, dstDescriptorPtrND;
    CHECK(hipHostMalloc(&srcDescriptorPtrND, sizeof(RpptGenericDesc)));
    CHECK(hipHostMalloc(&dstDescriptorPtrND, sizeof(RpptGenericDesc)));
    srcDescriptorPtrND->numDims = nDim + 1;
    srcDescriptorPtrND->offsetInBytes = 0;
    srcDescriptorPtrND->dataType = RpptDataType::F32;

    dstDescriptorPtrND->numDims = nDim + 1;
    dstDescriptorPtrND->offsetInBytes = 0;
    dstDescriptorPtrND->dataType = RpptDataType::F32;

    // set dims and compute strides
    srcDescriptorPtrND->dims[0] = batchSize;
    dstDescriptorPtrND->dims[0] = batchSize;
    for(int i = 1; i <= nDim; i++)
        srcDescriptorPtrND->dims[i] = roiTensor[nDim + i - 1];
    compute_strides(srcDescriptorPtrND);

    // if testCase is not normalize, then copy dims and strides from src to dst
    if(testCase != 0)
    {
        memcpy(dstDescriptorPtrND->dims, srcDescriptorPtrND->dims, srcDescriptorPtrND->numDims * sizeof(Rpp32u));
        memcpy(dstDescriptorPtrND->strides, srcDescriptorPtrND->strides, dstDescriptorPtrND->numDims * sizeof(Rpp32u));
    }
    set_generic_descriptor_layout(srcDescriptorPtrND, dstDescriptorPtrND, nDim, toggle, qaMode);

    Rpp32u bufferSize = 1;
    for(int i = 0; i <= nDim; i++)
        bufferSize *= srcDescriptorPtrND->dims[i];

    // allocate memory for input / output
    Rpp32f *inputF32 = NULL, *outputF32 = NULL;
    inputF32 = static_cast<Rpp32f *>(calloc(bufferSize, sizeof(Rpp32f)));
    outputF32 = static_cast<Rpp32f *>(calloc(bufferSize, sizeof(Rpp32f)));

    void *d_inputF32, *d_outputF32;
    CHECK(hipMalloc(&d_inputF32, bufferSize * sizeof(Rpp32f)));
    CHECK(hipMalloc(&d_outputF32, bufferSize * sizeof(Rpp32f)));

    // read input data
    if(qaMode)
        read_data(inputF32, nDim, 0, scriptPath, funcName);
    else
    {
        std::srand(0);
        for(int i = 0; i < bufferSize; i++)
            inputF32[i] = static_cast<float>((std::rand() % 255));
    }

    // copy data from HOST to HIP
    CHECK(hipMemcpy(d_inputF32, inputF32, bufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
    CHECK(hipDeviceSynchronize());

    rppHandle_t handle;
    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));
    rppCreateWithStreamAndBatchSize(&handle, stream, batchSize);
    bool externalMeanStd = true;

    double startWallTime, endWallTime;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0, wallTime = 0;
    string testCaseName;

    // case-wise RPP API and measure time script for Unit and Performance test
    printf("\nRunning log %d times (each time with a batch size of %d) and computing mean statistics...", numRuns, batchSize);
    for(int perfCount = 0; perfCount < numRuns; perfCount++)
    {
        switch(testCase)
        {
            case 2:
            {
                testCaseName  = "log";

                startWallTime = omp_get_wtime();
                rppt_log_gpu(d_inputF32, srcDescriptorPtrND, d_outputF32, dstDescriptorPtrND, roiTensor, handle);

                break;
            }
            default:
            {
                cout << "functionality is not supported" <<std::endl;
                exit(0);
            }
        }
        CHECK_RETURN_STATUS(hipDeviceSynchronize());
        endWallTime = omp_get_wtime();

        wallTime = endWallTime - startWallTime;
        maxWallTime = std::max(maxWallTime, wallTime);
        minWallTime = std::min(minWallTime, wallTime);
        avgWallTime += wallTime;
    }
    rppDestroyGPU(handle);

    // compare outputs if qaMode is true
    if(qaMode)
    {
        CHECK_RETURN_STATUS(hipDeviceSynchronize());
        CHECK_RETURN_STATUS(hipMemcpy(outputF32, d_outputF32, bufferSize * sizeof(Rpp32f), hipMemcpyDeviceToHost));
        CHECK_RETURN_STATUS(hipDeviceSynchronize());
        compare_output(outputF32, nDim, batchSize, bufferSize, dst, funcName, testCaseName, additionalParam, scriptPath, externalMeanStd);
    }
    if(!qaMode)
    {
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= numRuns;
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }


    free(inputF32);
    free(outputF32);
    CHECK(hipHostFree(roiTensor));
    CHECK(hipHostFree(srcDescriptorPtrND));
    CHECK(hipHostFree(dstDescriptorPtrND));
    CHECK(hipFree(d_inputF32));
    CHECK(hipFree(d_outputF32));

    return 0;
}