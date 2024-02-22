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
    const int MIN_ARG_COUNT = 7;
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_misc_hip <case number = 0:0> <test type 0/1> <toggle 0/1> <number of dimensions> <batch size> <num runs> <dst path>\n");
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
    qaMode = (testType == 0);

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
    RpptGenericDesc srcDescriptor, dstDescriptor;
    RpptGenericDescPtr srcDescriptorPtrND, dstDescriptorPtrND;
    srcDescriptorPtrND  = &srcDescriptor;
    srcDescriptorPtrND->numDims = nDim + 1;
    srcDescriptorPtrND->offsetInBytes = 0;
    srcDescriptorPtrND->dataType = RpptDataType::F32;

    dstDescriptorPtrND  = &dstDescriptor;
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
    inputF32 = (Rpp32f *)calloc(bufferSize, sizeof(Rpp32f));
    outputF32 = (Rpp32f *)calloc(bufferSize, sizeof(Rpp32f));

    void *d_inputF32, *d_outputF32;
    CHECK(hipMalloc(&d_inputF32, bufferSize * sizeof(Rpp32f)));
    CHECK(hipMalloc(&d_outputF32, bufferSize * sizeof(Rpp32f)));

    rppHandle_t handle;
    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));
    rppCreateWithStreamAndBatchSize(&handle, stream, batchSize);

    Rpp32f *meanTensor = nullptr, *stdDevTensor = nullptr;
    double startWallTime, endWallTime;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0, wallTime = 0;

    // case-wise RPP API and measure time script for Unit and Performance test
    printf("\nRunning normalize %d times (each time with a batch size of %d) and computing mean statistics...", numRuns, batchSize);
    for(int perfCount = 0; perfCount < numRuns; perfCount++)
    {
        switch(testCase)
        {
            case 1:
            {
                // Modify ROI to 4x5x7 when checking QA for axisMask = 6 alone(calls direct c code internally)
                int axisMask = 3; // 3D HWC Channel normalize axes(0,1)
                float scale = 1.0;
                float shift = 0.0;
                bool computeMean, computeStddev;
                computeMean = computeStddev = 0;

                // read input data
                if(qaMode)
                    read_data(inputF32, nDim, 0,  bufferSize, batchSize, axisMask, "HIP");
                else
                {
                    std::srand(0);
                    for(int i = 0; i < bufferSize; i++)
                        inputF32[i] = (float)(std::rand() % 255);
                }

                // copy data from HOST to HIP
                CHECK(hipMemcpy(d_inputF32, (void *)inputF32, bufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
                CHECK(hipDeviceSynchronize());

                if (qaMode && nDim == 3 && axisMask == 3 && (computeMean || computeStddev))
                {
                    std::cout<<"QA mode can only run with mean and stddev input from user when nDim is 3"<<std::endl;
                    return -1;
                }
                else if(qaMode && nDim == 3 && axisMask != 3 && (!computeMean || !computeStddev))
                {
                    std::cout<<"QA mode can only run with internal mean and stddev when nDim is 3"<<std::endl;
                    return -1;
                }

                if (qaMode && nDim == 4 && (!computeMean && !computeStddev))
                {
                    std::cout<<"QA mode can only run with internal mean and stddev when nDim is 4"<<std::endl;
                    return -1;
                }

                Rpp32u size = 1; // length of input tensors differ based on axisMask and nDim
                Rpp32u maxSize = 1;
                for(int batch = 0; batch < batchSize; batch++)
                {
                    size = 1;
                    for(int i = 0; i < nDim; i++)
                        size *= ((axisMask & (int)(pow(2,i))) >= 1) ? 1 : roiTensor[(nDim * 2 * batch) + nDim + i];
                    maxSize = max(maxSize, size);
                }

                // allocate memory if not memory is allocated
                if(meanTensor == nullptr)
                    CHECK(hipMalloc(&meanTensor, maxSize * batchSize * sizeof(Rpp32f)));

                if(stdDevTensor == nullptr)
                    CHECK(hipMalloc(&stdDevTensor, maxSize * batchSize * sizeof(Rpp32f)));

                if(!(computeMean && computeStddev))
                {
                    Rpp32f *meanTensorCPU = (Rpp32f *)malloc(maxSize * batchSize * sizeof(Rpp32f));
                    Rpp32f *stdDevTensorCPU = (Rpp32f *)malloc(maxSize * batchSize * sizeof(Rpp32f));
                    fill_mean_stddev_values(nDim, batchSize, size, meanTensorCPU, stdDevTensorCPU, qaMode);
                    CHECK(hipMemcpy(meanTensor, meanTensorCPU, maxSize * batchSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
                    CHECK(hipMemcpy(stdDevTensor, stdDevTensorCPU, maxSize * batchSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
                    CHECK(hipDeviceSynchronize());
                    free(meanTensorCPU);
                    free(stdDevTensorCPU);
                }

                startWallTime = omp_get_wtime();
                rppt_normalize_gpu(d_inputF32, srcDescriptorPtrND, d_outputF32, dstDescriptorPtrND, axisMask, meanTensor, stdDevTensor, computeMean, computeStddev, scale, shift, roiTensor, handle);

                // compare outputs if qaMode is true
                if(qaMode)
                {
                    CHECK(hipDeviceSynchronize());
                    CHECK(hipMemcpy(outputF32, d_outputF32, bufferSize * sizeof(Rpp32f), hipMemcpyDeviceToHost));
                    CHECK(hipDeviceSynchronize());
                    compare_output(outputF32, nDim, batchSize, dst, funcName, axisMask, "HIP");
                }
                break;
            }
            default:
            {
                cout << "functionality is not supported" <<std::endl;
                exit(0);
            }
        }
        if(!qaMode)
        {
            CHECK(hipDeviceSynchronize());
            endWallTime = omp_get_wtime();

            wallTime = endWallTime - startWallTime;
            maxWallTime = std::max(maxWallTime, wallTime);
            minWallTime = std::min(minWallTime, wallTime);
            avgWallTime += wallTime;
        }
    }
    rppDestroyGPU(handle);

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
    CHECK(hipFree(d_inputF32));
    CHECK(hipFree(d_outputF32));
    if(meanTensor != nullptr)
        CHECK(hipFree(meanTensor));
    if(stdDevTensor != nullptr)
        CHECK(hipFree(stdDevTensor));

    return 0;
}