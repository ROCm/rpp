/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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
    const int MIN_ARG_COUNT = 10;
    if(argc < MIN_ARG_COUNT)
    {
        cout << "\nImproper Usage! Needs all arguments!\n";
        cout << "\nUsage: ./Tensor_misc_hip <case number = 0:2> <test type 0/1> <toggle 0/1> <number of dimensions> <batch size> <num runs> <additional param> <dst path> <script path>\n";
        return -1;
    }
    Rpp32u testCase, testType, nDim, batchSize, numRuns, bitDepth, toggle;
    bool qaMode;

    testCase = atoi(argv[1]);
    testType = atoi(argv[2]);
    toggle = atoi(argv[3]);
    nDim = atoi(argv[4]);
    batchSize = atoi(argv[5]);
    numRuns = atoi(argv[6]);
    bitDepth = atoi(argv[7]);
    string dst = argv[9];
    string scriptPath = argv[10];
    qaMode = (testType == 0);
    bool axisMaskCase = (testCase == NORMALIZE || testCase == CONCAT);
    bool permOrderCase = (testCase == TRANSPOSE);
    bool broadCastCase = (testCase == TENSOR_ADD_TENSOR || testCase == TENSOR_SUBTRACT_TENSOR || testCase == TENSOR_MULTIPLY_TENSOR || testCase == TENSOR_DIVIDE_TENSOR);
    int additionalParam = (axisMaskCase || permOrderCase || broadCastCase) ? atoi(argv[8]) : 1;
    int axisMask = additionalParam, permOrder = additionalParam, broadCastFlag = additionalParam;

    if(qaMode && batchSize != 3)
    {
        cout << "QA mode can only run with batchsize 3" <<std::endl;
        return -1;
    }

    string funcName = augmentationMiscMap[testCase];
    if(funcName.empty())
    {
        cout << "\ncase " << testCase << " is not supported\n";
        return -1;
    }

    std::string bitdepthStr; // Variable to store the bit depth as a string
    switch(bitDepth)
    {
        case 0: bitdepthStr = "u8"; break;
        case 1: bitdepthStr = "f16"; break;
        case 2: bitdepthStr = "f32"; break;
        case 3: bitdepthStr = "u8_f32"; break;
        case 4: bitdepthStr = "u8_f32"; break;
        case 5: bitdepthStr = "i8"; break;
        case 6: bitdepthStr = "u8_i8"; break;
        case 7: bitdepthStr = "i16"; break;
        case 8: bitdepthStr = "u16"; break;
        case 9: bitdepthStr = "i32"; break;
        case 10: bitdepthStr = "u32"; break;
        case 11: bitdepthStr = "i16_f32"; break;
        default: bitdepthStr = "unknown"; break;
    }

    std::string func = funcName + "_" + std::to_string(nDim) + "d_" + bitdepthStr;
    if(axisMaskCase)
        func += "_axisMask" + std::to_string(axisMask);
    if(permOrderCase)
        func += "_permOrder" + std::to_string(permOrder);
    if(broadCastFlag == 1)
        func += "_broadcast_input2";
    else if(broadCastFlag == 2)
        func += "_broadcast_input1";

    // fill roi based on mode and number of dimensions
    Rpp32u *roiTensor, *dstRoiTensor, *roiTensorSecond = nullptr;
    CHECK_RETURN_STATUS(hipHostMalloc(&roiTensor, nDim * 2 * batchSize, sizeof(Rpp32u)));
    CHECK_RETURN_STATUS(hipHostMalloc(&dstRoiTensor, nDim * 2 * batchSize * sizeof(Rpp32u)));
    fill_roi_values(nDim, batchSize, roiTensor, qaMode);
    memcpy(dstRoiTensor, roiTensor, nDim * 2 * batchSize * sizeof(Rpp32u));
    if(testCase == CONCAT)
    {
        CHECK_RETURN_STATUS(hipHostMalloc(&roiTensorSecond, nDim * 2 * batchSize * sizeof(Rpp32u)));
        fill_roi_values(nDim, batchSize, roiTensorSecond, qaMode);
        dstRoiTensor[nDim + axisMask] = roiTensor[nDim + axisMask] + roiTensorSecond[nDim + axisMask];
    }
    if(broadCastCase)
    {
        CHECK_RETURN_STATUS(hipHostMalloc(&roiTensorSecond, nDim * 2 * batchSize * sizeof(Rpp32u)));
        fill_roi_values(nDim, batchSize, roiTensorSecond, qaMode, broadCastFlag);
    }

    // set src/dst generic tensor descriptors
    RpptGenericDescPtr srcDescriptorPtrND, srcDescriptorPtrNDSecond, dstDescriptorPtrND;
    CHECK_RETURN_STATUS(hipHostMalloc(&srcDescriptorPtrND, sizeof(RpptGenericDesc)));
    CHECK_RETURN_STATUS(hipHostMalloc(&dstDescriptorPtrND, sizeof(RpptGenericDesc)));

    // set dims and compute strides
    int offSetInBytes = 0;
    if(testCase == LOG1P && bitDepth == 11)
    {
        set_generic_descriptor(srcDescriptorPtrND, nDim, offSetInBytes, 7, batchSize, roiTensor);
        set_generic_descriptor(dstDescriptorPtrND, nDim, offSetInBytes, 2, batchSize, dstRoiTensor);
    }
    else if(testCase == LOG && bitDepth == 4)
    {
        set_generic_descriptor(srcDescriptorPtrND, nDim, offSetInBytes, 0, batchSize, roiTensor);
        set_generic_descriptor(dstDescriptorPtrND, nDim, offSetInBytes, 2, batchSize, dstRoiTensor);
    }
    else if(testCase == TENSOR_DIVIDE_TENSOR && bitDepth == 4)
    {
        set_generic_descriptor(srcDescriptorPtrND, nDim, offSetInBytes, 4, batchSize, roiTensor);
        set_generic_descriptor(dstDescriptorPtrND, nDim, offSetInBytes, 2, batchSize, dstRoiTensor);
    }
    else if(testCase == TENSOR_DIVIDE_TENSOR)
    {
        set_generic_descriptor(srcDescriptorPtrND, nDim, offSetInBytes, bitDepth, batchSize, roiTensor);
        set_generic_descriptor(dstDescriptorPtrND, nDim, offSetInBytes, 2, batchSize, dstRoiTensor);
    }
    else
    {
        set_generic_descriptor(srcDescriptorPtrND, nDim, offSetInBytes, bitDepth, batchSize, roiTensor);
        set_generic_descriptor(dstDescriptorPtrND, nDim, offSetInBytes, bitDepth, batchSize, dstRoiTensor);
    }
    set_generic_descriptor_layout(srcDescriptorPtrND, dstDescriptorPtrND, nDim, toggle, qaMode);

    srcDescriptorPtrNDSecond = nullptr;
    if(testCase == CONCAT || broadCastCase)
    {
        CHECK_RETURN_STATUS(hipHostMalloc(&srcDescriptorPtrNDSecond, sizeof(RpptGenericDesc)));
        set_generic_descriptor(srcDescriptorPtrNDSecond, nDim, offSetInBytes, bitDepth, batchSize, roiTensorSecond);
        set_generic_descriptor_layout(srcDescriptorPtrNDSecond, dstDescriptorPtrND, nDim, toggle, qaMode);
    }

    Rpp32u iBufferSize = 1;
    Rpp32u oBufferSize = 1;
    Rpp32u iBufferSizeSecond = 1;
    Rpp64u iBufferSizeInBytes = 1;
    Rpp64u oBufferSizeInBytes = 1;
    Rpp64u iBufferSizeSecondInBytes = 1;
    for(int i = 0; i <= nDim; i++)
    {
        iBufferSize *= srcDescriptorPtrND->dims[i];
        oBufferSize *= dstDescriptorPtrND->dims[i];
        if(testCase == CONCAT || broadCastCase)
            iBufferSizeSecond *= srcDescriptorPtrNDSecond->dims[i];
    }

    if(testCase == LOG1P && bitDepth == 11)
    {
        // LOG1P expects int16 input (we transform F32->I16 in inputI16), but the 'input' buffer used
        // here is F32 (we store F32 to then convert). So allocate as F32 to hold that data.
        iBufferSizeInBytes = iBufferSize * get_size_of_data_type(RpptDataType::F32);
        oBufferSizeInBytes = oBufferSize * get_size_of_data_type(RpptDataType::F32);
    }
    else
    {
        iBufferSizeInBytes = iBufferSize * get_size_of_data_type(srcDescriptorPtrND->dataType);
        oBufferSizeInBytes = oBufferSize * get_size_of_data_type(dstDescriptorPtrND->dataType);
    }

    // Allocate memory for input/output
    void *input = nullptr, *inputSecond = nullptr, *output = nullptr, *inputI16 = nullptr;
    void *d_input = nullptr, *d_inputSecond = nullptr, *d_output = nullptr, *d_inputI16 = nullptr;

    input = calloc(iBufferSizeInBytes, 1);
    output = calloc(oBufferSizeInBytes, 1);
    CHECK_RETURN_STATUS(hipMalloc(&d_input, iBufferSizeInBytes));
    CHECK_RETURN_STATUS(hipMalloc(&d_output, oBufferSizeInBytes));
    if(testCase == CONCAT || broadCastCase)
    {
        iBufferSizeSecondInBytes = iBufferSizeSecond * get_size_of_data_type(srcDescriptorPtrNDSecond->dataType);
        inputSecond = calloc(iBufferSizeSecond, get_size_of_data_type(srcDescriptorPtrNDSecond->dataType));
        CHECK_RETURN_STATUS(hipMalloc(&d_inputSecond, iBufferSizeSecondInBytes));
    }
    // read input data
    if(qaMode)
    {
        if(broadCastCase)
            read_data(input, nDim, 0, scriptPath, funcName, bitDepth, broadCastFlag);
        else if(bitDepth == 11) // log1p
            read_data(input, nDim, 0, scriptPath, funcName, 2);
        else if(bitDepth == 4) // log
            read_data(input, nDim, 0, scriptPath, funcName, 0);
        else
            read_data(input, nDim, 0, scriptPath, funcName, bitDepth);
        if(testCase == CONCAT)
            read_data(inputSecond, nDim, 0, scriptPath, funcName, bitDepth);
        if(broadCastCase)
        {
            if(bitDepth == 2) {
                Rpp32f *inputSecondTemp = static_cast<Rpp32f *>(inputSecond);
                Rpp32f *inputU8 = static_cast<Rpp32f *>(input);
                for (int i = 0; i < iBufferSizeSecond; i++)
                    inputSecondTemp[i] = inputU8[(i+1) % iBufferSize];
            }
            else if((bitDepth == 0) || (bitDepth == 4)) {
                Rpp8u *inputSecondTemp = static_cast<Rpp8u *>(inputSecond);
                Rpp8u *inputU8 = static_cast<Rpp8u *>(input);
                for (int i = 0; i < iBufferSizeSecond; i++)
                    inputSecondTemp[i] = inputU8[(i+1) % iBufferSize];
            }
        }
    }
    else
    {
        // Generic random data filling based on bitDepth
        Rpp32f *inputF32 = NULL, *inputF32Second = NULL, *outputF32 = NULL;
        Rpp16s *inputI16 = NULL;
        inputF32 = static_cast<Rpp32f *>(calloc(iBufferSize, sizeof(Rpp32f)));
        outputF32 = static_cast<Rpp32f *>(calloc(oBufferSize, sizeof(Rpp32f)));
        if((testCase == CONCAT) || (broadCastCase))
            inputF32Second = static_cast<Rpp32f *>(calloc(iBufferSizeSecond, sizeof(Rpp32f)));

        // Generate sample values in range based on number of bits for representation
        // Note : I32/U32 can represent higher range of values - Limit set just for testing purposes
        Rpp32u valLimit = 255;
        if((bitDepth == 7) || (bitDepth == 8))
            valLimit = 65535;
        if((bitDepth == 9) || (bitDepth == 10))
            valLimit = 262143;

        std::srand(0);
        for(int i = 0; i < iBufferSize; i++)
            inputF32[i] = static_cast<float>((std::rand() % valLimit));
        if((testCase == CONCAT) || (broadCastCase))
        {
            for(int i = 0; i < iBufferSizeSecond; i++)
                inputF32Second[i] = static_cast<float>((std::rand() % valLimit));
        }

        convert_input_bitdepth(inputF32, inputF32Second, input, inputSecond, bitDepth, iBufferSize, iBufferSizeSecond, iBufferSizeInBytes, iBufferSizeSecondInBytes, srcDescriptorPtrND, srcDescriptorPtrNDSecond, testCase);
    }

    if(testCase == LOG1P)
    {
        Rpp64u iBufferSizeInBytesI16 = iBufferSize * sizeof(Rpp16s);
        inputI16 = calloc(iBufferSize, sizeof(Rpp16s));
        CHECK_RETURN_STATUS(hipMalloc(&d_inputI16, iBufferSizeInBytesI16));

        Rpp32f *inputF32 = static_cast<Rpp32f *>(input);
        Rpp16s *inputI16_cast = static_cast<Rpp16s *>(inputI16);
        for (int i = 0; i < iBufferSize; i++)
            inputI16_cast[i] = static_cast<Rpp16s>(inputF32[i]);
    }

    // Copy data from Host to Device
    CHECK_RETURN_STATUS(hipMemcpy(d_input, input, iBufferSizeInBytes, hipMemcpyHostToDevice));
    if(testCase == CONCAT || broadCastCase)
    {
        CHECK_RETURN_STATUS(hipMemcpy(d_inputSecond, inputSecond, iBufferSizeSecondInBytes, hipMemcpyHostToDevice));
    }
    if(testCase == LOG1P)
    {
        Rpp64u iBufferSizeInBytesI16 = iBufferSize * sizeof(Rpp16s);
        CHECK_RETURN_STATUS(hipMemcpy(d_inputI16, inputI16, iBufferSizeInBytesI16, hipMemcpyHostToDevice));
    }

    Rpp32u *permTensor = nullptr;
    if(testCase == TRANSPOSE)
        CHECK_RETURN_STATUS(hipHostMalloc(&permTensor, nDim * sizeof(Rpp32u)));

    rppHandle_t handle;
    hipStream_t stream;
    CHECK_RETURN_STATUS(hipStreamCreate(&stream));
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
    rppCreate(&handle, batchSize, 0, stream, backend);

    Rpp32f *meanTensor = nullptr, *stdDevTensor = nullptr;
    Rpp32f *meanTensorCPU = nullptr, *stdDevTensorCPU = nullptr;
    bool externalMeanStd = true;

    Rpp32u missingFuncFlag = 0;
    double startWallTime, endWallTime;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0, wallTime = 0;
    string testCaseName;

    // case-wise RPP API and measure time script for Unit and Performance test
    cout << "\nRunning " << func << " " << numRuns << " times (each time with a batch size of " << batchSize << ") and computing mean statistics...";
    for(int perfCount = 0; perfCount < numRuns; perfCount++)
    {
        switch(testCase)
        {
            case TRANSPOSE:
            {
                testCaseName  = "transpose";
                fill_perm_values(nDim, permTensor, qaMode, permOrder);

                for(int i = 1; i <= nDim; i++)
                    dstDescriptorPtrND->dims[i] = roiTensor[nDim + permTensor[i - 1]];
                compute_strides(dstDescriptorPtrND);

                startWallTime = omp_get_wtime();
                if(bitDepth == 0 || bitDepth == 1 || bitDepth == 2 || bitDepth == 5)
                    rppt_transpose_gpu(d_input, srcDescriptorPtrND, d_output, dstDescriptorPtrND, permTensor, roiTensor, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case NORMALIZE:
            {
                testCaseName  = "normalize";
                float scale = 1.0;
                float shift = 0.0;

                // computeMeanStddev set to 3 means both mean and stddev should be computed internally.
                // Wherein 0th bit used to represent computeMean and 1st bit for computeStddev.
                Rpp8u computeMeanStddev = 3;
                externalMeanStd = !computeMeanStddev; // when mean and stddev is passed from user

                Rpp32u size = 1; // length of mean and stddev tensors differ based on axisMask and nDim
                Rpp32u maxSize = 1;
                for(int batch = 0; batch < batchSize; batch++)
                {
                    size = 1;
                    for(int i = 0; i < nDim; i++)
                        size *= ((axisMask & (int)(pow(2,i))) >= 1) ? 1 : roiTensor[(nDim * 2 * batch) + nDim + i];
                    maxSize = max(maxSize, size);
                }

                // allocate memory if no memory is allocated
                if(meanTensor == nullptr)
                    CHECK_RETURN_STATUS(hipMalloc(&meanTensor, maxSize * batchSize * sizeof(Rpp32f)));

                if(stdDevTensor == nullptr)
                    CHECK_RETURN_STATUS(hipMalloc(&stdDevTensor, maxSize * batchSize * sizeof(Rpp32f)));

                if(!computeMeanStddev)
                {
                    if(meanTensorCPU == nullptr)
                        meanTensorCPU = static_cast<Rpp32f *>(malloc(maxSize * sizeof(Rpp32f)));
                    if(stdDevTensorCPU == nullptr)
                        stdDevTensorCPU = static_cast<Rpp32f *>(malloc(maxSize * sizeof(Rpp32f)));
                    fill_mean_stddev_values(nDim, maxSize, meanTensorCPU, stdDevTensorCPU, qaMode, axisMask, scriptPath, bitDepth);
                    CHECK_RETURN_STATUS(hipMemcpy(meanTensor, meanTensorCPU, maxSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
                    CHECK_RETURN_STATUS(hipMemcpy(stdDevTensor, stdDevTensorCPU, maxSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
                    CHECK_RETURN_STATUS(hipDeviceSynchronize());
                }

                startWallTime = omp_get_wtime();
                if(bitDepth == 0 || bitDepth == 1 || bitDepth == 2 || bitDepth == 5)
                    rppt_normalize_gpu(d_input, srcDescriptorPtrND, d_output, dstDescriptorPtrND, axisMask, meanTensor, stdDevTensor, computeMeanStddev, scale, shift, roiTensor, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case LOG:
            {
                testCaseName  = "log";

                startWallTime = omp_get_wtime();
                if(bitDepth == 2 || bitDepth == 4)
                    rppt_log_gpu(d_input, srcDescriptorPtrND, d_output, dstDescriptorPtrND, roiTensor, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case CONCAT:
            {
                testCaseName  = "concat";
                startWallTime = omp_get_wtime();
                if(bitDepth == 0 || bitDepth == 1 || bitDepth == 2 || bitDepth == 5)
                    rppt_concat_gpu(d_input, d_inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond, d_output, dstDescriptorPtrND, axisMask, roiTensor, roiTensorSecond, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case LOG1P:
            {
                testCaseName  = "log1p";

                startWallTime = omp_get_wtime();
                if(bitDepth == 11)
                    rppt_log1p_gpu(d_inputI16, srcDescriptorPtrND, d_output, dstDescriptorPtrND, roiTensor, handle);
                else
                    missingFuncFlag = 1;
                    
                break;
            }
            case TENSOR_ADD_TENSOR:
            {
                testCaseName  = "tensor_add_tensor";

                startWallTime = omp_get_wtime();
                if(bitDepth == 0 || bitDepth == 1 || bitDepth == 2 || bitDepth == 5 || bitDepth == 7 || bitDepth == 8 || bitDepth == 9 || bitDepth == 10)
                {
                    if(broadCastFlag == 0)
                        rppt_tensor_add_tensor_gpu(d_input, d_inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond, d_output, dstDescriptorPtrND, RPP_BROADCAST_DISABLE, roiTensor, roiTensorSecond, handle);
                    else if(broadCastFlag == 2)
                        rppt_tensor_add_tensor_gpu(d_input, d_inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond, d_output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensor, roiTensorSecond, handle);
                    else
                        rppt_tensor_add_tensor_gpu(d_inputSecond, d_input, srcDescriptorPtrNDSecond, srcDescriptorPtrND, d_output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensorSecond, roiTensor, handle);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case TENSOR_SUBTRACT_TENSOR:
            {
                testCaseName  = "tensor_subtract_tensor";

                startWallTime = omp_get_wtime();
                if(bitDepth == 0 || bitDepth == 1 || bitDepth == 2 || bitDepth == 5 || bitDepth == 7 || bitDepth == 8 || bitDepth == 9 || bitDepth == 10)
                {
                    if(broadCastFlag == 0)
                        rppt_tensor_subtract_tensor_gpu(d_input, d_inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond, d_output, dstDescriptorPtrND, RPP_BROADCAST_DISABLE, roiTensor, roiTensorSecond, handle);
                    else if(broadCastFlag == 2)
                        rppt_tensor_subtract_tensor_gpu(d_input, d_inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond, d_output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensor, roiTensorSecond, handle);
                    else
                        rppt_tensor_subtract_tensor_gpu(d_inputSecond, d_input, srcDescriptorPtrNDSecond, srcDescriptorPtrND, d_output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensorSecond, roiTensor, handle);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case TENSOR_MULTIPLY_TENSOR:
            {
                testCaseName  = "tensor_multiply_tensor";

                startWallTime = omp_get_wtime();
                if(bitDepth == 0 || bitDepth == 1 || bitDepth == 2 || bitDepth == 5 || bitDepth == 7 || bitDepth == 8 || bitDepth == 9 || bitDepth == 10)
                {
                    if(broadCastFlag == 0)
                        rppt_tensor_multiply_tensor_gpu(d_input, d_inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond, d_output, dstDescriptorPtrND, RPP_BROADCAST_DISABLE, roiTensor, roiTensorSecond, handle);
                    else if(broadCastFlag == 2)
                        rppt_tensor_multiply_tensor_gpu(d_input, d_inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond, d_output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensor, roiTensorSecond, handle);
                    else
                        rppt_tensor_multiply_tensor_gpu(d_inputSecond, d_input, srcDescriptorPtrNDSecond, srcDescriptorPtrND, d_output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensorSecond, roiTensor, handle);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case TENSOR_DIVIDE_TENSOR:
            {
                testCaseName  = "tensor_divide_tensor";

                startWallTime = omp_get_wtime();
                if(bitDepth == 0 || bitDepth == 1 || bitDepth == 2 || bitDepth == 4 || bitDepth == 5 || bitDepth == 7 || bitDepth == 8 || bitDepth == 9 || bitDepth == 10)
                {
                    if(broadCastFlag == 0)
                        rppt_tensor_divide_tensor_gpu(d_input, d_inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond, d_output, dstDescriptorPtrND, RPP_BROADCAST_DISABLE, roiTensor, roiTensorSecond, handle);
                    else if(broadCastFlag == 2)
                        rppt_tensor_divide_tensor_gpu(d_input, d_inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond, d_output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensor, roiTensorSecond, handle);
                    else
                        rppt_tensor_divide_tensor_gpu(d_inputSecond, d_input, srcDescriptorPtrNDSecond, srcDescriptorPtrND, d_output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensorSecond, roiTensor, handle);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            default:
            {
                missingFuncFlag = 1;
                break;
            }
        }
        CHECK_RETURN_STATUS(hipDeviceSynchronize());
        endWallTime = omp_get_wtime();

        if (missingFuncFlag == 1)
        {
            cout << "\nThe functionality " << func << " doesn't yet exist in RPP\n";
            return RPP_ERROR_NOT_IMPLEMENTED;
        }

        wallTime = endWallTime - startWallTime;
        maxWallTime = std::max(maxWallTime, wallTime);
        minWallTime = std::min(minWallTime, wallTime);
        avgWallTime += wallTime;
    }

    // compare outputs if qaMode is true
    if(qaMode)
    {
        CHECK_RETURN_STATUS(hipMemcpy(output, d_output, oBufferSizeInBytes, hipMemcpyDeviceToHost));
        compare_output(output, nDim, batchSize, bitDepth, oBufferSize, dst, func, testCaseName, additionalParam, scriptPath, broadCastCase ? broadCastFlag : 0, externalMeanStd);
    }
    else
    {
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= numRuns;
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }
    rppDestroy(handle,backend);

    // Free device memory
    CHECK_RETURN_STATUS(hipFree(d_input));
    CHECK_RETURN_STATUS(hipFree(d_output));
    if(d_inputSecond != nullptr)
        CHECK_RETURN_STATUS(hipFree(d_inputSecond));
    if(d_inputI16 != nullptr)
        CHECK_RETURN_STATUS(hipFree(d_inputI16));
    if(meanTensor != nullptr)
        CHECK_RETURN_STATUS(hipFree(meanTensor));
    if(stdDevTensor != nullptr)
        CHECK_RETURN_STATUS(hipFree(stdDevTensor));

    free(input);
    free(output);
    if(inputSecond != nullptr)
        free(inputSecond);
    if(inputI16 != nullptr)
        free(inputI16);
    CHECK_RETURN_STATUS(hipHostFree(roiTensor));
    CHECK_RETURN_STATUS(hipHostFree(dstRoiTensor));
    if(roiTensorSecond != nullptr)
        CHECK_RETURN_STATUS(hipHostFree(roiTensorSecond));
    CHECK_RETURN_STATUS(hipHostFree(srcDescriptorPtrND));
    CHECK_RETURN_STATUS(hipHostFree(dstDescriptorPtrND));
    if(srcDescriptorPtrNDSecond != nullptr)
        CHECK_RETURN_STATUS(hipHostFree(srcDescriptorPtrNDSecond));
    if (permTensor != nullptr)
        CHECK_RETURN_STATUS(hipHostFree(permTensor));
    if(meanTensorCPU != nullptr)
        free(meanTensorCPU);
    if(stdDevTensorCPU != nullptr)
        free(stdDevTensorCPU);

    return 0;
}
