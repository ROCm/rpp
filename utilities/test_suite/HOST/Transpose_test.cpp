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
#include <stdlib.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <unistd.h>
#include <dirent.h>
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <ctime> 
#include "rpp.h"

using namespace std;
namespace fs = boost::filesystem;

void compute_strides(RpptGenericDescPtr descriptorPtr) 
{
    if (descriptorPtr->numDims > 0) 
    {
        uint64_t v = 1;
        for (int i = descriptorPtr->numDims - 1; i > 0; i--) 
        {
            descriptorPtr->strides[i] = v;
            v *= descriptorPtr->dims[i + 1];
        }
        descriptorPtr->strides[0] = v;
    }
}

int main()
{
    Rpp32u nDim, batchSize;
    cin>>nDim>>batchSize;
    
    // Set the number of threads to be used by OpenMP pragma for RPP batch processing on host.
    // If numThreads value passed is 0, number of OpenMP threads used by RPP will be set to batch size
    Rpp32u numThreads = 0;
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, batchSize, numThreads);
    
    // set src/dst generic tensor descriptors
    RpptGenericDesc descriptor;
    RpptGenericDescPtr descriptorPtrND = &descriptor;
    descriptorPtrND->numDims = nDim;
    descriptorPtrND->offsetInBytes = 0;
    descriptorPtrND->dataType = RpptDataType::F32;
    descriptorPtrND->layout = RpptLayout::NDHWC;
    
    int numValues = 1;
    Rpp32u *permTensor = (Rpp32u *)calloc(nDim, sizeof(Rpp32u));
    for (int i = 0; i < nDim; i++)
        permTensor[i] = nDim - 1 - i;
        
    Rpp32u *roiTensor = (Rpp32u *)calloc(nDim * batchSize, sizeof(Rpp32u));
    for (int i = 0; i < nDim; i++)
        roiTensor[i] = i + 2;
        
    descriptorPtrND->dims[0] = batchSize;
    for(int i = 1; i <= nDim; i++)
        descriptorPtrND->dims[i] = roiTensor[i - 1];
    compute_strides(descriptorPtrND);
    
    for(int i = 0; i <= nDim; i++)
        numValues *= descriptorPtrND->dims[i];
    
    Rpp32f *inputF32 = (Rpp32f *)calloc(numValues, sizeof(Rpp32f));
    Rpp32f *outputF32 = (Rpp32f *)calloc(numValues, sizeof(Rpp32f));
    
    std::srand(0); 
    for(int i = 0; i < numValues; i++)
        inputF32[i] = (float)(std::rand() % 255);
    
    rppt_transpose_generic_host(inputF32, descriptorPtrND, outputF32, descriptorPtrND, permTensor, roiTensor, handle);
    int cnt = 0;
    std::cerr<<"printing input matrix: "<<std::endl;
    for(int i = 0; i < roiTensor[0]; i++)
    {
        for(int j = 0; j < roiTensor[1]; j++)
        {
            for(int k = 0; k < roiTensor[2]; k++)
            {
                std::cerr<<inputF32[cnt]<<" ";
                cnt++;
            }
            std::cerr<<std::endl;
        }
        std::cerr<<std::endl;
    }
    
    cnt  = 0;
    std::cerr<<"printing output matrix: "<<std::endl;
    for(int i = 0; i < roiTensor[permTensor[0]]; i++)
    {
        for(int j = 0; j < roiTensor[permTensor[1]]; j++)
        {
            for(int k = 0; k < roiTensor[permTensor[2]]; k++)
            {
                std::cerr<<outputF32[cnt]<<" ";
                cnt++;   
            }
            std::cerr<<std::endl;
        }
        std::cerr<<std::endl;
    }
    
    free(inputF32);
    free(outputF32);
    free(permTensor);
    free(roiTensor);   
    return 0;
}