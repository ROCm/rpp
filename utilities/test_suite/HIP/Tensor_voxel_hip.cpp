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
#include "rpp.h"
#include "../rpp_test_suite_common.h"
#include "nifti1.h"

using namespace std;
typedef int16_t NIFTI_DATATYPE;

#define MIN_HEADER_SIZE 348

// reads nifti-1 header file
static int read_nifti_header_file(char* const header_file, nifti_1_header *niftiHeader)
{
    nifti_1_header hdr;

    // open and read header
    FILE *fp = fopen(header_file,"r");
    if (fp == NULL)
    {
        fprintf(stderr, "\nError opening header file %s\n", header_file);
        exit(1);
    }
    int ret = fread(&hdr, MIN_HEADER_SIZE, 1, fp);
    if (ret != 1)
    {
        fprintf(stderr, "\nError reading header file %s\n", header_file);
        exit(1);
    }
    fclose(fp);

    // print header information
    fprintf(stderr, "\n%s header information:", header_file);
    fprintf(stderr, "\nNIFTI1 XYZT dimensions: %d %d %d %d", hdr.dim[1], hdr.dim[2], hdr.dim[3], hdr.dim[4]);
    fprintf(stderr, "\nNIFTI1 Datatype code and bits/pixel: %d %d", hdr.datatype, hdr.bitpix);
    fprintf(stderr, "\nNIFTI1 Scaling slope and intercept: %.6f %.6f", hdr.scl_slope, hdr.scl_inter);
    fprintf(stderr, "\nNIFTI1 Byte offset to data in datafile: %ld", (long)(hdr.vox_offset));
    fprintf(stderr, "\n");

    *niftiHeader = hdr;

    return(0);
}

// reads nifti-1 data file
inline void read_nifti_data_file(char* const data_file, nifti_1_header *niftiHeader, NIFTI_DATATYPE *data)
{
    nifti_1_header hdr = *niftiHeader;
    int ret;

    // open the datafile, jump to data offset
    FILE *fp = fopen(data_file, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "\nError opening data file %s\n", data_file);
        exit(1);
    }
    ret = fseek(fp, (long)(hdr.vox_offset), SEEK_SET);
    if (ret != 0)
    {
        fprintf(stderr, "\nError doing fseek() to %ld in data file %s\n", (long)(hdr.vox_offset), data_file);
        exit(1);
    }

    ret = fread(data, sizeof(NIFTI_DATATYPE), hdr.dim[1] * hdr.dim[2] * hdr.dim[3], fp);
    if (ret != hdr.dim[1] * hdr.dim[2] * hdr.dim[3])
    {
        fprintf(stderr, "\nError reading volume 1 from %s (%d)\n", data_file, ret);
        exit(1);
    }
    fclose(fp);
}

inline void write_nifti_file(nifti_1_header *niftiHeader, NIFTI_DATATYPE *niftiData)
{
    nifti_1_header hdr = *niftiHeader;
    nifti1_extender pad = {0,0,0,0};
    FILE *fp;
    int ret, i;

    // write first hdr.vox_offset bytes of header
    string niiOutputString = "nifti_output.nii";
    const char *niiOutputFile = niiOutputString.c_str();
    fp = fopen(niiOutputFile,"w");
    if (fp == NULL)
    {
        fprintf(stderr, "\nError opening header file %s for write\n",niiOutputFile);
        exit(1);
    }
    ret = fwrite(&hdr, hdr.vox_offset, 1, fp);
    if (ret != 1)
    {
        fprintf(stderr, "\nError writing header file %s\n",niiOutputFile);
        exit(1);
    }

    // for nii files, write extender pad and image data
    ret = fwrite(&pad, 4, 1, fp);
    if (ret != 1)
    {
        fprintf(stderr, "\nError writing header file extension pad %s\n",niiOutputFile);
        exit(1);
    }

    ret = fwrite(niftiData, (size_t)(hdr.bitpix/8), hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4], fp);
    if (ret != hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4])
    {
        fprintf(stderr, "\nError writing data to %s\n",niiOutputFile);
        exit(1);
    }

    fclose(fp);
}

inline void write_image_from_nifti_opencv(uchar *niftiDataU8, nifti_1_header *niftiHeader, int zPlane)
{
    nifti_1_header hdr = *niftiHeader;
    int xyFrameSize = hdr.dim[1] * hdr.dim[2];
    uchar *niftiDataU8Temp = &niftiDataU8[xyFrameSize * zPlane];
    cv::Mat matOutputImage = cv::Mat(hdr.dim[2], hdr.dim[1], CV_8UC1, niftiDataU8Temp);
    string fileName = "nifti_single_zPlane_" + std::to_string(zPlane) + ".jpg";
    cv::imwrite(fileName, matOutputImage);
}

// TODO: Fix issue in writing video
// inline void write_video_from_nifti_opencv(uchar *niftiDataU8, nifti_1_header *niftiHeader, int zPlaneMin, int zPlaneMax)
// {
//     nifti_1_header hdr = *niftiHeader;
//     int xyFrameSize = hdr.dim[1] * hdr.dim[2];
//     uchar *niftiDataU8Temp = &niftiDataU8[xyFrameSize * zPlaneMin];

//     //  opencv video writer create
//     cv::Size frameSize(hdr.dim[1], hdr.dim[2]);
//     cv::VideoWriter videoOutput("niftiVideoOutput.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 15, frameSize);

//     for (int zFrame = zPlaneMin; zFrame < zPlaneMax; zFrame++)
//     {
//         cv::Mat matOutputImageU8 = cv::Mat(hdr.dim[2], hdr.dim[1], CV_8UC1, niftiDataU8Temp);
//         videoOutput.write(matOutputImageU8);
//         niftiDataU8Temp += xyFrameSize;
//     }

//     //  opencv video writer release
//     videoOutput.release();
// }

// Convert default NIFTI_DATATYPE unstrided buffer to RpptDataType::F32 strided buffer
template<typename T>
inline void convert_input_niftitype_to_float_generic(T *input, nifti_1_header *niftiHeader, float *inputF32, RpptGenericDescPtr descriptorPtr3D)
{
    nifti_1_header hdr = *niftiHeader;
    T *inputTemp = input;
    float *inputF32Temp = inputF32;

    if (descriptorPtr3D->layout == RpptLayout::NCDHW)
    {
        int increment = descriptorPtr3D->strides[3] - hdr.dim[1];
        for (int n = 0; n < descriptorPtr3D->dims[0]; n++)
        {
            for (int c = 0; c < hdr.dim[4]; c++)
            {
                for (int d = 0; d < hdr.dim[3]; d++)
                {
                    for (int h = 0; h < hdr.dim[2]; h++)
                    {
                        for (int w = 0; w < hdr.dim[1]; w++)
                        {
                            *inputF32Temp++ = (float)*inputTemp++;
                        }
                        inputF32Temp += increment;
                    }
                }
            }
        }
    }
    else if (descriptorPtr3D->layout == RpptLayout::NDHWC)
    {
        // TODO: to be implemented
    }
}

// Convert RpptDataType::F32 strided buffer to default NIFTI_DATATYPE unstrided buffer
template<typename T>
inline void convert_output_float_to_niftitype_generic(float *outputF32, RpptGenericDescPtr descriptorPtr3D, T *output, nifti_1_header *niftiHeader)
{
    nifti_1_header hdr = *niftiHeader;
    T *outputTemp = output;
    float *outputF32Temp = outputF32;

    if (descriptorPtr3D->layout == RpptLayout::NCDHW)
    {
        int increment = descriptorPtr3D->strides[3] - hdr.dim[1];
        for (int n = 0; n < descriptorPtr3D->dims[0]; n++)
        {
            for (int c = 0; c < hdr.dim[4]; c++)
            {
                for (int d = 0; d < hdr.dim[3]; d++)
                {
                    for (int h = 0; h < hdr.dim[2]; h++)
                    {
                        for (int w = 0; w < hdr.dim[1]; w++)
                        {
                            *outputTemp++ = (T)*outputF32Temp++;
                        }
                        outputF32Temp += increment;
                    }
                }
            }
        }
    }
    else if (descriptorPtr3D->layout == RpptLayout::NDHWC)
    {
        // TODO: to be implemented
    }
}

int main(int argc, char * argv[])
{
    int layoutType, testCase;
    char *header_file, *data_file;

    if (argc != 5)
    {
        fprintf(stderr, "\nUsage: %s <header file> <data file> <layoutType = 0 for NCDHW / 1 for NDHWC> <testCase = 0 to 1>\n", argv[0]);
        exit(1);
    }

    header_file = argv[1];
    data_file = argv[2];
    layoutType = atoi(argv[3]); // 0 for NCDHW / 1 for NDHWC
    testCase = atoi(argv[4]); // 0 to 1

    if ((layoutType != 0) && (layoutType != 1))
    {
        fprintf(stderr, "\nUsage: %s <header file> <data file> <layoutType = 0 for NCDHW / 1 for NDHWC>\n", argv[0]);
        exit(1);
    }
    if ((testCase < 0) || (testCase > 1))
    {
        fprintf(stderr, "\nUsage: %s <header file> <data file> <layoutType = 0 for NCDHW / 1 for NDHWC>\n", argv[0]);
        exit(1);
    }

    NIFTI_DATATYPE *niftiData = NULL;
    nifti_1_header niftiHeader;

    // read nifti header file
    read_nifti_header_file(header_file, &niftiHeader);

    // allocate buffer and read first 3D volume from data file
    uint dataSize = niftiHeader.dim[1] * niftiHeader.dim[2] * niftiHeader.dim[3];
    uint dataSizeInBytes = dataSize * sizeof(NIFTI_DATATYPE);
    niftiData = (NIFTI_DATATYPE *) malloc(dataSizeInBytes);
    if (niftiData == NULL)
    {
        fprintf(stderr, "\nError allocating data buffer for %s\n",data_file);
        exit(1);
    }

    // read nifti data file
    read_nifti_data_file(data_file, &niftiHeader, niftiData);

    // set parameters to load into descriptor3D
    int batchSize, maxX, maxY, maxZ, numChannels, offsetInBytes;
    batchSize = 1;                      // Can be modified for batch processing
    maxX = niftiHeader.dim[1];          // Can be modified to obtain maxX of multiple 3D images for batch processing
    maxY = niftiHeader.dim[2];          // Can be modified to obtain maxY of multiple 3D images for batch processing
    maxZ = niftiHeader.dim[3];          // Can be modified to obtain maxZ of multiple 3D images for batch processing
    numChannels = niftiHeader.dim[4];
    offsetInBytes = 0;

    // optionally set maxX as a multiple of 8 for RPP optimal CPU/GPU processing
    maxX = ((maxX / 8) * 8) + 8;

    // set src/dst generic tensor descriptors
    RpptGenericDesc descriptor3D;
    RpptGenericDescPtr descriptorPtr3D = &descriptor3D;
    set_generic_descriptor(descriptorPtr3D, batchSize, maxX, maxY, maxZ, numChannels, offsetInBytes, layoutType);

    // set src/dst xyzwhd ROI tensors
    void *pinnedMemROI;
    hipHostMalloc(&pinnedMemROI, batchSize * sizeof(RpptRoiXyzwhd));
    RpptRoiXyzwhd *roiGenericSrcPtr = reinterpret_cast<RpptRoiXyzwhd *>(pinnedMemROI);
    roiGenericSrcPtr[0].xyz.x = 0;                      // start X dim
    roiGenericSrcPtr[0].xyz.y = 0;                      // start Y dim
    roiGenericSrcPtr[0].xyz.z = 0;                      // start Z dim
    roiGenericSrcPtr[0].roiWidth = niftiHeader.dim[1];  // length in X dim
    roiGenericSrcPtr[0].roiHeight = niftiHeader.dim[2]; // length in Y dim
    roiGenericSrcPtr[0].roiDepth = niftiHeader.dim[3];  // length in Z dim

    // Set buffer sizes in pixels for src/dst
    Rpp64u iBufferSize = (Rpp64u)descriptorPtr3D->strides[0] * (Rpp64u)descriptorPtr3D->dims[0];
    Rpp64u oBufferSize = iBufferSize;   // User can provide a different oBufferSize

    // Set buffer sizes in bytes for src/dst (including offsets)
    Rpp64u iBufferSizeInBytes = iBufferSize * sizeof(float) + descriptorPtr3D->offsetInBytes;
    Rpp64u oBufferSizeInBytes = iBufferSizeInBytes;

    // Allocate host memory in float for RPP strided buffer
    float *inputF32 = static_cast<float *>(calloc(iBufferSizeInBytes, 1));
    float *outputF32 = static_cast<float *>(calloc(oBufferSizeInBytes, 1));

    // Convert default NIFTI_DATATYPE unstrided buffer to RpptDataType::F32 strided buffer
    convert_input_niftitype_to_float_generic(niftiData, &niftiHeader, inputF32, descriptorPtr3D);

    // Allocate hip memory in float for RPP strided buffer
    void *d_inputF32, *d_outputF32;
    hipMalloc(&d_inputF32, iBufferSizeInBytes);
    hipMalloc(&d_outputF32, oBufferSizeInBytes);

    // Copy input buffer to hip
    hipMemcpy(d_inputF32, inputF32, iBufferSizeInBytes, hipMemcpyHostToDevice);

    // set argument tensors
    void *pinnedMemArgs;
    hipHostMalloc(&pinnedMemArgs, 2 * batchSize * sizeof(Rpp32f));

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, batchSize);

    // case-wise RPP API

    int missingFuncFlag = 0;
    double startWallTime, endWallTime, wallTime;
    switch (testCase)
    {
    case 0:
    {
        Rpp32f *mulTensor = reinterpret_cast<Rpp32f *>(pinnedMemArgs);
        Rpp32f *addTensor = mulTensor + batchSize;

        for (int i = 0; i < batchSize; i++)
        {
            mulTensor[i] = 80;
            addTensor[i] = 5;
        }

        startWallTime = omp_get_wtime();
        rppt_fmadd_scalar_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, mulTensor, addTensor, roiGenericSrcPtr, handle);
        break;
    }
    case 1:
    {
        startWallTime = omp_get_wtime();
        break;
    }
    default:
    {
        missingFuncFlag = 1;
        break;
    }
    }

    hipDeviceSynchronize();
    endWallTime = omp_get_wtime();
    wallTime = endWallTime - startWallTime;
    wallTime *= 1000;
    if (missingFuncFlag == 1)
    {
        printf("\nThe functionality doesn't yet exist in RPP\n");
        return -1;
    }
    cout << "\n\nGPU Backend Wall Time: " << wallTime <<" ms per nifti file"<< endl;

    // Copy output buffer to host
    hipMemcpy(outputF32, d_outputF32, oBufferSizeInBytes, hipMemcpyDeviceToHost);

    // Convert RpptDataType::F32 strided buffer to default NIFTI_DATATYPE unstrided buffer
    convert_output_float_to_niftitype_generic(outputF32, descriptorPtr3D, niftiData, &niftiHeader);

    // optionally normalize and write specific zPlanes to jpg images or mp4 video
    uchar *niftiDataU8 = (uchar *) malloc(dataSizeInBytes);
    NIFTI_DATATYPE min = niftiData[0];
    NIFTI_DATATYPE max = niftiData[0];
    for (int i = 0; i < dataSize; i++)
    {
        min = std::min(min, niftiData[i]);
        max = std::max(max, niftiData[i]);
    }
    float multiplier = 255.0f / (max - min);
    for (int i = 0; i < dataSize; i++)
        niftiDataU8[i] = (uchar)((niftiData[i] - min) * multiplier);
    for (int zFrame = 0; zFrame < niftiHeader.dim[3]; zFrame++)
        write_image_from_nifti_opencv(niftiDataU8, &niftiHeader, zFrame);
    // int zPlaneMin = 0, zPlaneMax = niftiHeader.dim[3] - 1;
    // write_video_from_nifti_opencv(niftiDataU8, &niftiHeader, zPlaneMin, zPlaneMax);

    // write nifti file
    write_nifti_file(&niftiHeader, niftiData);

    rppDestroyGPU(handle);

    // Free memory
    free(niftiData);
    free(inputF32);
    free(outputF32);
    hipHostFree(pinnedMemROI);
    hipHostFree(pinnedMemArgs);
    hipFree(d_inputF32);
    hipFree(d_outputF32);

    system("convert -delay 10 -loop 0 $(ls -v | grep jpg) niftiOutput.gif");

    return(0);
}