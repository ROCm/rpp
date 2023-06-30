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
#define RPPRANGECHECK(value)     (value < -32768) ? -32768 : ((value < 32767) ? value : 32767)

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

inline void write_image_from_nifti_opencv(uchar *niftiDataXYFrameU8, int niftiHeaderImageWidth, RpptRoiXyzwhd *roiGenericSrcPtr, uchar *outputBufferOpenCV, int zPlane, int Channel)
{
    uchar *outputBufferOpenCVRow = outputBufferOpenCV;
    uchar *niftiDataXYFrameU8Row = niftiDataXYFrameU8;
    for(int i = 0; i < roiGenericSrcPtr[0].roiHeight; i++)
    {
        memcpy(outputBufferOpenCVRow, niftiDataXYFrameU8Row, roiGenericSrcPtr[0].roiWidth);
        outputBufferOpenCVRow += roiGenericSrcPtr[0].roiWidth;
        niftiDataXYFrameU8Row += niftiHeaderImageWidth;
    }
    cv::Mat matOutputImage = cv::Mat(roiGenericSrcPtr[0].roiHeight, roiGenericSrcPtr[0].roiWidth, CV_8UC1, outputBufferOpenCV);
    string fileName = "nifti_single_zPlane_chn_"+ std::to_string(Channel)+ "_" + std::to_string(zPlane) + ".jpg";
    cv::imwrite(fileName, matOutputImage);

    // nifti_1_header hdr = *niftiHeader;
    // int xyFrameSize = hdr.dim[1] * hdr.dim[2];
    // uchar *niftiDataU8Temp = &niftiDataU8[xyFrameSize * zPlane];
    // cv::Mat matOutputImage = cv::Mat(hdr.dim[2], hdr.dim[1], CV_8UC1, niftiDataU8Temp);
    // string fileName = "nifti_single_zPlane_" + std::to_string(zPlane) + ".jpg";
    // cv::imwrite(fileName, matOutputImage);
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

//     for (int zPlane = zPlaneMin; zPlane < zPlaneMax; zPlane++)
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
                            *outputF32Temp = RPPRANGECHECK(*outputF32Temp);
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
    int layoutType, testCase, testType;
    char *header_file, *data_file;

    if (argc != 6)
    {
        fprintf(stderr, "\nUsage: %s <header file> <data file> <layoutType = 0 for NCDHW / 1 for NDHWC> <testCase = 0 to 1> <testType = 0 - unit test/ 1 - performance test>\n", argv[0]);
        exit(1);
    }

    header_file = argv[1];
    data_file = argv[2];
    layoutType = atoi(argv[3]); // 0 for NCDHW / 1 for NDHWC
    testCase = atoi(argv[4]); // 0 to 1
    testType = atoi(argv[5]); // 0 - unit test / 1 - performance test

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

	// Set ROI tensors types for src
    RpptRoi3DType roiTypeSrc;
    roiTypeSrc = RpptRoi3DType::XYZWHD;

    // allocate buffer and read first 3D volume from data file
    uint dataSize = niftiHeader.dim[1] * niftiHeader.dim[2] * niftiHeader.dim[3];
    uint dataSizeInBytes = dataSize * sizeof(NIFTI_DATATYPE);
    niftiData = (NIFTI_DATATYPE *) calloc(dataSizeInBytes, 1);
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
    numChannels = 3;                    //Temporary value set to 3 for running pln3, the actual value should be obtained from niftiHeader.dim[4].
    offsetInBytes = 0;

    // optionally set maxX as a multiple of 8 for RPP optimal CPU/GPU processing
    maxX = ((maxX / 8) * 8) + 8;

    // set src/dst generic tensor descriptors
    RpptGenericDesc descriptor3D;
    RpptGenericDescPtr descriptorPtr3D = &descriptor3D;
    set_generic_descriptor(descriptorPtr3D, batchSize, maxX, maxY, maxZ, numChannels, offsetInBytes, layoutType);

    // set src/dst xyzwhd ROI tensors
    //RpptRoiXyzwhd *roiGenericSrcPtr = reinterpret_cast<RpptRoiXyzwhd *>(calloc(batchSize, sizeof(RpptRoiXyzwhd)));
    RpptROI3D *roiGenericSrcPtr = (RpptROI3D *) calloc(batchSize, sizeof(RpptROI3D));

    // optionally pick full image as ROI or a smaller slice of the 3D tensor in X/Y/Z dimensions
    // option 1 - test using roi as the whole 3D image - not sliced (example for 240 x 240 x 155 x 1)
    // roiGenericSrcPtr[0].xyzwhdROI.xyz.x = 0;                              // start X dim = 0
    // roiGenericSrcPtr[0].xyzwhdROI.xyz.y = 0;                              // start Y dim = 0
    // roiGenericSrcPtr[0].xyzwhdROI.xyz.z = 0;                              // start Z dim = 0
    // roiGenericSrcPtr[0].xyzwhdROI.roiWidth = niftiHeader.dim[1];          // length in X dim = 240
    // roiGenericSrcPtr[0].xyzwhdROI.roiHeight = niftiHeader.dim[2];         // length in Y dim = 240
    // roiGenericSrcPtr[0].xyzwhdROI.roiDepth = niftiHeader.dim[3];          // length in Z dim = 155
    // option 2 - test using roi as a smaller 3D tensor slice - sliced in X, Y and Z dims (example for 240 x 240 x 155 x 1)
    // roiGenericSrcPtr[0].xyzwhdROI.xyz.x = niftiHeader.dim[1] / 4;         // start X dim = 60
    // roiGenericSrcPtr[0].xyzwhdROI.xyz.y = niftiHeader.dim[2] / 4;         // start Y dim = 60
    // roiGenericSrcPtr[0].xyzwhdROI.xyz.z = niftiHeader.dim[3] / 3;         // start Z dim = 51
    // roiGenericSrcPtr[0].xyzwhdROI.roiWidth = niftiHeader.dim[1] / 2;      // length in X dim = 120
    // roiGenericSrcPtr[0].xyzwhdROI.roiHeight = niftiHeader.dim[2] / 2;     // length in Y dim = 120
    // roiGenericSrcPtr[0].xyzwhdROI.roiDepth = niftiHeader.dim[3] / 3;      // length in Z dim = 51
    // option 3 - test using roi as a smaller 3D tensor slice - sliced in only Z dim (example for 240 x 240 x 155 x 1)
    // roiGenericSrcPtr[0].xyzwhdROI.xyz.x = 0;                              // start X dim = 0
    // roiGenericSrcPtr[0].xyzwhdROI.xyz.y = 0;                              // start Y dim = 0
    // roiGenericSrcPtr[0].xyzwhdROI.xyz.z = niftiHeader.dim[3] / 3;         // start Z dim = 51
    // roiGenericSrcPtr[0].xyzwhdROI.roiWidth = niftiHeader.dim[1];          // length in X dim = 240
    // roiGenericSrcPtr[0].xyzwhdROI.roiHeight = niftiHeader.dim[2];         // length in Y dim = 240
    // roiGenericSrcPtr[0].xyzwhdROI.roiDepth = niftiHeader.dim[3] / 3;      // length in Z dim = 51
    // option 4 - test using roi as a smaller 3D tensor slice - sliced in only X and Z dim (example for 240 x 240 x 155 x 1)
    roiGenericSrcPtr[0].xyzwhdROI.xyz.x = niftiHeader.dim[1] / 5;         // start X dim = 48
    roiGenericSrcPtr[0].xyzwhdROI.xyz.y = 0;                              // start Y dim = 0
    roiGenericSrcPtr[0].xyzwhdROI.xyz.z = niftiHeader.dim[3] / 3;         // start Z dim = 51
    roiGenericSrcPtr[0].xyzwhdROI.roiWidth = niftiHeader.dim[1] * 3 / 5;  // length in X dim = 144
    roiGenericSrcPtr[0].xyzwhdROI.roiHeight = niftiHeader.dim[2];         // length in Y dim = 240
    roiGenericSrcPtr[0].xyzwhdROI.roiDepth = niftiHeader.dim[3] / 3;      // length in Z dim = 51

    // Set buffer sizes in pixels for src/dst
    Rpp64u iBufferSize = (Rpp64u)descriptorPtr3D->strides[0] * (Rpp64u)descriptorPtr3D->dims[0] * numChannels;
    Rpp64u oBufferSize = iBufferSize;   // User can provide a different oBufferSize

    // Set buffer sizes in bytes for src/dst (including offsets)
    Rpp64u iBufferSizeInBytes = iBufferSize * sizeof(float) + descriptorPtr3D->offsetInBytes;
    Rpp64u oBufferSizeInBytes = iBufferSizeInBytes;

    // Allocate host memory in float for RPP strided buffer
    float *inputF32 = static_cast<float *>(calloc(iBufferSizeInBytes, 1));
    float *outputF32 = static_cast<float *>(calloc(oBufferSizeInBytes, 1));

    // Convert default NIFTI_DATATYPE unstrided buffer to RpptDataType::F32 strided buffer
    Rpp64u increment = ((Rpp64u)descriptorPtr3D->strides[0] * (Rpp64u)descriptorPtr3D->dims[0]);
    for(int i = 0; i < numChannels; i++) //temporary change to process pln3
    {
        convert_input_niftitype_to_float_generic(niftiData, &niftiHeader, inputF32 + i * increment, descriptorPtr3D);
    }

    // set argument tensors
    void *pinnedMemArgs;
    pinnedMemArgs = calloc(2 * batchSize , sizeof(Rpp32f));

    // Run case-wise RPP API and measure time
    // Set the number of threads to be used by OpenMP pragma for RPP batch processing on host.
    // If numThreads value passed is 0, number of OpenMP threads used by RPP will be set to batch size
    Rpp32u numThreads = 0;
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, batchSize, numThreads);

    // case-wise RPP API

    int numRuns = 1;
    if(testType == 1)
        numRuns = 100;

    int missingFuncFlag = 0;
    double startWallTime, endWallTime, wallTime;
    double maxWallTime = 0, minWallTime = 5000, avgWallTime = 0;
    for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
    {
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
            rppt_fmadd_scalar_host(inputF32, descriptorPtr3D, outputF32, descriptorPtr3D, mulTensor, addTensor, roiGenericSrcPtr, roiTypeSrc, handle);
            break;
        }
        case 1:
        {
            startWallTime = omp_get_wtime();
            rppt_slice_host(inputF32, descriptorPtr3D, outputF32, descriptorPtr3D, roiGenericSrcPtr, roiTypeSrc, handle);
            break;
        }
        default:
        {
            missingFuncFlag = 1;
            break;
        }
        }

        endWallTime = omp_get_wtime();
        wallTime = endWallTime - startWallTime;
        maxWallTime = std::max(maxWallTime, wallTime);
        minWallTime = std::min(minWallTime, wallTime);
        avgWallTime += wallTime;
        wallTime *= 1000;
        if (missingFuncFlag == 1)
        {
            printf("\nThe functionality doesn't yet exist in RPP\n");
            return -1;
        }
        if(testType == 0)
            cout << "\n\nCPU Backend Wall Time: " << wallTime <<" ms per nifti file"<< endl;
    }

    if(testType == 1)
    {
        // Display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= numRuns;
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }

    if(testType == 0)
    {
        for(int i = 0; i < numChannels; i++) // temporary changes to process pln3
        {
            // Convert RpptDataType::F32 strided buffer to default NIFTI_DATATYPE unstrided buffer
            Rpp64u increment = ((Rpp64u)descriptorPtr3D->strides[0] * (Rpp64u)descriptorPtr3D->dims[0]);
            convert_output_float_to_niftitype_generic(outputF32 + i * increment, descriptorPtr3D, niftiData, &niftiHeader);

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
            int xyFrameSize = niftiHeader.dim[1] * niftiHeader.dim[2];
            int xyFrameSizeROI = roiGenericSrcPtr[0].xyzwhdROI.roiWidth * roiGenericSrcPtr[0].xyzwhdROI.roiHeight;
            uchar *outputBufferOpenCV = (uchar *)calloc(xyFrameSizeROI, sizeof(uchar));
            for (int zPlane = roiGenericSrcPtr[0].xyzwhdROI.xyz.z; zPlane < roiGenericSrcPtr[0].xyzwhdROI.xyz.z + roiGenericSrcPtr[0].xyzwhdROI.roiDepth; zPlane++)
            {
                write_image_from_nifti_opencv(niftiDataU8, niftiHeader.dim[1], (RpptRoiXyzwhd *)roiGenericSrcPtr, outputBufferOpenCV, zPlane, i);
                niftiDataU8 += xyFrameSize;
            }
            // int zPlaneMin = 0, zPlaneMax = niftiHeader.dim[3] - 1;
            // write_video_from_nifti_opencv(niftiDataU8, &niftiHeader, zPlaneMin, zPlaneMax);

            // write nifti file
            write_nifti_file(&niftiHeader, niftiData);
            std::string command = "convert -delay 10 -loop 0 $(ls -v | grep jpg) niftiOutput_chn" + std::to_string(i) + ".gif";
            system(command.c_str());
            free(outputBufferOpenCV);
        }
    }

    rppDestroyHost(handle);

    // Free memory
    free(niftiData);
    free(inputF32);
    free(outputF32);
    free(roiGenericSrcPtr);
    free(pinnedMemArgs);

    return(0);
}