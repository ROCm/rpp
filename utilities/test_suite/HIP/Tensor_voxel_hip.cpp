#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nifti1.h"

using namespace std;
typedef int16_t NIFTI_DATATYPE;

#define MIN_HEADER_SIZE 348

// reads nifti-1 header file
static int read_nifti_header_file(char* const hdr_file, nifti_1_header *niftiHeader)
{
    nifti_1_header hdr;

    // open and read header
    FILE *fp = fopen(hdr_file,"r");
    if (fp == NULL)
    {
        fprintf(stderr, "\nError opening header file %s\n", hdr_file);
        exit(1);
    }
    int ret = fread(&hdr, MIN_HEADER_SIZE, 1, fp);
    if (ret != 1)
    {
        fprintf(stderr, "\nError reading header file %s\n", hdr_file);
        exit(1);
    }
    fclose(fp);

    // print header information
    fprintf(stderr, "\n%s header information:", hdr_file);
    fprintf(stderr, "\nXYZT dimensions: %d %d %d %d", hdr.dim[1], hdr.dim[2], hdr.dim[3], hdr.dim[4]);
    fprintf(stderr, "\nDatatype code and bits/pixel: %d %d", hdr.datatype, hdr.bitpix);
    fprintf(stderr, "\nScaling slope and intercept: %.6f %.6f", hdr.scl_slope, hdr.scl_inter);
    fprintf(stderr, "\nByte offset to data in datafile: %ld", (long)(hdr.vox_offset));
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

int main(int argc, char * argv[])
{
    char *hdr_file, *data_file;

    if (argc != 3)
    {
        fprintf(stderr, "\nUsage: %s <-r|-w> <header file> <data file>\n",argv[0]);
        exit(1);
    }

    hdr_file = argv[1];
    data_file = argv[2];

    NIFTI_DATATYPE *niftiData = NULL;
    nifti_1_header niftiHeader;

    // read nifti header file
    read_nifti_header_file(hdr_file, &niftiHeader);

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

    free(niftiData);

    return(0);
}