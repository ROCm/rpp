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
#include "rpp.h"
#include "nifti1.h"

using namespace std;
namespace fs = boost::filesystem;
typedef int16_t NIFTI_DATATYPE;

#define MIN_HEADER_SIZE 348
#define RPPRANGECHECK(value)     (value < -32768) ? -32768 : ((value < 32767) ? value : 32767)
#define DEBUG_MODE 1
#define TOLERANCE 0.01

// Opens a folder and recursively search for .nii files
void open_folder(const string& folderPath, vector<string>& niiFileNames, vector<string>& niiFilePath)
{
    auto src_dir = opendir(folderPath.c_str());
    struct dirent* entity;
    std::string fileName = " ";

    if (src_dir == nullptr)
        std::cerr << "\n ERROR: Failed opening the directory at " <<folderPath;

    while((entity = readdir(src_dir)) != nullptr)
    {
        string entry_name(entity->d_name);
        if (entry_name == "." || entry_name == "..")
            continue;
        fileName = entity->d_name;
        std::string filePath = folderPath;
        filePath.append("/");
        filePath.append(entity->d_name);
        fs::path pathObj(filePath);
        if(fs::exists(pathObj) && fs::is_directory(pathObj))
            open_folder(filePath, niiFileNames, niiFilePath);

        if (fileName.size() > 4 && fileName.substr(fileName.size() - 4) == ".nii")
        {
            niiFilePath.push_back(filePath);
            niiFileNames.push_back(entity->d_name);
        }
    }
    if(niiFileNames.empty())
        std::cerr << "\n Did not load any file from " << folderPath;

    closedir(src_dir);
}

// Searches for .nii files in input folders
void search_nii_files(const string& folder_path, vector<string>& niiFileNames, vector<string>& niiFilePath)
{
    vector<string> entry_list;
    string full_path = folder_path;
    auto sub_dir = opendir(folder_path.c_str());
    if (!sub_dir)
    {
        std::cerr << "ERROR: Failed opening the directory at "<< folder_path << std::endl;
        exit(0);
    }

    struct dirent* entity;
    while ((entity = readdir(sub_dir)) != nullptr)
    {
        string entry_name(entity->d_name);
        if (entry_name == "." || entry_name == "..")
            continue;
        entry_list.push_back(entry_name);
    }
    closedir(sub_dir);
    sort(entry_list.begin(), entry_list.end());

    for (unsigned dir_count = 0; dir_count < entry_list.size(); ++dir_count)
    {
        string subfolder_path = full_path + "/" + entry_list[dir_count];
        fs::path pathObj(subfolder_path);
        if (fs::exists(pathObj) && fs::is_regular_file(pathObj))
        {
            // ignore files with extensions .tar, .zip, .7z
            auto file_extension_idx = subfolder_path.find_last_of(".");
            if (file_extension_idx != std::string::npos)
            {
                std::string file_extension = subfolder_path.substr(file_extension_idx+1);
                if ((file_extension == "tar") || (file_extension == "zip") || (file_extension == "7z") || (file_extension == "rar"))
                    continue;
            }
            if (entry_list[dir_count].size() > 4 && entry_list[dir_count].substr(entry_list[dir_count].size() - 4) == ".nii")
            {
                niiFileNames.push_back(entry_list[dir_count]);
                niiFilePath.push_back(subfolder_path);
            }
        }
        else if (fs::exists(pathObj) && fs::is_directory(pathObj))
            open_folder(subfolder_path, niiFileNames, niiFilePath);
    }
}

inline void remove_substring(string &str, string &pattern)
{
    std::string::size_type i = str.find(pattern);
    while (i != std::string::npos)
    {
        str.erase(i, pattern.length());
        i = str.find(pattern, i);
   }
}

inline void compare_output(Rpp32f* output, Rpp64u oBufferSize, string func, int layoutType)
{
    string refPath = get_current_dir_name();
    string pattern = "/build";
    remove_substring(refPath, pattern);
    string csvName = "";
    if(layoutType = 0)
        csvName = func + "_nifti_output_pkd3.csv";
    else if(layoutType == 1)
        csvName = func + "_nifti_output_pln3.csv";
    else
        csvName = func + "_nifti_output_pln1.csv";

    string refFile = refPath + "/../REFERENCE_OUTPUT_VOXEL/"+ func + "/" + csvName;

    ifstream file(refFile);
    int *refOutput;
    refOutput = (int *)malloc(oBufferSize * sizeof(Rpp32f));
    string line,word;
    int index = 0;
    int mismatches = 0;

    // Load the refennce output values from files and store in vector
    if(file.is_open())
    {
        while(getline(file, line))
        {
            stringstream str(line);
            while(getline(str, word, ','))
            {
                refOutput[index] = stoi(word);
                index++;
            }
        }
    }
    else
    {
        cout<<"Could not open the reference output. Please check the path specified\n";
        return;
    }
    for(int i = 0; i < oBufferSize; i++)
    {
        if(refOutput[i] != static_cast<int>(output[i]))
            mismatches++;
    }
     std::cerr<<"\n "<<mismatches;
    std::cerr << std::endl << "Results for " << func << " :" << std::endl;
    std::string status = func + ": ";
    if(mismatches <= (TOLERANCE * oBufferSize))
    {
        std::cerr << "PASSED!" << std::endl;
        status += "PASSED";
    }
    else
    {
        std::cerr << "FAILED! ";
        status += "FAILED";
    }
    free(refOutput);
}

// sets generic descriptor dimensions and strides of src/dst
inline void set_generic_descriptor(RpptGenericDescPtr descriptorPtr3D, int noOfImages, int maxX, int maxY, int maxZ, int numChannels, int offsetInBytes, int layoutType)
{
    descriptorPtr3D->numDims = 5;
    descriptorPtr3D->offsetInBytes = offsetInBytes;
    descriptorPtr3D->dataType = RpptDataType::F32;

    if (layoutType == 0)
    {
        descriptorPtr3D->layout = RpptLayout::NDHWC;
        descriptorPtr3D->dims[0] = noOfImages;
        descriptorPtr3D->dims[1] = maxZ;
        descriptorPtr3D->dims[2] = maxY;
        descriptorPtr3D->dims[3] = maxX;
        descriptorPtr3D->dims[4] = numChannels;
    }
    else if (layoutType == 1 || layoutType == 2)
    {
        descriptorPtr3D->layout = RpptLayout::NCDHW;
        descriptorPtr3D->dims[0] = noOfImages;
        descriptorPtr3D->dims[1] = numChannels;
        descriptorPtr3D->dims[2] = maxZ;
        descriptorPtr3D->dims[3] = maxY;
        descriptorPtr3D->dims[4] = maxX;
    }

    descriptorPtr3D->strides[0] = descriptorPtr3D->dims[1] * descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[1] = descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[2] = descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[3] = descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[4] = 1;
}

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

inline void write_nifti_file(nifti_1_header *niftiHeader, NIFTI_DATATYPE *niftiData, int batchCount)
{
    nifti_1_header hdr = *niftiHeader;
    // nifti1_extender pad = {0,0,0,0};
    FILE *fp;
    int ret, i;

    // write first hdr.vox_offset bytes of header
    string niiOutputString = std::to_string(batchCount)+"_nifti_output.nii";
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
    // ret = fwrite(&pad, 4, 1, fp);
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

inline void write_image_from_nifti_opencv(uchar *niftiDataXYFrameU8, int niftiHeaderImageWidth, RpptRoiXyzwhd *roiGenericSrcPtr, uchar *outputBufferOpenCV, int zPlane, int Channel, int batchCount)
{
    uchar *outputBufferOpenCVRow = outputBufferOpenCV;
    uchar *niftiDataXYFrameU8Row = niftiDataXYFrameU8;
    for(int i = 0; i < roiGenericSrcPtr[batchCount].roiHeight; i++)
    {
        memcpy(outputBufferOpenCVRow, niftiDataXYFrameU8Row, roiGenericSrcPtr[batchCount].roiWidth);
        outputBufferOpenCVRow += roiGenericSrcPtr[batchCount].roiWidth;
        niftiDataXYFrameU8Row += niftiHeaderImageWidth;
    }
    cv::Mat matOutputImage = cv::Mat(roiGenericSrcPtr[batchCount].roiHeight, roiGenericSrcPtr[batchCount].roiWidth, CV_8UC1, outputBufferOpenCV);
    string fileName = "nifti_" + std::to_string(batchCount) + "_zPlane_chn_"+ std::to_string(Channel)+ "_" + std::to_string(zPlane) + ".jpg";
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
inline void convert_input_niftitype_to_Rpp32f_generic(T **niftyInput, nifti_1_header headerData[], Rpp32f *inputF32, RpptGenericDescPtr descriptorPtr3D)
{
     bool replicateToAllChannels = false;
    // nifti_1_header headerData[] = niftiHeader;
    Rpp32u depthStride, rowStride, channelStride, channelIncrement;
    // Rpp32u niftyStride = headerData.dim[1] * headerData.dim[2] * headerData.dim[3];
    if (descriptorPtr3D->layout == RpptLayout::NCDHW)
    {
        depthStride = descriptorPtr3D->strides[2];
        rowStride = descriptorPtr3D->strides[3];
        channelStride = descriptorPtr3D->strides[1];
        channelIncrement = 1;
        // niftyStride = niftyStride * descriptorPtr3D->dims[1];
        // replicateToAllChannels = (descriptorPtr3D->dims[1] == 3 && headerData.dim[4] == 1);
        if(descriptorPtr3D->dims[1] == 3)
            replicateToAllChannels = true;                            //temporary chnage to replicate the data for pln3 using pln1 data
    }
    else if (descriptorPtr3D->layout == RpptLayout::NDHWC)
    {
        depthStride = descriptorPtr3D->strides[1];
        rowStride = descriptorPtr3D->strides[2];
        channelStride = 1;
        channelIncrement = 3;
        // niftyStride = niftyStride * descriptorPtr3D->dims[4];
        // replicateToAllChannels = (descriptorPtr3D->dims[4] == 3 && headerData.dim[4] == 1);
        replicateToAllChannels = true;
    }
    if (replicateToAllChannels)
    {
        for (int batchcount = 0; batchcount < descriptorPtr3D->dims[0]; batchcount++)
        {
            T *niftyInputTemp = niftyInput[batchcount];
            Rpp32f *outputF32Temp = inputF32 + batchcount * descriptorPtr3D->strides[0];
            Rpp32f *outputChannelR = outputF32Temp;
            Rpp32f *outputChannelG = outputChannelR + channelStride;
            Rpp32f *outputChannelB = outputChannelG + channelStride;
            for (int d = 0; d < headerData[batchcount].dim[3]; d++)
            {
                Rpp32f *outputDepthR = outputChannelR;
                Rpp32f *outputDepthG = outputChannelG;
                Rpp32f *outputDepthB = outputChannelB;
                for (int h = 0; h < headerData[batchcount].dim[2]; h++)
                {
                    Rpp32f *outputRowR = outputDepthR;
                    Rpp32f *outputRowG = outputDepthG;
                    Rpp32f *outputRowB = outputDepthB;
                    for (int w = 0; w < headerData[batchcount].dim[1]; w++)
                    {
                        *outputRowR = static_cast<Rpp32f>(*niftyInputTemp);
                        *outputRowG = static_cast<Rpp32f>(*niftyInputTemp);
                        *outputRowB = static_cast<Rpp32f>(*niftyInputTemp);

                        niftyInputTemp++;
                        outputRowR += channelIncrement;
                        outputRowG += channelIncrement;
                        outputRowB += channelIncrement;
                    }
                    outputDepthR += rowStride;
                    outputDepthG += rowStride;
                    outputDepthB += rowStride;
                }
                outputChannelR += depthStride;
                outputChannelG += depthStride;
                outputChannelB += depthStride;
            }
        }
    }
    else
    {
        for (int batchcount = 0; batchcount < descriptorPtr3D->dims[0]; batchcount++)
        {
            T *niftyInputTemp = niftyInput[batchcount];
            Rpp32f *outputTemp = inputF32 + batchcount * descriptorPtr3D->strides[0];
            for (int c = 0; c < headerData[batchcount].dim[4]; c++)
            {
                Rpp32f *outputChannel = outputTemp;
                for (int d = 0; d < headerData[batchcount].dim[3]; d++)
                {
                    Rpp32f *outputDepth = outputChannel;
                    for (int h = 0; h < headerData[batchcount].dim[2]; h++)
                    {
                        Rpp32f *outputRow = outputDepth;
                        for (int w = 0; w < headerData[batchcount].dim[1]; w++)
                        {
                            *outputRow++ = static_cast<Rpp32f>(*niftyInputTemp++);
                        }
                        outputDepth += rowStride;
                    }
                    outputChannel += depthStride;
                }
                outputTemp += channelStride;
            }
        }
    }
}

// Convert RpptDataType::F32 strided buffer to default NIFTI_DATATYPE unstrided buffer
template<typename T>
inline void convert_output_Rpp32f_to_niftitype_generic(Rpp32f *input, RpptGenericDescPtr descriptorPtr3D, T *niftyOutput, nifti_1_header *niftiHeader)
{
    nifti_1_header headerData = *niftiHeader;
    Rpp32u niftyStride = headerData.dim[1] * headerData.dim[2] * headerData.dim[3];
    if (descriptorPtr3D->layout == RpptLayout::NCDHW)
    {
        niftyStride = niftyStride * descriptorPtr3D->dims[1];
        Rpp32f *inputTemp = input;
        T *niftyOutputTemp = niftyOutput;
        for (int d = 0; d < headerData.dim[3]; d++)
        {
            Rpp32f *inputDepth = inputTemp;
            for (int h = 0; h < headerData.dim[2]; h++)
            {
                Rpp32f *inputRow = inputDepth;
                for (int w = 0; w < headerData.dim[1]; w++)
                {
                    *inputRow = RPPRANGECHECK(*inputRow);
                    *niftyOutputTemp++ = (T)*inputRow++;
                }
                inputDepth += descriptorPtr3D->strides[3];
            }
            inputTemp += descriptorPtr3D->strides[2];
        }
    }
    else if (descriptorPtr3D->layout == RpptLayout::NDHWC)
    {
        niftyStride = niftyStride * descriptorPtr3D->dims[4];
        Rpp32f *inputTemp = input;
        T *niftyOutputTemp = niftyOutput;
        for (int d = 0; d < headerData.dim[3]; d++)
        {
            Rpp32f *inputDepth = inputTemp;
            for (int h = 0; h < headerData.dim[2]; h++)
            {
                Rpp32f *inputRow = inputDepth;
                for (int w = 0; w < headerData.dim[1]; w++)
                {
                    *inputRow = RPPRANGECHECK(*inputRow);
                    *niftyOutputTemp = (T)*inputRow;

                    inputRow += 3;
                    niftyOutputTemp++;
                }
                inputDepth += descriptorPtr3D->strides[2];
            }
            inputTemp += descriptorPtr3D->strides[1];
        }
    }
}

int main(int argc, char * argv[])
{
    int layoutType, testCase, testType, qaFlag;
    char *header_file, *data_file;

    if (argc < 7)
    {
        fprintf(stderr, "\nUsage: %s <header file> <data file> <layoutType = 0 - PKD3/ 1 - PLN3/ 2 - PLN1> <testCase = 0 to 1> <testType = 0 - unit test/ 1 - performance test>\n", argv[0]);
        exit(1);
    }

    header_file = argv[1];
    data_file = argv[2];
    layoutType = atoi(argv[3]); // 0 for PKD3 // 1 for PLN3 // 2 for PLN1
    testCase = atoi(argv[4]); // 0 to 1
    testType = atoi(argv[5]); // 0 - unit test / 1 - performance test
    qaFlag = atoi(argv[6]); //0 - QA disabled / 1 - QA enabled

    if ((layoutType < 0) || (layoutType > 2))
    {
        fprintf(stderr, "\nUsage: %s <header file> <data file> <layoutType = 0 - PKD3/ 1 - PLN3/ 2 - PLN1>\n", argv[0]);
        exit(1);
    }
    if ((testCase < 0) || (testCase > 4))
    {
        fprintf(stderr, "\nUsage: %s <header file> <data file> <layoutType = 0 for NCDHW / 1 for NDHWC>\n", argv[0]);
        exit(1);
    }

    int numChannels, offsetInBytes;
    int batchSize = 0, maxX = 0, maxY = 0, maxZ = 0;
    vector<string> headerNames, headerPath, dataFileNames, dataFilePath;
    search_nii_files(header_file, headerNames, headerPath);
    search_nii_files(data_file, dataFileNames, dataFilePath);
    batchSize = dataFileNames.size();

    // NIFTI_DATATYPE *niftiData = NULL;
    NIFTI_DATATYPE** niftiDataArray = (NIFTI_DATATYPE**)malloc(batchSize * sizeof(NIFTI_DATATYPE*));
    nifti_1_header niftiHeader[batchSize];

    // read nifti header file
    for(int i = 0; i < batchSize; i++)
    {
        read_nifti_header_file((char *)headerPath[i].c_str(), &niftiHeader[i]);
        // allocate buffer and read first 3D volume from data file
        uint dataSize = niftiHeader[i].dim[1] * niftiHeader[i].dim[2] * niftiHeader[i].dim[3];
        uint dataSizeInBytes = dataSize * sizeof(NIFTI_DATATYPE);
        niftiDataArray[i] = (NIFTI_DATATYPE *) calloc(dataSizeInBytes, 1);
        if (niftiDataArray[i] == NULL)
        {
            fprintf(stderr, "\nError allocating data buffer for %s\n",data_file);
            exit(1);
        }
        // read nifti data file
        read_nifti_data_file((char *)dataFilePath[i].c_str(), &niftiHeader[i], niftiDataArray[i]);
        maxX = max(static_cast<int>(niftiHeader[i].dim[1]), maxX);
        maxY = max(static_cast<int>(niftiHeader[i].dim[2]), maxY);
        maxZ = max(static_cast<int>(niftiHeader[i].dim[3]), maxZ);
    }

    // Set ROI tensors types for src
    RpptRoi3DType roiTypeSrc;
    roiTypeSrc = RpptRoi3DType::XYZWHD;

    numChannels = (layoutType == 2) ? 1: 3;                    //Temporary value set to 3 for running pln3, the actual value should be obtained from niftiHeader.dim[4].
    offsetInBytes = 0;

    // optionally set maxX as a multiple of 8 for RPP optimal CPU/GPU processing
    maxX = ((maxX / 8) * 8) + 8;

    // set src/dst generic tensor descriptors
    RpptGenericDesc descriptor3D;
    RpptGenericDescPtr descriptorPtr3D = &descriptor3D;
    set_generic_descriptor(descriptorPtr3D, batchSize, maxX, maxY, maxZ, numChannels, offsetInBytes, layoutType);

    // set src/dst xyzwhd ROI tensors
    //RpptRoiXyzwhd *roiGenericSrcPtr = reinterpret_cast<RpptRoiXyzwhd *>(calloc(batchSize, sizeof(RpptRoiXyzwhd)));
    void *pinnedMemROI;
    hipHostMalloc(&pinnedMemROI, batchSize * sizeof(RpptROI3D));
    RpptROI3D *roiGenericSrcPtr = reinterpret_cast<RpptROI3D *>(pinnedMemROI);

    // optionally pick full image as ROI or a smaller slice of the 3D tensor in X/Y/Z dimensions
    for(int i = 0; i < batchSize; i++)
    {
        // option 1 - test using roi as the whole 3D image - not sliced (example for 240 x 240 x 155 x 1)
        roiGenericSrcPtr[i].xyzwhdROI.xyz.x = 0;                              // start X dim = 0
        roiGenericSrcPtr[i].xyzwhdROI.xyz.y = 0;                              // start Y dim = 0
        roiGenericSrcPtr[i].xyzwhdROI.xyz.z = 0;                              // start Z dim = 0
        roiGenericSrcPtr[i].xyzwhdROI.roiWidth = niftiHeader[i].dim[1];          // length in X dim
        roiGenericSrcPtr[i].xyzwhdROI.roiHeight = niftiHeader[i].dim[2];         // length in Y dim
        roiGenericSrcPtr[i].xyzwhdROI.roiDepth = niftiHeader[i].dim[3];          // length in Z dim
        // option 2 - test using roi as a smaller 3D tensor slice - sliced in X, Y and Z dims (example for 240 x 240 x 155 x 1)
        // roiGenericSrcPtr[i].xyzwhdROI.xyz.x = niftiHeader.dim[1] / 4;         // start X dim = 60
        // roiGenericSrcPtr[i].xyzwhdROI.xyz.y = niftiHeader[i].dim[2] / 4;         // start Y dim = 60
        // roiGenericSrcPtr[i].xyzwhdROI.xyz.z = niftiHeader[i].dim[3] / 3;         // start Z dim = 51
        // roiGenericSrcPtr[i].xyzwhdROI.roiWidth = niftiHeader[i].dim[1] / 2;      // length in X dim = 120
        // roiGenericSrcPtr[i].xyzwhdROI.roiHeight = niftiHeader[i].dim[2] / 2;     // length in Y dim = 120
        // roiGenericSrcPtr[i].xyzwhdROI.roiDepth = niftiHeader[i].dim[3] / 3;      // length in Z dim = 51
        // option 3 - test using roi as a smaller 3D tensor slice - sliced in only Z dim (example for 240 x 240 x 155 x 1)
        // roiGenericSrcPtr[i].xyzwhdROI.xyz.x = 0;                              // start X dim = 0
        // roiGenericSrcPtr[i].xyzwhdROI.xyz.y = 0;                              // start Y dim = 0
        // roiGenericSrcPtr[i].xyzwhdROI.xyz.z = niftiHeader[i].dim[3] / 3;         // start Z dim = 51
        // roiGenericSrcPtr[i].xyzwhdROI.roiWidth = niftiHeader[i].dim[1];          // length in X dim = 240
        // roiGenericSrcPtr[i].xyzwhdROI.roiHeight = niftiHeader[i].dim[2];         // length in Y dim = 240
        // roiGenericSrcPtr[i].xyzwhdROI.roiDepth = niftiHeader[i].dim[3] / 3;      // length in Z dim = 51
        // option 4 - test using roi as a smaller 3D tensor slice - sliced in only X and Z dim (example for 240 x 240 x 155 x 1)
        // roiGenericSrcPtr[i].xyzwhdROI.xyz.x = niftiHeader[i].dim[1] / 5;         // start X dim = 48
        // roiGenericSrcPtr[i].xyzwhdROI.xyz.y = 0;                              // start Y dim = 0
        // roiGenericSrcPtr[i].xyzwhdROI.xyz.z = niftiHeader[i].dim[3] / 3;         // start Z dim = 51
        // roiGenericSrcPtr[i].xyzwhdROI.roiWidth = niftiHeader[i].dim[1] * 3 / 5;  // length in X dim = 144
        // roiGenericSrcPtr[i].xyzwhdROI.roiHeight = niftiHeader[i].dim[2];         // length in Y dim = 240
        // roiGenericSrcPtr[i].xyzwhdROI.roiDepth = niftiHeader[i].dim[3] / 3;      // length in Z dim = 51
    }

    // Set buffer sizes in pixels for src/dst
    Rpp64u iBufferSize = (Rpp64u)descriptorPtr3D->strides[0] * (Rpp64u)descriptorPtr3D->dims[0]; //  (d x h x w x c) x (n)
    Rpp64u oBufferSize = iBufferSize;   // User can provide a different oBufferSize

    // Set buffer sizes in bytes for src/dst (including offsets)
    Rpp64u iBufferSizeInBytes = iBufferSize * sizeof(Rpp32f) + descriptorPtr3D->offsetInBytes;
    Rpp64u oBufferSizeInBytes = iBufferSizeInBytes;

    // Allocate host memory in Rpp32f for RPP strided buffer
    Rpp32f *inputF32 = static_cast<Rpp32f *>(calloc(iBufferSizeInBytes, 1));
    Rpp32f *outputF32 = static_cast<Rpp32f *>(calloc(oBufferSizeInBytes, 1));

    // Convert default NIFTI_DATATYPE unstrided buffer to RpptDataType::F32 strided buffer
    convert_input_niftitype_to_Rpp32f_generic(niftiDataArray, niftiHeader, inputF32 , descriptorPtr3D);

    // Allocate hip memory in float for RPP strided buffer
    void *d_inputF32, *d_outputF32;
    hipMalloc(&d_inputF32, iBufferSizeInBytes);
    hipMalloc(&d_outputF32, oBufferSizeInBytes);

    // Copy input buffer to hip
    hipMemcpy(d_inputF32, inputF32, iBufferSizeInBytes, hipMemcpyHostToDevice);

    // set argument tensors
    void *pinnedMemArgs;
    pinnedMemArgs = calloc(2 * batchSize , sizeof(Rpp32f));

    // Set the number of threads to be used by OpenMP pragma for RPP batch processing on host.
    // If numThreads value passed is 0, number of OpenMP threads used by RPP will be set to batch size
    Rpp32u numThreads = 0;
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, batchSize, numThreads);

    // Run case-wise RPP API and measure time
    int numRuns = 1;
    if(testType == 1)
        numRuns = 1000;

    int missingFuncFlag = 0;
    double startWallTime, endWallTime, wallTime;
    double maxWallTime = 0, minWallTime = 5000, avgWallTime = 0;
    string testCaseName;
    for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
    {
        switch (testCase)
        {
            case 0:
            {
                testCaseName = "fmadd";
                Rpp32f *mulTensor = reinterpret_cast<Rpp32f *>(pinnedMemArgs);
                Rpp32f *addTensor = mulTensor + batchSize;

                for (int i = 0; i < batchSize; i++)
                {
                    mulTensor[i] = 80;
                    addTensor[i] = 5;
                }

                startWallTime = omp_get_wtime();
                rppt_fmadd_scalar_gpu(inputF32, descriptorPtr3D, outputF32, descriptorPtr3D, mulTensor, addTensor, roiGenericSrcPtr, roiTypeSrc, handle);
                break;
            }
            case 1:
            {
                testCaseName = "slice";
                startWallTime = omp_get_wtime();
                rppt_slice_gpu(inputF32, descriptorPtr3D, outputF32, descriptorPtr3D, roiGenericSrcPtr, roiTypeSrc, handle);
                break;
            }
            case 2:
            {
                testCaseName = "flip";
                Rpp32u horizontalTensor[batchSize];
                Rpp32u verticalTensor[batchSize];
                Rpp32u depthTensor[batchSize];

                for (int i = 0; i < batchSize; i++)
                {
                    horizontalTensor[i] = 1;
                    verticalTensor[i] = 0;
                    depthTensor[i] = 0;
                }

                startWallTime = omp_get_wtime();
                rppt_flip_voxel_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, horizontalTensor, verticalTensor, depthTensor, roiGenericSrcPtr, roiTypeSrc, handle);
                break;
            }
            case 3:
            {
                testCaseName = "add";
                Rpp32f addTensor[batchSize];

                for (int i = 0; i < batchSize; i++)
                    addTensor[i] = 40;

                startWallTime = omp_get_wtime();
                rppt_add_scalar_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, addTensor, roiGenericSrcPtr, roiTypeSrc, handle);
                break;
            }
            case 4:
            {
                testCaseName = "subtract";
                Rpp32f subtractTensor[batchSize];

                for (int i = 0; i < batchSize; i++)
                    subtractTensor[i] = -40;

                startWallTime = omp_get_wtime();
                rppt_subtract_scalar_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, subtractTensor, roiGenericSrcPtr, roiTypeSrc, handle);
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

        // Copy output buffer to host
        hipMemcpy(outputF32, d_outputF32, oBufferSizeInBytes, hipMemcpyDeviceToHost);
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
        if(DEBUG_MODE)
        {
            std::ofstream refFile;
            refFile.open("nifti_single.csv");
            for (int i = 0; i < oBufferSize; i++)
                refFile << static_cast<int>(*(outputF32 + i)) << ",";
            refFile.close();
        }

        if(qaFlag)
        {
            compare_output(outputF32, oBufferSize, testCaseName, layoutType);
        }

        for(int batchCount = 0; batchCount < batchSize; batchCount++)
        {
            Rpp32f *outputTemp = outputF32 + batchCount * descriptorPtr3D->strides[0];
            for(int i = 0; i < numChannels; i++) // temporary changes to process pln3
            {
                int xyFrameSize = niftiHeader[batchCount].dim[1] * niftiHeader[batchCount].dim[2];
                int xyFrameSizeROI = roiGenericSrcPtr[batchCount].xyzwhdROI.roiWidth * roiGenericSrcPtr[batchCount].xyzwhdROI.roiHeight;

                uint dataSize = niftiHeader[batchCount].dim[1] * niftiHeader[batchCount].dim[2] * niftiHeader[batchCount].dim[3];
                uchar *niftiDataU8 = (uchar *) malloc(dataSize * sizeof(uchar));
                uchar *outputBufferOpenCV = (uchar *)calloc(xyFrameSizeROI, sizeof(uchar));

                // Convert RpptDataType::F32 strided buffer to default NIFTI_DATATYPE unstrided buffer
                Rpp64u increment;
                if (descriptorPtr3D->layout == RpptLayout::NCDHW)
                    increment = ((Rpp64u)descriptorPtr3D->strides[1] * (Rpp64u)descriptorPtr3D->dims[0]);
                else
                    increment = 1;
                convert_output_Rpp32f_to_niftitype_generic(outputTemp + i * increment, descriptorPtr3D, niftiDataArray[batchCount], &niftiHeader[batchCount]);
                NIFTI_DATATYPE min = niftiDataArray[batchCount][0];
                NIFTI_DATATYPE max = niftiDataArray[batchCount][0];
                for (int i = 0; i < dataSize; i++)
                {
                    min = std::min(min, niftiDataArray[batchCount][i]);
                    max = std::max(max, niftiDataArray[batchCount][i]);
                }
                Rpp32f multiplier = 255.0f / (max - min);
                for (int i = 0; i < dataSize; i++)
                    niftiDataU8[i] = (uchar)((niftiDataArray[batchCount][i] - min) * multiplier);

                uchar *niftiDataU8Temp = niftiDataU8;
                for (int zPlane = roiGenericSrcPtr[batchCount].xyzwhdROI.xyz.z; zPlane < roiGenericSrcPtr[batchCount].xyzwhdROI.xyz.z + roiGenericSrcPtr[batchCount].xyzwhdROI.roiDepth; zPlane++)
                {
                    write_image_from_nifti_opencv(niftiDataU8Temp, niftiHeader[batchCount].dim[1], (RpptRoiXyzwhd *)roiGenericSrcPtr, outputBufferOpenCV, zPlane, i, batchCount);
                    niftiDataU8Temp += xyFrameSize;
                }

                write_nifti_file(&niftiHeader[batchCount], niftiDataArray[batchCount], batchCount);

                if(i == 0)
                {
                    std::string command = "convert -delay 10 -loop 0 $(ls -v | grep jpg | grep chn_0_) niftiOutput_"+ std::to_string(batchCount) +"_chn" + std::to_string(i) + ".gif";
                    system(command.c_str());
                }
                if(i == 1)
                {
                    std::string command = "convert -delay 10 -loop 0 $(ls -v | grep jpg | grep chn_1_) niftiOutput_"+ std::to_string(batchCount) +"_chn" + std::to_string(i) + ".gif";
                    system(command.c_str());
                }
                if(i == 2)
                {
                    std::string command = "convert -delay 10 -loop 0 $(ls -v | grep jpg | grep chn_2_) niftiOutput_" + std::to_string(batchCount) +"_chn" + std::to_string(i) + ".gif";
                    system(command.c_str());
                }
                free(niftiDataU8);
                free(outputBufferOpenCV);
            }
        }
    }

    rppDestroyHost(handle);

    // Free memory
    free(niftiDataArray);
    free(inputF32);
    free(outputF32);
    hipHostFree(pinnedMemROI);
    hipHostFree(pinnedMemArgs);
    hipFree(d_inputF32);
    hipFree(d_outputF32);

    return(0);
}