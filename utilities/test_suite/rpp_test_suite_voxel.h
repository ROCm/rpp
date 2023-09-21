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
#define CUTOFF 1
#define TOLERANCE 0.01
#define MAX_IMAGE_DUMP 100
#define MAX_BATCH_SIZE 512

std::map<int, string> augmentationMap =
{
    {0, "fmadd"},
    {1, "slice"},
};

void replicate_last_file_to_fill_batch(const string& lastFilePath, vector<string>& filePathVector, vector<string>& fileNamesVector, const string& lastFileName, int noOfFiles, int batchCount)
{
    int remainingFiles = batchCount - (noOfFiles % batchCount);
    std::string filePath = lastFilePath;
    std::string fileName = lastFileName;
    if (noOfFiles > 0 && ( noOfFiles < batchCount || noOfFiles % batchCount != 0 ))
    {
        for (int i = 0; i < remainingFiles; i++)
        {
            filePathVector.push_back(filePath);
            fileNamesVector.push_back(fileName);
        }
    }
}

// Opens a folder and recursively search for .nii files
void open_folder(const string& folderPath, vector<string>& niiFileNames, vector<string>& niiFilePath)
{
    auto src_dir = opendir(folderPath.c_str());
    struct dirent* entity;
    std::string fileName = " ";

    if (src_dir == nullptr)
        std::cout << "\n ERROR: Failed opening the directory at " <<folderPath;

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
        std::cout << "\n Did not load any file from " << folderPath;

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
        std::cout << "ERROR: Failed opening the directory at "<< folder_path << std::endl;
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
static int read_nifti_header_file(char* const header_file, nifti_1_header &niftiHeader)
{
    nifti_1_header hdr;

    // open and read header
    FILE *fp = fopen(header_file,"r");
    if (fp == NULL)
    {
        fprintf(stdout, "\nError opening header file %s\n", header_file);
        exit(1);
    }
    int ret = fread(&hdr, MIN_HEADER_SIZE, 1, fp);
    if (ret != 1)
    {
        fprintf(stdout, "\nError reading header file %s\n", header_file);
        exit(1);
    }
    fclose(fp);

    // print header information
    fprintf(stdout, "\n%s header information:", header_file);
    fprintf(stdout, "\nNIFTI1 XYZT dimensions: %d %d %d %d", hdr.dim[1], hdr.dim[2], hdr.dim[3], hdr.dim[4]);
    fprintf(stdout, "\nNIFTI1 Datatype code and bits/pixel: %d %d", hdr.datatype, hdr.bitpix);
    fprintf(stdout, "\nNIFTI1 Scaling slope and intercept: %.6f %.6f", hdr.scl_slope, hdr.scl_inter);
    fprintf(stdout, "\nNIFTI1 Byte offset to data in datafile: %ld", (long)(hdr.vox_offset));
    fprintf(stdout, "\n");

    niftiHeader = hdr;

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
        fprintf(stdout, "\nError opening data file %s\n", data_file);
        exit(1);
    }
    ret = fseek(fp, (long)(hdr.vox_offset), SEEK_SET);
    if (ret != 0)
    {
        fprintf(stdout, "\nError doing fseek() to %ld in data file %s\n", (long)(hdr.vox_offset), data_file);
        exit(1);
    }

    ret = fread(data, sizeof(NIFTI_DATATYPE), hdr.dim[1] * hdr.dim[2] * hdr.dim[3], fp);
    if (ret != hdr.dim[1] * hdr.dim[2] * hdr.dim[3])
    {
        fprintf(stdout, "\nError reading volume 1 from %s (%d)\n", data_file, ret);
        exit(1);
    }
    fclose(fp);
}

inline void write_nifti_file(nifti_1_header *niftiHeader, NIFTI_DATATYPE *niftiData, int batchCount, int chn, string dstPath, string func)
{
    nifti_1_header hdr = *niftiHeader;
    FILE *fp;
    int ret, i;

    // write first hdr.vox_offset bytes of header
    string niiOutputString = dstPath + "/" + std::to_string(batchCount) + "_" + func + "_chn_" + std::to_string(chn)+"_nifti_output.nii";
    const char *niiOutputFile = niiOutputString.c_str();
    fp = fopen(niiOutputFile,"w");
    if (fp == NULL)
    {
        fprintf(stdout, "\nError opening header file %s for write\n",niiOutputFile);
        exit(1);
    }
    ret = fwrite(&hdr, hdr.vox_offset, 1, fp);
    if (ret != 1)
    {
        fprintf(stdout, "\nError writing header file %s\n",niiOutputFile);
        exit(1);
    }

    if (ret != 1)
    {
        fprintf(stdout, "\nError writing header file extension pad %s\n",niiOutputFile);
        exit(1);
    }

    ret = fwrite(niftiData, (size_t)(hdr.bitpix/8), hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4], fp);
    if (ret != hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4])
    {
        fprintf(stdout, "\nError writing data to %s\n",niiOutputFile);
        exit(1);
    }

    fclose(fp);
}

inline void write_image_from_nifti_opencv(uchar *niftiDataXYFrameU8, int niftiHeaderImageWidth, RpptRoiXyzwhd *roiGenericSrcPtr, uchar *outputBufferOpenCV, int zPlane, int Channel, int batchCount, string dst_path, string func, int index)
{
    static int imageCount = 0;
    if (imageCount > MAX_IMAGE_DUMP)
        exit(0);
    uchar *outputBufferOpenCVRow = outputBufferOpenCV;
    uchar *niftiDataXYFrameU8Row = niftiDataXYFrameU8;
    for(int i = 0; i < roiGenericSrcPtr[batchCount].roiHeight; i++)
    {
        memcpy(outputBufferOpenCVRow, niftiDataXYFrameU8Row, roiGenericSrcPtr[batchCount].roiWidth);
        outputBufferOpenCVRow += roiGenericSrcPtr[batchCount].roiWidth;
        niftiDataXYFrameU8Row += niftiHeaderImageWidth;
    }
    cv::Mat matOutputImage = cv::Mat(roiGenericSrcPtr[batchCount].roiHeight, roiGenericSrcPtr[batchCount].roiWidth, CV_8UC1, outputBufferOpenCV);
    string fileName = dst_path + "/" + func +"_nifti_" + std::to_string(index) + "_zPlane_chn_"+ std::to_string(Channel) + "_" + std::to_string(zPlane) + ".jpg";
    cv::imwrite(fileName, matOutputImage);
    imageCount++;
}

// Convert default NIFTI_DATATYPE unstrided buffer to RpptDataType::F32 strided buffer
template<typename T>
inline void convert_input_niftitype_to_Rpp32f_generic(T **niftyInput, nifti_1_header headerData[], Rpp32f *inputF32, RpptGenericDescPtr descriptorPtr3D)
{
    bool replicateToAllChannels = false;
    Rpp32u depthStride, rowStride, channelStride, channelIncrement;
    if (descriptorPtr3D->layout == RpptLayout::NCDHW)
    {
        depthStride = descriptorPtr3D->strides[2];
        rowStride = descriptorPtr3D->strides[3];
        channelStride = descriptorPtr3D->strides[1];
        channelIncrement = 1;
        if(descriptorPtr3D->dims[1] == 3)
            replicateToAllChannels = true;                            //temporary chnage to replicate the data for pln3 using pln1 data
    }
    else if (descriptorPtr3D->layout == RpptLayout::NDHWC)
    {
        depthStride = descriptorPtr3D->strides[1];
        rowStride = descriptorPtr3D->strides[2];
        channelStride = 1;
        channelIncrement = 3;
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

// read nifti data file
inline void read_nifti_data(vector<string>::const_iterator dataFilePathStart, vector<string>::const_iterator dataFilePathEnd, NIFTI_DATATYPE** niftiDataArray, nifti_1_header* niftiHeader)
{
    int i = 0;
    for ( ; dataFilePathStart != dataFilePathEnd; ++dataFilePathStart, i++)
    {
        const string& dataFilePath = *dataFilePathStart;
        uint dataSize = niftiHeader[i].dim[1] * niftiHeader[i].dim[2] * niftiHeader[i].dim[3];
        uint dataSizeInBytes = dataSize * sizeof(NIFTI_DATATYPE);
        niftiDataArray[i] = (NIFTI_DATATYPE *) calloc(dataSizeInBytes, 1);
        if (niftiDataArray[i] == NULL)
        {
            fprintf(stdout, "\nError allocating data buffer for %s\n", dataFilePath.c_str());
            exit(1);
        }
        // read nifti data file
        read_nifti_data_file((char *)dataFilePath.c_str(), &niftiHeader[i], niftiDataArray[i]);
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

// compares the output of PKD3-PKD3 variants
void compare_outputs_pkd(Rpp32f* output, Rpp32f* refOutput, int &fileMatch, RpptGenericDescPtr descriptorPtr3D, RpptRoiXyzwhd *roiGenericSrcPtr)
{
    Rpp32f *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef, *depthTemp, *depthTempRef;
    for(int Cnt = 0; Cnt < descriptorPtr3D->dims[0]; Cnt++)
    {
        outputTemp = output + Cnt * descriptorPtr3D->strides[0];
        outputTempRef = refOutput + Cnt * descriptorPtr3D->strides[0];
        int height = roiGenericSrcPtr[Cnt].roiHeight;
        int width = roiGenericSrcPtr[Cnt].roiWidth * descriptorPtr3D->dims[4];
        int depth = roiGenericSrcPtr[Cnt].roiDepth;
        int depthStride = descriptorPtr3D->strides[1];
        int rowStride = descriptorPtr3D->strides[2];
        int matched_idx = 0;
        for(int d = 0; d < depth; d++)
        {
            depthTemp = outputTemp + d * depthStride;
            depthTempRef = outputTempRef + d * depthStride;
            for(int i = 0; i < height; i++)
            {
                rowTemp = depthTemp + i * rowStride;
                rowTempRef = depthTempRef + i * rowStride;
                for(int j = 0; j < width; j++)
                {
                    outVal = rowTemp + j;
                    outRefVal = rowTempRef + j;
                    int diff = abs(*outVal - *outRefVal);
                    if(diff <= CUTOFF)
                        matched_idx++;
                }
            }
        }
        if(matched_idx >= (1 - TOLERANCE) * (height * width * depth) && matched_idx !=0)
            fileMatch++;
    }
}

// compares the output of PLN3-PLN3 and PLN1-PLN1 variants
void compare_outputs_pln(Rpp32f* output, Rpp32f* refOutput, int &fileMatch, RpptGenericDescPtr descriptorPtr3D, RpptRoiXyzwhd *roiGenericSrcPtr)
{
    Rpp32f *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef, *outputTempChn, *outputTempRefChn, *depthTemp, *depthTempRef;
    for(int Cnt = 0; Cnt < descriptorPtr3D->dims[0]; Cnt++)
    {
        outputTemp = output + Cnt * descriptorPtr3D->strides[0];
        outputTempRef = refOutput + Cnt * descriptorPtr3D->strides[0];
        int height = roiGenericSrcPtr[Cnt].roiHeight;
        int width = roiGenericSrcPtr[Cnt].roiWidth;
        int depth = roiGenericSrcPtr[Cnt].roiDepth;
        int depthStride = descriptorPtr3D->strides[2];
        int rowStride = descriptorPtr3D->strides[3];
        int channelStride = descriptorPtr3D->strides[1];
        int matched_idx = 0;

        for(int c = 0; c < descriptorPtr3D->dims[1]; c++)
        {
            outputTempChn = outputTemp + c * channelStride;
            outputTempRefChn = outputTempRef + c * channelStride;
            for(int d = 0; d < depth; d++)
            {
                depthTemp = outputTempChn + d * depthStride;
                depthTempRef = outputTempRefChn + d * depthStride;
                for(int i = 0; i < height; i++)
                {
                    rowTemp = depthTemp + i * rowStride;
                    rowTempRef = depthTempRef + i * rowStride;
                    for(int j = 0; j < width; j++)
                    {
                        outVal = rowTemp + j;
                        outRefVal = rowTempRef + j ;
                        int diff = abs(*outVal - *outRefVal);
                        if(diff <= CUTOFF)
                            matched_idx++;
                    }
                }
            }
        }
        if(matched_idx >= (1 - TOLERANCE) * (height * width * descriptorPtr3D->dims[1] * depth) && matched_idx !=0)
            fileMatch++;
    }
}

inline void compare_output(Rpp32f* output, Rpp64u oBufferSize, string func, int layoutType, RpptGenericDescPtr descriptorPtr3D, RpptRoiXyzwhd *roiGenericSrcPtr, string dst)
{
    string refPath = get_current_dir_name();
    string pattern = "/build";
    remove_substring(refPath, pattern);
    string csvName = "";
    if(layoutType == 0)
        csvName = func + "_nifti_output_pkd3.csv";
    else if(layoutType == 1)
        csvName = func + "_nifti_output_pln3.csv";
    else
        csvName = func + "_nifti_output_pln1.csv";

    string refFile = refPath + "/../REFERENCE_OUTPUT_VOXEL/"+ func + "/" + csvName;

    ifstream file(refFile);
    float *refOutput;
    refOutput = (float *)malloc(oBufferSize * sizeof(Rpp32f));
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
                refOutput[index] = stof(word);
                index++;
            }
        }
    }
    else
    {
        cout<<"Could not open the reference output. Please check the path specified\n";
        return;
    }

    int fileMatch = 0;
    if(layoutType == 0)
        compare_outputs_pkd(output, refOutput, fileMatch, descriptorPtr3D, roiGenericSrcPtr);
    else
        compare_outputs_pln(output, refOutput, fileMatch, descriptorPtr3D, roiGenericSrcPtr);

    std::cout << std::endl << "Results for " << func << " :" << std::endl;
    std::string status = func + ": ";
    if(fileMatch == descriptorPtr3D->dims[0])
    {
        std::cout << "PASSED!" << std::endl;
        status += "PASSED";
    }
    else
    {
        std::cout << "FAILED! " << fileMatch << "/" << descriptorPtr3D->dims[0] << " outputs are matching with reference outputs" << std::endl;
        status += "FAILED";
    }

    // Append the QA results to file
    std::string qaResultsPath = dst + "/QA_results.txt";
    std:: ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open())
    {
        qaResults << status << std::endl;
        qaResults.close();
    }
}