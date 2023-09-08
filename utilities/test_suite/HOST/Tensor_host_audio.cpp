#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include "rpp.h"
#include "../rpp_test_suite_common.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half/half.hpp>
#include <fstream>
#include <iomanip>
#include <vector>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

// Include this header file to use functions from libsndfile
#include <sndfile.h>

using namespace std;
using half_float::half;

typedef half Rpp16f;

#define DEBUG_MODE 0

void verify_output(Rpp32f *dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstDims, string testCase, vector<string> audioNames, string dst)
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "HOST/build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "REFERENCE_OUTPUTS_AUDIO/";
    std::cerr<<"\n "<<ref_path;
    int file_match = 0;
    for (int batchcount = 0; batchcount < dstDescPtr->n; batchcount++)
    {
        string current_file_name = audioNames[batchcount];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + testCase + "/" + testCase + "_ref_" + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);
        if(!ref_file.is_open())
        {
            cerr<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;
            break;
        }
        int matched_indices = 0;
        Rpp32f ref_val, out_val;
        Rpp32f *dstPtrCurrent = dstPtr + batchcount * dstDescPtr->strides.nStride;
        Rpp32f *dstPtrRow = dstPtrCurrent;
        for(int i = 0; i < dstDims[batchcount].height; i++)
        {
            Rpp32f *dstPtrTemp = dstPtrRow;
            for(int j = 0; j < dstDims[batchcount].width; j++)
            {
                ref_file>>ref_val;
                out_val = dstPtrTemp[j];
                bool invalid_comparision = ((out_val == 0.0f) && (ref_val != 0.0f));
                if(!invalid_comparision && abs(out_val - ref_val) < 1e-20)
                    matched_indices += 1;
            }
            dstPtrRow += dstDescPtr->strides.hStride;
        }
        ref_file.close();
        if(matched_indices == (dstDims[batchcount].width * dstDims[batchcount].height) && matched_indices !=0)
            file_match++;
    }
    std::string status = testCase + ": ";
    std::cerr<<std::endl<<"Results for Test case: "<<testCase<<std::endl;
    if(file_match == dstDescPtr->n)
    {
        std::cerr<<"PASSED!"<<std::endl;
        status += "PASSED";
    }
    else
    {
        std::cerr<<"FAILED! "<<file_match<<"/"<<dstDescPtr->n<<" outputs are matching with reference outputs"<<std::endl;
        status += "FAILED";
    }
    std::string qaResultsPath = dst + "/QA_results.txt";
    std:: ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open())
    {
        qaResults << status << std::endl;
        qaResults.close();
    }
}

void verify_non_silent_region_detection(float *detectedIndex, float *detectionLength, string testCase, int bs, vector<string> audioNames, string dst)
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "HOST/build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "REFERENCE_OUTPUTS_AUDIO/";
    int file_match = 0;
    for (int i = 0; i < bs; i++)
    {
        string current_file_name = audioNames[i];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + testCase + "/" + testCase + "_ref_" + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);
        if(!ref_file.is_open())
        {
            cerr<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;
            break;
        }

        Rpp32s ref_index, ref_length;
        Rpp32s out_index, out_length;
        ref_file>>ref_index;
        ref_file>>ref_length;
        out_index = detectedIndex[i];
        out_length = detectionLength[i];

        if((out_index == ref_index) && (out_length == ref_length))
            file_match += 1;
        ref_file.close();
    }
    std::cerr<<std::endl<<"Results for Test case: "<<testCase<<std::endl;
    if(file_match == bs)
        std::cerr<<"PASSED!"<<std::endl;
    else
        std::cerr<<"FAILED! "<<file_match<<"/"<<bs<<" outputs are matching with reference outputs"<<std::endl;
}

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 3;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_host_audio <src folder> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:3>\n");
        return -1;
    }

    char *src = argv[1];
    int ip_bitDepth = atoi(argv[2]);
    int testCase = atoi(argv[3]);
    int testType = atoi(argv[4]);
    int numRuns = atoi(argv[5]);
    int batchSize = atoi(argv[6]);
    int qaFlag = atoi(argv[7]);
    char *dst = argv[8];

    // Set case names
    string funcName = audioAugmentationMap[testCase];
    if (funcName.empty())
    {
        if (testType == 0)
            printf("\ncase %d is not supported\n", testCase);

        return -1;
    }

    // Initialize tensor descriptors
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    // Set src/dst data types in tensor descriptors
    if (ip_bitDepth == 2)
    {
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    }

    // Other initializations
    int missingFuncFlag = 0;
    int i = 0, j = 0, fileCnt = 0;
    int maxChannels = 0;
    int maxSrcWidth = 0, maxSrcHeight = 0;
    int maxDstWidth = 0, maxDstHeight = 0;
    unsigned long long count = 0;
    unsigned long long iBufferSize = 0;
    unsigned long long oBufferSize = 0;
    static int noOfAudioFiles = 0;

    // String ops on function name
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");

    string func = funcName;
    std::cout << "\nRunning " << func;

    // Get number of audio files
    vector<string> audioNames;
    vector<string> audioFilePath;

    search_files_recursive(src, audioNames, audioFilePath, ".wav");
    noOfAudioFiles = audioNames.size();

    if(noOfAudioFiles < batchSize || ((noOfAudioFiles % batchSize) != 0))
    {
        replicate_last_file_to_fill_batch(audioFilePath[noOfAudioFiles - 1], audioFilePath, audioNames, audioNames[noOfAudioFiles - 1], noOfAudioFiles, batchSize);
        noOfAudioFiles = audioNames.size();
    }

    // Initialize the AudioPatch for source
    Rpp32s *srcLengthTensor = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));
    Rpp32s *channelsTensor = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));
    RpptImagePatch *srcDims = (RpptImagePatch *) calloc(noOfAudioFiles, sizeof(RpptImagePatch));
    RpptImagePatch *dstDims = (RpptImagePatch *) calloc(noOfAudioFiles, sizeof(RpptImagePatch));

    // Set Height as 1 for src, dst
    maxSrcHeight = 1;
    maxDstHeight = 1;

    for(int cnt = 0; cnt < noOfAudioFiles ; cnt++)
    {
        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;

        //The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (!(infile = sf_open (audioFilePath[cnt].c_str(), SFM_READ, &sfinfo)))
        {
            sf_close (infile);
            continue;
        }

        srcLengthTensor[count] = sfinfo.frames;
        channelsTensor[count] = sfinfo.channels;

        srcDims[count].width = sfinfo.frames;
        dstDims[count].width = sfinfo.frames;
        srcDims[count].height = 1;
        dstDims[count].height = 1;

        maxSrcWidth = std::max(maxSrcWidth, srcLengthTensor[count]);
        maxDstWidth = std::max(maxDstWidth, srcLengthTensor[count]);
        maxChannels = std::max(maxChannels, channelsTensor[count]);

        // Close input
        sf_close (infile);
        count++;

    }


    // Set numDims, offset, n/c/h/w values for src/dst
    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    srcDescPtr->offsetInBytes = 0;
    dstDescPtr->offsetInBytes = 0;

    srcDescPtr->n = batchSize;
    dstDescPtr->n = batchSize;

    srcDescPtr->h = maxSrcHeight;
    dstDescPtr->h = maxDstHeight;

    srcDescPtr->w = maxSrcWidth;
    dstDescPtr->w = maxDstWidth;

    srcDescPtr->c = maxChannels;
    if(testCase == 3)
        dstDescPtr->c = 1;
    else
        dstDescPtr->c = maxChannels;

    // Optionally set w stride as a multiple of 8 for src
    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
    srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
    srcDescPtr->strides.wStride = srcDescPtr->c;
    srcDescPtr->strides.cStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
    dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
    dstDescPtr->strides.wStride = dstDescPtr->c;
    dstDescPtr->strides.cStride = 1;

    // Set buffer sizes for src/dst
    iBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)srcDescPtr->n;
    oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;

    // Initialize host buffers for input & output
    Rpp32f *inputf32 = (Rpp32f *)calloc(iBufferSize, sizeof(Rpp32f));
    Rpp32f *outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, srcDescPtr->n, 8);
    int noOfIterations = (int)audioNames.size() / batchSize;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double cpuTime, wallTime;
    string testCaseName;
    for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
    {
        for(int iterCount = 0; iterCount < noOfIterations; iterCount++)
        {
            for(int cnt = 0; cnt < batchSize; cnt++)
            {
                Rpp32f *input_temp_f32;
                input_temp_f32 = inputf32 + (cnt * srcDescPtr->strides.nStride);

                SNDFILE	*infile;
                SF_INFO sfinfo;
                int	readcount;

                // The SF_INFO struct must be initialized before using it
                memset (&sfinfo, 0, sizeof (sfinfo));
                if (!(infile = sf_open (audioFilePath[fileCnt].c_str(), SFM_READ, &sfinfo)))
                {
                    sf_close (infile);
                    continue;
                }

                int bufferLength = sfinfo.frames * sfinfo.channels;
                if(ip_bitDepth == 2)
                {
                    readcount = (int) sf_read_float (infile, input_temp_f32, bufferLength);
                    if(readcount != bufferLength)
                        std::cerr<<"F32 Unable to read audio file completely"<<std::endl;
                }
                fileCnt++;
                count++;

                // Close input
                sf_close (infile);
            }
            clock_t startCpuTime, endCpuTime;
            double startWallTime, endWallTime;
            switch (testCase)
            {
                case 0:
                {
                    testCaseName = "non_silent_region_detection";
                    Rpp32f detectedIndex[batchSize];
                    Rpp32f detectionLength[batchSize];
                    Rpp32f cutOffDB = -60.0;
                    Rpp32s windowLength = 2048;
                    Rpp32f referencePower = 0.0f;
                    Rpp32s resetInterval = 8192;

                    startWallTime = omp_get_wtime();
                    startCpuTime= clock();
                    if (ip_bitDepth == 2)
                    {
                        rppt_non_silent_region_detection_host(inputf32, srcDescPtr, srcLengthTensor, detectedIndex, detectionLength, cutOffDB, windowLength, referencePower, resetInterval, handle);
                    }
                    else
                        missingFuncFlag = 1;

                    if((testType == 0 || qaFlag ==1) && batchSize == 8)
                        verify_non_silent_region_detection(detectedIndex, detectionLength, testCaseName, batchSize, audioNames, dst);

                    break;
                }
                default:
                {
                    missingFuncFlag = 1;
                    break;
                }
            }
            endCpuTime = clock();
            endWallTime = omp_get_wtime();
            cpuTime = ((double)(endCpuTime - startCpuTime)) / CLOCKS_PER_SEC;
            wallTime = endWallTime - startWallTime;
            if (missingFuncFlag == 1)
            {
                printf("\nThe functionality %s doesn't yet exist in RPP\n", func.c_str());
                return -1;
            }

            maxWallTime = std::max(maxWallTime, wallTime);
            minWallTime = std::min(minWallTime, wallTime);
            avgWallTime += wallTime;
            cpuTime *= 1000;
            wallTime *= 1000;

            if (testType == 0)
            {
                if ((qaFlag == 1 || testType == 0) && ((testCase == 3 && batchSize == 1) || (testCase != 3 && batchSize == 8) && (testCase !=0)))
                    verify_output(outputf32, dstDescPtr, dstDims, testCaseName, audioNames, dst);

                cout <<"\n\n";
                cout <<"CPU Backend Clock Time: "<< cpuTime <<" ms/batch"<< endl;
                cout <<"CPU Backend Wall Time: "<< wallTime <<" ms/batch"<< endl;

                // If DEBUG_MODE is set to 1 dump the outputs to csv files for debugging
                if(DEBUG_MODE && iterCount == 0 && testCase != 0)
                {
                    std::ofstream refFile;
                    refFile.open(func + ".csv");
                    for (int i = 0; i < oBufferSize; i++)
                        refFile << static_cast<int>(*(outputf32 + i)) << ",";
                    refFile.close();
                }
            }
        }
        fileCnt = 0;
    }

    rppDestroyHost(handle);

    if(testType == 1)
    {
        // Display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns * noOfIterations);
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }

    cout<<endl;

    // Free memory
    free(srcLengthTensor);
    free(channelsTensor);
    free(srcDims);
    free(dstDims);
    free(inputf32);
    free(outputf32);

    return 0;
}