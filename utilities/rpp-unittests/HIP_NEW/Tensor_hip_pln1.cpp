#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "/opt/rocm/rpp/include/rpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half.hpp>
#include <fstream>
#include "helpers/testSuite_helper.hpp"

using namespace cv;
using namespace std;
using half_float::half;

typedef half Rpp16f;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))
#define RPPMAX2(a,b) ((a > b) ? a : b)
#define RPPMIN2(a,b) ((a < b) ? a : b)

int main(int argc, char **argv)
{
    // Handle inputs

    const int MIN_ARG_COUNT = 8;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_host_pln1 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:81> <verbosity = 0/1>\n");
        return -1;
    }
    if (atoi(argv[5]) != 0)
    {
        printf("\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = 0\n");
        return -1;
    }

    if (atoi(argv[7]) == 1)
    {
        printf("\nInputs for this test case are:");
        printf("\nsrc1 = %s", argv[1]);
        printf("\nsrc2 = %s", argv[2]);
        printf("\ndst = %s", argv[3]);
        printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[4]);
        printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[5]);
        printf("\ncase number (1:7) = %s", argv[6]);
    }

    char *src = argv[1];
    char *src_second = argv[2];
    char *dst = argv[3];
    int ip_bitDepth = atoi(argv[4]);
    unsigned int outputFormatToggle = atoi(argv[5]);
    int test_case = atoi(argv[6]);

    int ip_channel = 1;

    // Set case names

    char funcType[1000] = {"Tensor_HOST_PLN1_toPLN1"};

    char funcName[1000];
    switch (test_case)
    {
    case 0:
        strcpy(funcName, "brightness");
        outputFormatToggle = 0;
        break;
    }

    // Initialize tensor descriptors

    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    // Set src/dst layouts in tensor descriptors

    srcDescPtr->layout = RpptLayout::NCHW;
    dstDescPtr->layout = RpptLayout::NCHW;

    // Set src/dst data types in tensor descriptors

    if (ip_bitDepth == 0)
    {
        strcat(funcName, "_u8_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;
    }
    else if (ip_bitDepth == 1)
    {
        strcat(funcName, "_f16_");
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 2)
    {
        strcat(funcName, "_f32_");
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 3)
    {
        strcat(funcName, "_u8_f16_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 4)
    {
        strcat(funcName, "_u8_f32_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 5)
    {
        strcat(funcName, "_i8_");
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;
    }
    else if (ip_bitDepth == 6)
    {
        strcat(funcName, "_u8_i8_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::I8;
    }

    // Other initializations

    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxHeight = 0, maxWidth = 0;
    int maxDstHeight = 0, maxDstWidth = 0;
    unsigned long long count = 0;
    unsigned long long ioBufferSize = 0;
    unsigned long long oBufferSize = 0;
    static int noOfImages = 0;
    Mat image, image_second;

    // String ops on function name

    char func[1000];
    strcpy(func, funcName);
    strcat(func, funcType);
    printf("\nRunning %s...", func);

    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");

    char src1_second[1000];
    strcpy(src1_second, src_second);
    strcat(src1_second, "/");

    strcat(funcName, funcType);
    strcat(dst, "/");
    strcat(dst, funcName);

    // Get number of images

    struct dirent *de;
    DIR *dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfImages += 1;
    }
    closedir(dr);

    // Initialize ROI tensors for src/dst

    RpptROI *roiTensorPtrSrc = (RpptROI *) calloc(noOfImages, sizeof(RpptROI));
    RpptROI *roiTensorPtrDst = (RpptROI *) calloc(noOfImages, sizeof(RpptROI));

    RpptROI *d_roiTensorPtrSrc, *d_roiTensorPtrDst;
    hipMalloc(&d_roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    hipMalloc(&d_roiTensorPtrDst, noOfImages * sizeof(RpptROI));

    // Set ROI tensors types for src/dst

    RpptRoiType roiTypeSrc, roiTypeDst;
    roiTypeSrc = RpptRoiType::XYWH;
    roiTypeDst = RpptRoiType::XYWH;

    // Set maxHeight, maxWidth and ROIs for src/dst

    const int images = noOfImages;
    char imageNames[images][1000];

    DIR *dr1 = opendir(src);
    while ((de = readdir(dr1)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        strcpy(imageNames[count], de->d_name);
        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, imageNames[count]);

        image = imread(temp, 0);

        roiTensorPtrSrc[count].xywhROI.xy.x = 0;
        roiTensorPtrSrc[count].xywhROI.xy.y = 0;
        roiTensorPtrSrc[count].xywhROI.roiWidth = image.cols;
        roiTensorPtrSrc[count].xywhROI.roiHeight = image.rows;

        roiTensorPtrDst[count].xywhROI.xy.x = 0;
        roiTensorPtrDst[count].xywhROI.xy.y = 0;
        roiTensorPtrDst[count].xywhROI.roiWidth = image.cols;
        roiTensorPtrDst[count].xywhROI.roiHeight = image.rows;

        maxHeight = RPPMAX2(maxHeight, roiTensorPtrSrc[count].xywhROI.roiHeight);
        maxWidth = RPPMAX2(maxWidth, roiTensorPtrSrc[count].xywhROI.roiWidth);
        maxDstHeight = RPPMAX2(maxDstHeight, roiTensorPtrDst[count].xywhROI.roiHeight);
        maxDstWidth = RPPMAX2(maxDstWidth, roiTensorPtrDst[count].xywhROI.roiWidth);

        count++;
    }
    closedir(dr1);

    // Set numDims, offset, n/c/h/w values, n/c/h/w strides for src/dst

    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    srcDescPtr->offset = 0;
    dstDescPtr->offset = 0;

    srcDescPtr->n = noOfImages;
    srcDescPtr->c = ip_channel;
    srcDescPtr->h = maxHeight;
    srcDescPtr->w = maxWidth;

    dstDescPtr->n = noOfImages;
    dstDescPtr->c = ip_channel;
    dstDescPtr->h = maxDstHeight;
    dstDescPtr->w = maxDstWidth;

    // Optionally set w stride as a multiple of 8 for src/dst

    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

    srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.hStride = srcDescPtr->w;
    srcDescPtr->strides.wStride = 1;

    if (dstDescPtr->layout == RpptLayout::NHWC)
    {
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = ip_channel * dstDescPtr->w;
        dstDescPtr->strides.wStride = ip_channel;
        dstDescPtr->strides.cStride = 1;
    }
    else if (dstDescPtr->layout == RpptLayout::NCHW)
    {
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;
    }

    // Set buffer sizes for src/dst

    ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
    oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

    // Initialize host buffers for src/dst

    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(oBufferSize, sizeof(Rpp8u));

    // Set 8u host buffers for src/dst

    DIR *dr2 = opendir(src);
    DIR *dr2_second = opendir(src_second);
    count = 0;
    i = 0;

    Rpp32u elementsInRowMax = srcDescPtr->w * ip_channel;

    while ((de = readdir(dr2)) != NULL)
    {
        Rpp8u *input_temp, *input_second_temp;
        input_temp = input + (i * srcDescPtr->strides.nStride);
        input_second_temp = input_second + (i * srcDescPtr->strides.nStride);

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;

        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, de->d_name);

        char temp_second[1000];
        strcpy(temp_second, src1_second);
        strcat(temp_second, de->d_name);

        image = imread(temp, 0);
        image_second = imread(temp_second, 0);

        Rpp8u *ip_image = image.data;
        Rpp8u *ip_image_second = image_second.data;

        Rpp32u elementsInRow = roiTensorPtrSrc[i].xywhROI.roiWidth * ip_channel;

        for (j = 0; j < roiTensorPtrSrc[i].xywhROI.roiHeight; j++)
        {
            memcpy(input_temp, ip_image, elementsInRow * sizeof (Rpp8u));
            memcpy(input_second_temp, ip_image_second, elementsInRow * sizeof (Rpp8u));
            ip_image += elementsInRow;
            ip_image_second += elementsInRow;
            input_temp += elementsInRowMax;
            input_second_temp += elementsInRowMax;
        }
        i++;
        count += srcDescPtr->strides.nStride;
    }
    closedir(dr2);

    // Convert inputs to test various other bit depths and copy to hip buffers

    Rpp16f *inputf16, *inputf16_second, *outputf16;
    Rpp32f *inputf32, *inputf32_second, *outputf32;
    Rpp8s *inputi8, *inputi8_second, *outputi8;
    int *d_input, *d_input_second, *d_inputf16, *d_inputf16_second, *d_inputf32, *d_inputf32_second, *d_inputi8, *d_inputi8_second;
    int *d_output, *d_outputf16, *d_outputf32, *d_outputi8;

    if (ip_bitDepth == 0)
    {
        hipMalloc(&d_input, ioBufferSize * sizeof(Rpp8u));
        hipMalloc(&d_input_second, ioBufferSize * sizeof(Rpp8u));
        hipMalloc(&d_output, oBufferSize * sizeof(Rpp8u));
        hipMemcpy(d_input, input, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_input_second, input_second, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_output, output, oBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 1)
    {
        inputf16 = (Rpp16f *)calloc(ioBufferSize, sizeof(Rpp16f));
        inputf16_second = (Rpp16f *)calloc(ioBufferSize, sizeof(Rpp16f));
        outputf16 = (Rpp16f *)calloc(oBufferSize, sizeof(Rpp16f));

        Rpp8u *inputTemp, *input_secondTemp;
        Rpp16f *inputf16Temp, *inputf16_secondTemp;

        inputTemp = input;
        input_secondTemp = input_second;

        inputf16Temp = inputf16;
        inputf16_secondTemp = inputf16_second;

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf16Temp = ((Rpp16f)*inputTemp) / 255.0;
            *inputf16_secondTemp = ((Rpp16f)*input_secondTemp) / 255.0;
            inputTemp++;
            inputf16Temp++;
            input_secondTemp++;
            inputf16_secondTemp++;
        }

        hipMalloc(&d_inputf16, ioBufferSize * sizeof(Rpp16f));
        hipMalloc(&d_inputf16_second, ioBufferSize * sizeof(Rpp16f));
        hipMalloc(&d_outputf16, oBufferSize * sizeof(Rpp16f));
        hipMemcpy(d_inputf16, inputf16, ioBufferSize * sizeof(Rpp16f), hipMemcpyHostToDevice);
        hipMemcpy(d_inputf16_second, inputf16_second, ioBufferSize * sizeof(Rpp16f), hipMemcpyHostToDevice);
        hipMemcpy(d_outputf16, outputf16, oBufferSize * sizeof(Rpp16f), hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 2)
    {
        inputf32 = (Rpp32f *)calloc(ioBufferSize, sizeof(Rpp32f));
        inputf32_second = (Rpp32f *)calloc(ioBufferSize, sizeof(Rpp32f));
        outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));

        Rpp8u *inputTemp, *input_secondTemp;
        Rpp32f *inputf32Temp, *inputf32_secondTemp;

        inputTemp = input;
        input_secondTemp = input_second;

        inputf32Temp = inputf32;
        inputf32_secondTemp = inputf32_second;

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf32Temp = ((Rpp32f)*inputTemp) / 255.0;
            *inputf32_secondTemp = ((Rpp32f)*input_secondTemp) / 255.0;
            inputTemp++;
            inputf32Temp++;
            input_secondTemp++;
            inputf32_secondTemp++;
        }

        hipMalloc(&d_inputf32, ioBufferSize * sizeof(Rpp32f));
        hipMalloc(&d_inputf32_second, ioBufferSize * sizeof(Rpp32f));
        hipMalloc(&d_outputf32, oBufferSize * sizeof(Rpp32f));
        hipMemcpy(d_inputf32, inputf32, ioBufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice);
        hipMemcpy(d_inputf32_second, inputf32_second, ioBufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice);
        hipMemcpy(d_outputf32, outputf32, oBufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 3)
    {
        outputf16 = (Rpp16f *)calloc(oBufferSize, sizeof(Rpp16f));
        hipMalloc(&d_input, ioBufferSize * sizeof(Rpp8u));
        hipMalloc(&d_input_second, ioBufferSize * sizeof(Rpp8u));
        hipMalloc(&d_outputf16, oBufferSize * sizeof(Rpp16f));
        hipMemcpy(d_input, input, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_input_second, input_second, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_outputf16, outputf16, oBufferSize * sizeof(Rpp16f), hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 4)
    {
        outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));
        hipMalloc(&d_input, ioBufferSize * sizeof(Rpp8u));
        hipMalloc(&d_input_second, ioBufferSize * sizeof(Rpp8u));
        hipMalloc(&d_outputf32, oBufferSize * sizeof(Rpp32f));
        hipMemcpy(d_input, input, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_input_second, input_second, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_outputf32, outputf32, oBufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 5)
    {
        inputi8 = (Rpp8s *)calloc(ioBufferSize, sizeof(Rpp8s));
        inputi8_second = (Rpp8s *)calloc(ioBufferSize, sizeof(Rpp8s));
        outputi8 = (Rpp8s *)calloc(oBufferSize, sizeof(Rpp8s));

        Rpp8u *inputTemp, *input_secondTemp;
        Rpp8s *inputi8Temp, *inputi8_secondTemp;

        inputTemp = input;
        input_secondTemp = input_second;

        inputi8Temp = inputi8;
        inputi8_secondTemp = inputi8_second;

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputi8Temp = (Rpp8s) (((Rpp32s) *inputTemp) - 128);
            *inputi8_secondTemp = (Rpp8s) (((Rpp32s) *input_secondTemp) - 128);
            inputTemp++;
            inputi8Temp++;
            input_secondTemp++;
            inputi8_secondTemp++;
        }

        hipMalloc(&d_inputi8, ioBufferSize * sizeof(Rpp8s));
        hipMalloc(&d_inputi8_second, ioBufferSize * sizeof(Rpp8s));
        hipMalloc(&d_outputi8, oBufferSize * sizeof(Rpp8s));
        hipMemcpy(d_inputi8, inputi8, ioBufferSize * sizeof(Rpp8s), hipMemcpyHostToDevice);
        hipMemcpy(d_inputi8_second, inputi8_second, ioBufferSize * sizeof(Rpp8s), hipMemcpyHostToDevice);
        hipMemcpy(d_outputi8, outputi8, oBufferSize * sizeof(Rpp8s), hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 6)
    {
        outputi8 = (Rpp8s *)calloc(oBufferSize, sizeof(Rpp8s));
        hipMalloc(&d_input, ioBufferSize * sizeof(Rpp8u));
        hipMalloc(&d_input_second, ioBufferSize * sizeof(Rpp8u));
        hipMalloc(&d_outputi8, oBufferSize * sizeof(Rpp8s));
        hipMemcpy(d_input, input, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_input_second, input_second, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_outputi8, outputi8, oBufferSize * sizeof(Rpp8s), hipMemcpyHostToDevice);
    }

    // Run case-wise RPP API and measure time

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    clock_t start, end;
    double gpu_time_used;

    string test_case_name;

    switch (test_case)
    {
    case 0:
    {
        test_case_name = "brightness";

        Rpp32f alpha[images];
        Rpp32f beta[images];
        for (i = 0; i < images; i++)
        {
            alpha[i] = 1.75;
            beta[i] = 50;

            // xywhROI override sample
            // roiTensorPtrSrc[i].xywhROI.xy.x = 0;
            // roiTensorPtrSrc[i].xywhROI.xy.y = 0;
            // roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
            // roiTensorPtrSrc[i].xywhROI.roiHeight = 180;

            // ltrbROI override sample
            // roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
            // roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
            // roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
            // roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
        }

        // Change RpptRoiType for ltrbROI override sample
        // roiTypeSrc = RpptRoiType::LTRB;
        // roiTypeDst = RpptRoiType::LTRB;

        hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

        start = clock();

        if (ip_bitDepth == 0)
            rppt_brightness_gpu(d_input, srcDescPtr, d_output, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            rppt_brightness_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
            rppt_brightness_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        else
            missingFuncFlag = 1;

        end = clock();

        break;
    }
    default:
        missingFuncFlag = 1;
        break;
    }

    if (missingFuncFlag == 1)
    {
        printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
        return -1;
    }

    // Display measured times

    gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    cout << "\nGPU Time - BatchPD : " << gpu_time_used;
    printf("\n");

    // Reconvert other bit depths to 8u for output display purposes

    string fileName = std::to_string(ip_bitDepth);
    ofstream outputFile (fileName + ".csv");

    if (ip_bitDepth == 0)
    {
        hipMemcpy(output, d_output, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToHost);
        Rpp8u *outputTemp;
        outputTemp = output;

        if (outputFile.is_open())
        {
            for (int i = 0; i < oBufferSize; i++)
            {
                outputFile << (Rpp32u) *outputTemp << ",";
                outputTemp++;
            }
            outputFile.close();
        }
        else
            cout << "Unable to open file!";

    }
    else if ((ip_bitDepth == 1) || (ip_bitDepth == 3))
    {
        hipMemcpy(outputf16, d_outputf16, oBufferSize * sizeof(Rpp16f), hipMemcpyDeviceToHost);
        Rpp8u *outputTemp;
        outputTemp = output;
        Rpp16f *outputf16Temp;
        outputf16Temp = outputf16;

        if (outputFile.is_open())
        {
            for (int i = 0; i < oBufferSize; i++)
            {
                outputFile << *outputf16Temp << ",";
                *outputTemp = (Rpp8u)RPPPIXELCHECK(*outputf16Temp * 255.0);
                outputf16Temp++;
                outputTemp++;
            }
            outputFile.close();
        }
        else
            cout << "Unable to open file!";

    }
    else if ((ip_bitDepth == 2) || (ip_bitDepth == 4))
    {
        hipMemcpy(outputf32, d_outputf32, oBufferSize * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        Rpp8u *outputTemp;
        outputTemp = output;
        Rpp32f *outputf32Temp;
        outputf32Temp = outputf32;

        if (outputFile.is_open())
        {
            for (int i = 0; i < oBufferSize; i++)
            {
                outputFile << *outputf32Temp << ",";
                *outputTemp = (Rpp8u)RPPPIXELCHECK(*outputf32Temp * 255.0);
                outputf32Temp++;
                outputTemp++;
            }
            outputFile.close();
        }
        else
            cout << "Unable to open file!";
    }
    else if ((ip_bitDepth == 5) || (ip_bitDepth == 6))
    {
        hipMemcpy(outputi8, d_outputi8, oBufferSize * sizeof(Rpp8s), hipMemcpyDeviceToHost);
        Rpp8u *outputTemp;
        outputTemp = output;
        Rpp8s *outputi8Temp;
        outputi8Temp = outputi8;

        if (outputFile.is_open())
        {
            for (int i = 0; i < oBufferSize; i++)
            {
                outputFile << (Rpp32s) *outputi8Temp << ",";
                *outputTemp = (Rpp8u) RPPPIXELCHECK(((Rpp32s) *outputi8Temp) + 128);
                outputi8Temp++;
                outputTemp++;
            }
            outputFile.close();
        }
        else
            cout << "Unable to open file!";
    }

    // Calculate exact dstROI in XYWH format for OpenCV dump

    if (roiTypeSrc == RpptRoiType::LTRB)
    {
        for (int i = 0; i < dstDescPtr->n; i++)
        {
            int ltX = roiTensorPtrSrc[i].ltrbROI.lt.x;
            int ltY = roiTensorPtrSrc[i].ltrbROI.lt.y;
            int rbX = roiTensorPtrSrc[i].ltrbROI.rb.x;
            int rbY = roiTensorPtrSrc[i].ltrbROI.rb.y;

            roiTensorPtrSrc[i].xywhROI.xy.x = ltX;
            roiTensorPtrSrc[i].xywhROI.xy.y = ltY;
            roiTensorPtrSrc[i].xywhROI.roiWidth = rbX - ltX + 1;
            roiTensorPtrSrc[i].xywhROI.roiHeight = rbY - ltY + 1;
        }
    }

    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = dstDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = dstDescPtr->h;

    for (int i = 0; i < dstDescPtr->n; i++)
    {
        roiTensorPtrSrc[i].xywhROI.roiWidth = RPPMIN2(roiPtrDefault->xywhROI.roiWidth - roiTensorPtrSrc[i].xywhROI.xy.x, roiTensorPtrSrc[i].xywhROI.roiWidth);
        roiTensorPtrSrc[i].xywhROI.roiHeight = RPPMIN2(roiPtrDefault->xywhROI.roiHeight - roiTensorPtrSrc[i].xywhROI.xy.y, roiTensorPtrSrc[i].xywhROI.roiHeight);
        roiTensorPtrSrc[i].xywhROI.xy.x = RPPMAX2(roiPtrDefault->xywhROI.xy.x, roiTensorPtrSrc[i].xywhROI.xy.x);
        roiTensorPtrSrc[i].xywhROI.xy.y = RPPMAX2(roiPtrDefault->xywhROI.xy.y, roiTensorPtrSrc[i].xywhROI.xy.y);
    }

    rppDestroyGPU(handle);

    // OpenCV dump

    mkdir(dst, 0700);
    strcat(dst, "/");
    count = 0;
    elementsInRowMax = dstDescPtr->w * ip_channel;

    for (j = 0; j < dstDescPtr->n; j++)
    {
        int height = roiTensorPtrSrc[j].xywhROI.roiHeight;
        int width = roiTensorPtrSrc[j].xywhROI.roiWidth;

        int op_size = height * width * ip_channel;
        Rpp8u *temp_output = (Rpp8u *)calloc(op_size, sizeof(Rpp8u));
        Rpp8u *temp_output_row;
        temp_output_row = temp_output;
        Rpp32u elementsInRow = width * ip_channel;
        Rpp8u *output_row = output + count;

        for (int k = 0; k < height; k++)
        {
            memcpy(temp_output_row, (output_row), elementsInRow * sizeof (Rpp8u));
            temp_output_row += elementsInRow;
            output_row += elementsInRowMax;
        }
        count += dstDescPtr->strides.nStride;

        char temp[1000];
        strcpy(temp, dst);
        strcat(temp, imageNames[j]);

        Mat mat_op_image;
        mat_op_image = Mat(height, width, CV_8UC1, temp_output);
        imwrite(temp, mat_op_image);

        free(temp_output);
    }

    // Free memory

    free(roiTensorPtrSrc);
    free(roiTensorPtrDst);
    hipFree(d_roiTensorPtrSrc);
    hipFree(d_roiTensorPtrDst);
    free(input);
    free(input_second);
    free(output);

    if (ip_bitDepth == 0)
    {
        hipFree(d_input);
        hipFree(d_input_second);
        hipFree(d_output);
    }
    else if (ip_bitDepth == 1)
    {
        free(inputf16);
        free(inputf16_second);
        free(outputf16);
        hipFree(d_inputf16);
        hipFree(d_inputf16_second);
        hipFree(d_outputf16);
    }
    else if (ip_bitDepth == 2)
    {
        free(inputf32);
        free(inputf32_second);
        free(outputf32);
        hipFree(d_inputf32);
        hipFree(d_inputf32_second);
        hipFree(d_outputf32);
    }
    else if (ip_bitDepth == 3)
    {
        free(outputf16);
        hipFree(d_input);
        hipFree(d_input_second);
        hipFree(d_outputf16);
    }
    else if (ip_bitDepth == 4)
    {
        free(outputf32);
        hipFree(d_input);
        hipFree(d_input_second);
        hipFree(d_outputf32);
    }
    else if (ip_bitDepth == 5)
    {
        free(inputi8);
        free(inputi8_second);
        free(outputi8);
        hipFree(d_inputi8);
        hipFree(d_inputi8_second);
        hipFree(d_outputi8);
    }
    else if (ip_bitDepth == 6)
    {
        free(outputi8);
        hipFree(d_input);
        hipFree(d_input_second);
        hipFree(d_outputi8);
    }

    return 0;
}
