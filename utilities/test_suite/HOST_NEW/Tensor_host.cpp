#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "rpp.h"
#include "rpp_test_suite_common.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half/half.hpp>
#include <fstream>
#include "helpers/testSuite_helper.hpp"

using namespace cv;
using namespace std;
using half_float::half;

typedef half Rpp16f;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))
#define RPPMAX2(a, b) ((a > b) ? a : b)
#define RPPMIN2(a, b) ((a < b) ? a : b)


int main(int argc, char **argv)
{
    // Handle inputs

    const int MIN_ARG_COUNT = 11;

    char *src = argv[1];
    char *src_second = argv[2];
    string dst = argv[3];

    int ip_bitDepth = atoi(argv[4]);
    unsigned int outputFormatToggle = atoi(argv[5]);
    int test_case = atoi(argv[6]);
    int num_iterations = atoi(argv[7]);
    int test_type = atoi(argv[8]);     // 0 for unit and 1 for performance test
    int layout_type = atoi(argv[9]); // 0 for pkd3 / 1 for pln3 / 2 for pln1

    bool additionalParamCase = (test_case == 8 || test_case == 21);
    bool kernelSizeCase = false;
    bool interpolationTypeCase = (test_case == 21);
    bool noiseTypeCase = (test_case == 8);
    bool pln1OutTypeCase = (test_case == 86);

    unsigned int verbosity = additionalParamCase ? atoi(argv[8]) : atoi(argv[7]);
    unsigned int additionalParam = additionalParamCase ? atoi(argv[7]) : 1;

    if (verbosity == 1)
    {
        printf("\nInputs for this test case are:");
        printf("\nsrc1 = %s", argv[1]);
        printf("\nsrc2 = %s", argv[2]);
        if (test_type == 0)
            printf("\ndst = %s", argv[3]);

        printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[4]);
        printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[5]);
        printf("\ncase number (0:86) = %s", argv[6]);
        printf("\nNumber of times to run = %s", argv[7]);
        printf("\nUNIT/PERFORMANCE - 0/1 = %s", argv[8]);

    }

    Rpp32u elementsInRow;
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        if (test_type == 0)
        {
            printf("\nUsage: <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:86> <number of iterations > 0> <verbosity = 0/1>>\n");
            if (layout_type == 2)
            {
                if (atoi(argv[5]) != 0)
                {
                    printf("\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = 0\n");
                    return -1;
                }
            }
        }
    }

    int ip_channel;
    string funcType;

    // Initialize tensor descriptors
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    if(layout_type == 0)
    {
        ip_channel = 3;
        funcType = "Tensor_HOST_PKD3";
        srcDescPtr->layout = RpptLayout::NHWC;

        // Set src/dst layouts in tensor descriptors
        if (pln1OutTypeCase)
        {
            funcType += "_toPLN1";
            dstDescPtr->layout = RpptLayout::NCHW;
        }
        else
        {
            if (outputFormatToggle == 0)
            {
                funcType += "_toPKD3";
                dstDescPtr->layout = RpptLayout::NHWC;
            }
            else if (outputFormatToggle == 1)
            {
                funcType += "_toPLN3";
                dstDescPtr->layout = RpptLayout::NCHW;
            }
        }
    }
    else if(layout_type == 1)
    {
        ip_channel = 3;
        funcType = "Tensor_HOST_PLN3";
        srcDescPtr->layout = RpptLayout::NCHW;

        // Set src/dst layouts in tensor descriptors
        if (pln1OutTypeCase)
        {
            funcType += "_toPLN1";
            dstDescPtr->layout = RpptLayout::NCHW;
        }
        else
        {
            if (outputFormatToggle == 0)
            {
                funcType += "_toPLN3";
                dstDescPtr->layout = RpptLayout::NCHW;
            }
            else if (outputFormatToggle == 1)
            {
                funcType += "_toPKD3";
                dstDescPtr->layout = RpptLayout::NHWC;
            }
        }
    }
    else
    {
        ip_channel = 1;
        funcType = "Tensor_HOST_PLN1";
        funcType += "_toPLN1";

        // Set src/dst layouts in tensor descriptors
        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
    }

    // Set case names
    string funcName = "";
    funcName += funcType;

    switch(test_case)
    {
        case 0:
            funcName = "brightness";
            break;
        case 1:
            funcName = "gamma_correction";
            break;
        case 2:
            funcName = "blend";
            break;
        case 4:
            funcName = "contrast";
            break;
        case 8:
            funcName = "noise";
            break;
        case 13:
            funcName = "exposure";
            break;
        case 20:
            funcName = "flip";
            break;
        case 21:
            funcName = "resize";
            break;
        case 31:
            funcName = "color_cast";
            break;
        case 36:
            funcName = "color_twist";
            break;
        case 37:
            funcName = "crop";
            break;
        case 38:
            funcName = "crop_mirror_normalize";
            break;
        case 70:
            funcName = "copy";
            break;
        case 80:
            funcName = "resize_mirror_normalize";
            break;
        case 81:
            funcName = "color_jitter";
            break;
        case 83:
            funcName = "gridmask";
            break;
        case 84:
            funcName = "spatter";
            break;
        case 85:
            funcName = "swap_channels";
            break;
        case 86:
            funcName = "color_to_greyscale";
            break;
        default:
            funcName = "test_case";
            break;
    }

    // Set src/dst data types in tensor descriptors
    set_data_type(ip_bitDepth, funcName, srcDescPtr, dstDescPtr);

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
    string src1 = "";
    src1 = src;
    src1 += "/";
    string src1_second = "";
    src1_second = src_second;
    src1_second += "/";

    string func="";
    func = funcName;
    func += funcType;

    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    if (kernelSizeCase)
    {
        char additionalParam_char[2];
        std::sprintf(additionalParam_char, "%u", additionalParam);
        func += "_kSize";
        func += additionalParam_char;
        if (test_type == 0)
        {
            dst += "_kSize";
            dst += additionalParam_char;
        }
    }
    else if (interpolationTypeCase)
    {
        std::string interpolationTypeName;
        interpolationTypeName = get_interpolation_type(additionalParam, interpolationType);
        func += "_interpolationType";
        func += interpolationTypeName.c_str();
        if (test_type == 0)
        {
            dst += "_interpolationType";
            dst += interpolationTypeName.c_str();
        }
    }
    else if (noiseTypeCase)
    {
        std::string noiseTypeName;
        noiseTypeName = get_noise_type(additionalParam);
        func += "_noiseType";
        func += noiseTypeName.c_str();
        if (test_type == 0)
        {
            dst += "_noiseType";
            dst += noiseTypeName.c_str();
        }
    }
    printf("\nRunning %s...", func.c_str());
    dst += "/";
    dst += func;

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
    RpptROI *roiTensorPtrSrc = (RpptROI *)calloc(noOfImages, sizeof(RpptROI));
    RpptROI *roiTensorPtrDst = (RpptROI *)calloc(noOfImages, sizeof(RpptROI));

    // Initialize the ImagePatch for source and destination
    RpptImagePatch *srcImgSizes = (RpptImagePatch *)calloc(noOfImages, sizeof(RpptImagePatch));
    RpptImagePatch *dstImgSizes = (RpptImagePatch *)calloc(noOfImages, sizeof(RpptImagePatch));

    // Set ROI tensors types for src/dst
    RpptRoiType roiTypeSrc, roiTypeDst;
    roiTypeSrc = RpptRoiType::XYWH;
    roiTypeDst = RpptRoiType::XYWH;

    // Set maxHeight, maxWidth and ROIs for src/dst
    const int images = noOfImages;
    string imageNames[images];

    DIR *dr1 = opendir(src);
    while ((de = readdir(dr1)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        imageNames[count] = de->d_name;
        string temp = "";
        temp = src1;
        temp += imageNames[count];
        if (layout_type == 0 || layout_type == 1)
            image = imread(temp, 1);

        else
            image = imread(temp, 0);

        roiTensorPtrSrc[count].xywhROI.xy.x = 0;
        roiTensorPtrSrc[count].xywhROI.xy.y = 0;
        roiTensorPtrSrc[count].xywhROI.roiWidth = image.cols;
        roiTensorPtrSrc[count].xywhROI.roiHeight = image.rows;

        roiTensorPtrDst[count].xywhROI.xy.x = 0;
        roiTensorPtrDst[count].xywhROI.xy.y = 0;
        roiTensorPtrDst[count].xywhROI.roiWidth = image.cols;
        roiTensorPtrDst[count].xywhROI.roiHeight = image.rows;

        srcImgSizes[count].width = roiTensorPtrSrc[count].xywhROI.roiWidth;
        srcImgSizes[count].height = roiTensorPtrSrc[count].xywhROI.roiHeight;
        dstImgSizes[count].width = roiTensorPtrDst[count].xywhROI.roiWidth;
        dstImgSizes[count].height = roiTensorPtrDst[count].xywhROI.roiHeight;

        maxHeight = RPPMAX2(maxHeight, roiTensorPtrSrc[count].xywhROI.roiHeight);
        maxWidth = RPPMAX2(maxWidth, roiTensorPtrSrc[count].xywhROI.roiWidth);
        maxDstHeight = RPPMAX2(maxDstHeight, roiTensorPtrDst[count].xywhROI.roiHeight);
        maxDstWidth = RPPMAX2(maxDstWidth, roiTensorPtrDst[count].xywhROI.roiWidth);

        count++;
    }
    closedir(dr1);

    // Set numDims, offset, n/c/h/w values for src/dst
    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    srcDescPtr->offsetInBytes = 0;
    dstDescPtr->offsetInBytes = 0;

    srcDescPtr->n = noOfImages;
    srcDescPtr->h = maxHeight;
    srcDescPtr->w = maxWidth;
    srcDescPtr->c = ip_channel;

    dstDescPtr->n = noOfImages;
    dstDescPtr->h = maxDstHeight;
    dstDescPtr->w = maxDstWidth;
    if (layout_type == 0 || layout_type == 1)
        dstDescPtr->c = (pln1OutTypeCase) ? 1 : ip_channel;
    else
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst
    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
    set_nchw_strides(layout_type, srcDescPtr, dstDescPtr);

    // Set buffer sizes for src/dst
    ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
    oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

    // Initialize host buffers for src/dst
    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(oBufferSize, sizeof(Rpp8u));

    Rpp16f *inputf16 = (Rpp16f *)calloc(ioBufferSize, sizeof(Rpp16f));
    Rpp16f *inputf16_second = (Rpp16f *)calloc(ioBufferSize, sizeof(Rpp16f));
    Rpp16f *outputf16 = (Rpp16f *)calloc(oBufferSize, sizeof(Rpp16f));

    Rpp32f *inputf32 = (Rpp32f *)calloc(ioBufferSize, sizeof(Rpp32f));
    Rpp32f *inputf32_second = (Rpp32f *)calloc(ioBufferSize, sizeof(Rpp32f));
    Rpp32f *outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));

    Rpp8s *inputi8 = (Rpp8s *)calloc(ioBufferSize, sizeof(Rpp8s));
    Rpp8s *inputi8_second = (Rpp8s *)calloc(ioBufferSize, sizeof(Rpp8s));
    Rpp8s *outputi8 = (Rpp8s *)calloc(oBufferSize, sizeof(Rpp8s));

    // Set 8u host buffers for src/dst
    DIR *dr2 = opendir(src);
    DIR *dr2_second = opendir(src_second);
    count = 0;
    i = 0;

    while ((de = readdir(dr2)) != NULL)
    {
        Rpp8u *input_temp, *input_second_temp;
        input_temp = input + (i * srcDescPtr->strides.nStride);
        input_second_temp = input_second + (i * srcDescPtr->strides.nStride);

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;

        string temp;
        temp = src1;
        temp += de->d_name;

        string temp_second = "";
        temp_second = src1_second;
        temp_second += de->d_name;

        if (layout_type == 0 || layout_type == 1)
        {
            image = imread(temp, 1);
            image_second = imread(temp_second, 1);
        }
        else if (layout_type == 2)
        {
            image = imread(temp, 0);
            image_second = imread(temp_second, 0);
        }

        Rpp8u *ip_image = image.data;
        Rpp8u *ip_image_second = image_second.data;

        Rpp32u elementsInRow;
        if (layout_type == 0 || layout_type == 1)
            elementsInRow = roiTensorPtrSrc[i].xywhROI.roiWidth * srcDescPtr->c;
        else if(layout_type == 2)
            elementsInRow = roiTensorPtrSrc[i].xywhROI.roiWidth;

        for (j = 0; j < roiTensorPtrSrc[i].xywhROI.roiHeight; j++)
        {
            memcpy(input_temp, ip_image, elementsInRow * sizeof(Rpp8u));
            memcpy(input_second_temp, ip_image_second, elementsInRow * sizeof(Rpp8u));
            ip_image += elementsInRow;
            ip_image_second += elementsInRow;
            input_temp += srcDescPtr->w * srcDescPtr->c;
            input_second_temp += srcDescPtr->w * srcDescPtr->c;
        }
        i++;
        count++;
    }
    closedir(dr2);

    if (layout_type == 1)
    {
        // Convert default OpenCV PKD3 to PLN3 for first input batch
        Rpp8u *inputCopy = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
        memcpy(inputCopy, input, ioBufferSize * sizeof(Rpp8u));

        Rpp8u *inputTemp, *inputCopyTemp;

        // omp_set_dynamic(0);
        // #pragma omp parallel for num_threads(noOfImages)
        for (int count = 0; count < noOfImages; count++)
        {
            Rpp8u *inputTempR, *inputTempG, *inputTempB;
            inputTemp = input + count * srcDescPtr->strides.nStride;
            inputCopyTemp = inputCopy + count * srcDescPtr->strides.nStride;
            inputTempR = inputTemp;
            inputTempG = inputTempR + srcDescPtr->strides.cStride;
            inputTempB = inputTempG + srcDescPtr->strides.cStride;
            for (int i = 0; i < srcDescPtr->h; i++)
            {
                for (int j = 0; j < srcDescPtr->w; j++)
                {
                    *inputTempR = *inputCopyTemp;
                    inputCopyTemp++;
                    inputTempR++;
                    *inputTempG = *inputCopyTemp;
                    inputCopyTemp++;
                    inputTempG++;
                    *inputTempB = *inputCopyTemp;
                    inputCopyTemp++;
                    inputTempB++;
                }
            }
        }

        free(inputCopy);

        // Convert default OpenCV PKD3 to PLN3 for second input batch
        Rpp8u *inputSecondCopy = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
        memcpy(inputSecondCopy, input_second, ioBufferSize * sizeof(Rpp8u));

        Rpp8u *inputSecondTemp, *inputSecondCopyTemp;
        inputSecondTemp = input_second;
        inputSecondCopyTemp = inputSecondCopy;
        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(noOfImages)
        for (int count = 0; count < noOfImages; count++)
        {
            Rpp8u *inputSecondTempR, *inputSecondTempG, *inputSecondTempB;
            inputSecondTempR = inputSecondTemp;
            inputSecondTempG = inputSecondTempR + srcDescPtr->strides.cStride;
            inputSecondTempB = inputSecondTempG + srcDescPtr->strides.cStride;

            for (int i = 0; i < srcDescPtr->h; i++)
            {
                for (int j = 0; j < srcDescPtr->w; j++)
                {
                    *inputSecondTempR = *inputSecondCopyTemp;
                    inputSecondCopyTemp++;
                    inputSecondTempR++;
                    *inputSecondTempG = *inputSecondCopyTemp;
                    inputSecondCopyTemp++;
                    inputSecondTempG++;
                    *inputSecondTempB = *inputSecondCopyTemp;
                    inputSecondCopyTemp++;
                    inputSecondTempB++;
                }
            }

            inputSecondTemp += srcDescPtr->strides.nStride;
        }

        free(inputSecondCopy);
    }

    // Convert inputs to test various other bit depths
    if (ip_bitDepth == 1)
    {
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
    }
    else if (ip_bitDepth == 2)
    {
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
    }
    else if (ip_bitDepth == 5)
    {
        Rpp8u *inputTemp, *input_secondTemp;
        Rpp8s *inputi8Temp, *inputi8_secondTemp;

        inputTemp = input;
        input_secondTemp = input_second;

        inputi8Temp = inputi8;
        inputi8_secondTemp = inputi8_second;

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputi8Temp = (Rpp8s)(((Rpp32s)*inputTemp) - 128);
            *inputi8_secondTemp = (Rpp8s)(((Rpp32s)*input_secondTemp) - 128);
            inputTemp++;
            inputi8Temp++;
            input_secondTemp++;
            inputi8_secondTemp++;
        }
    }

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, noOfImages);

    double max_time_used = 0, min_time_used = 500, avg_time_used = 0;
    double cpu_time_used, omp_time_used;
    string test_case_name;

    // case-wise RPP API and measure time script for Unit and Performance test
    printf("\nRunning %s %d times (each time with a batch size of %d images) and computing mean statistics...", func.c_str(), num_iterations, noOfImages);
    for (int perfRunCount = 0; perfRunCount < num_iterations; perfRunCount++)
    {
        clock_t start, end;
        double start_omp, end_omp;
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
                }

                // Uncomment to run test case with an xywhROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }*/

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_brightness_host(input, srcDescPtr, output, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_brightness_host(inputf16, srcDescPtr, outputf16, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_brightness_host(inputf32, srcDescPtr, outputf32, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_brightness_host(inputi8, srcDescPtr, outputi8, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 1:
            {
                test_case_name = "gamma_correction";

                Rpp32f gammaVal[images];
                for (i = 0; i < images; i++)
                {
                    gammaVal[i] = 1.9;
                }

                // Uncomment to run test case with an xywhROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }*/

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_gamma_correction_host(input, srcDescPtr, output, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_gamma_correction_host(inputf16, srcDescPtr, outputf16, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_gamma_correction_host(inputf32, srcDescPtr, outputf32, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_gamma_correction_host(inputi8, srcDescPtr, outputi8, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 2:
            {
                test_case_name = "blend";

                Rpp32f alpha[images];
                for (i = 0; i < images; i++)
                {
                    alpha[i] = 0.4;
                }

                // Uncomment to run test case with an xywhROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }*/

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_blend_host(input, input_second, srcDescPtr, output, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_blend_host(inputf16, inputf16_second, srcDescPtr, outputf16, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_blend_host(inputf32, inputf32_second, srcDescPtr, outputf32, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_blend_host(inputi8, inputi8_second, srcDescPtr, outputi8, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 4:
            {
                test_case_name = "contrast";

                Rpp32f contrastFactor[images];
                Rpp32f contrastCenter[images];
                for (i = 0; i < images; i++)
                {
                    contrastFactor[i] = 2.96;
                    contrastCenter[i] = 128;
                }

                // Uncomment to run test case with an xywhROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }*/

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_contrast_host(input, srcDescPtr, output, dstDescPtr, contrastFactor, contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_contrast_host(inputf16, srcDescPtr, outputf16, dstDescPtr, contrastFactor, contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_contrast_host(inputf32, srcDescPtr, outputf32, dstDescPtr, contrastFactor, contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_contrast_host(inputi8, srcDescPtr, outputi8, dstDescPtr, contrastFactor, contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 8:
            {
                test_case_name = "noise";

                switch (additionalParam)
                {
                case 0:
                {
                    Rpp32f noiseProbabilityTensor[images];
                    Rpp32f saltProbabilityTensor[images];
                    Rpp32f saltValueTensor[images];
                    Rpp32f pepperValueTensor[images];
                    Rpp32u seed = 1255459;
                    for (i = 0; i < images; i++)
                    {
                        noiseProbabilityTensor[i] = 0.1f;
                        saltProbabilityTensor[i] = 0.5f;
                        saltValueTensor[i] = 1.0f;
                        pepperValueTensor[i] = 0.0f;
                    }

                    // Uncomment to run test case with an xywhROI override
                    /*for (i = 0; i < images; i++)
                    {
                        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                        roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                        roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                    }*/

                    // Uncomment to run test case with an ltrbROI override
                    /*for (i = 0; i < images; i++)
                        roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                        roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                        roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                        roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    }
                    roiTypeSrc = RpptRoiType::LTRB;
                    roiTypeDst = RpptRoiType::LTRB;*/

                    start_omp = omp_get_wtime();
                    start = clock();
                    if (ip_bitDepth == 0)
                        rppt_salt_and_pepper_noise_host(input, srcDescPtr, output, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 1)
                        rppt_salt_and_pepper_noise_host(inputf16, srcDescPtr, outputf16, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 2)
                        rppt_salt_and_pepper_noise_host(inputf32, srcDescPtr, outputf32, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 3)
                        missingFuncFlag = 1;
                    else if (ip_bitDepth == 4)
                        missingFuncFlag = 1;
                    else if (ip_bitDepth == 5)
                        rppt_salt_and_pepper_noise_host(inputi8, srcDescPtr, outputi8, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 6)
                        missingFuncFlag = 1;
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 1:
                {
                    missingFuncFlag = 1;
                    break;
                }
                case 2:
                {
                    missingFuncFlag = 1;
                    break;
                }
                default:
                {
                    missingFuncFlag = 1;
                    break;
                }
                }

                break;
            }

            case 13:
            {
                test_case_name = "exposure";

                Rpp32f exposureFactor[images];
                for (i = 0; i < images; i++)
                {
                    exposureFactor[i] = 1.4;
                }

                // Uncomment to run test case with an xywhROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }*/

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_exposure_host(input, srcDescPtr, output, dstDescPtr, exposureFactor, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_exposure_host(inputf16, srcDescPtr, outputf16, dstDescPtr, exposureFactor, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_exposure_host(inputf32, srcDescPtr, outputf32, dstDescPtr, exposureFactor, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_exposure_host(inputi8, srcDescPtr, outputi8, dstDescPtr, exposureFactor, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 20:
            {
                test_case_name = "flip";

                Rpp32u horizontalFlag[images];
                Rpp32u verticalFlag[images];
                for (i = 0; i < images; i++)
                {
                    horizontalFlag[i] = 1;
                    verticalFlag[i] = 0;
                }

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_flip_host(input, srcDescPtr, output, dstDescPtr, horizontalFlag, verticalFlag, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_flip_host(inputf16, srcDescPtr, outputf16, dstDescPtr, horizontalFlag, verticalFlag, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_flip_host(inputf32, srcDescPtr, outputf32, dstDescPtr, horizontalFlag, verticalFlag, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_flip_host(inputi8, srcDescPtr, outputi8, dstDescPtr, horizontalFlag, verticalFlag, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 21:
            {
                test_case_name = "resize";

                for (i = 0; i < images; i++)
                {
                    dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
                    dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
                }

                // Uncomment to run test case with an xywhROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }*/

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_resize_host(input, srcDescPtr, output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_resize_host(inputf16, srcDescPtr, outputf16, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_resize_host(inputf32, srcDescPtr, outputf32, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_resize_host(inputi8, srcDescPtr, outputi8, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 31:
            {
                test_case_name = "color_cast";

                RpptRGB rgbTensor[images];
                Rpp32f alphaTensor[images];
                for (i = 0; i < images; i++)
                {
                    rgbTensor[i].R = 0;
                    rgbTensor[i].G = 0;
                    rgbTensor[i].B = 100;
                    alphaTensor[i] = 0.5;
                }

                // Uncomment to run test case with an xywhROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }*/

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_color_cast_host(input, srcDescPtr, output, dstDescPtr, rgbTensor, alphaTensor, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_color_cast_host(inputf16, srcDescPtr, outputf16, dstDescPtr, rgbTensor, alphaTensor, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_color_cast_host(inputf32, srcDescPtr, outputf32, dstDescPtr, rgbTensor, alphaTensor, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_color_cast_host(inputi8, srcDescPtr, outputi8, dstDescPtr, rgbTensor, alphaTensor, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 36:
            {
                test_case_name = "color_twist";

                Rpp32f brightness[images];
                Rpp32f contrast[images];
                Rpp32f hue[images];
                Rpp32f saturation[images];
                for (i = 0; i < images; i++)
                {
                    brightness[i] = 1.4;
                    contrast[i] = 0.0;
                    hue[i] = 60.0;
                    saturation[i] = 1.9;
                }

                // Uncomment to run test case with an xywhROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }*/

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_color_twist_host(input, srcDescPtr, output, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_color_twist_host(inputf16, srcDescPtr, outputf16, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_color_twist_host(inputf32, srcDescPtr, outputf32, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_color_twist_host(inputi8, srcDescPtr, outputi8, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }
            case 37:
            {
                test_case_name = "crop";

                // Uncomment to run test case with an xywhROI override
                for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_crop_host(input, srcDescPtr, output, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_crop_host(inputf16, srcDescPtr, outputf16, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_crop_host(inputf32, srcDescPtr, outputf32, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_crop_host(inputi8, srcDescPtr, outputi8, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 38:
            {
                test_case_name = "crop_mirror_normalize";
                Rpp32f mean[images * 3];
                Rpp32f stdDev[images* 3];
                Rpp32u mirror[images];
                for (int i = 0, j = 0; i < images ; i++, j+=3)
                {
                    mean[j] = 0.0;
                    stdDev[j] = 1.0;
                    mirror[i] = 1;
                }

                // Uncomment to run test case with an xywhROI override
                for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_crop_mirror_normalize_host(input, srcDescPtr, output, dstDescPtr, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_crop_mirror_normalize_host(inputf16, srcDescPtr, outputf16, dstDescPtr, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_crop_mirror_normalize_host(inputf32, srcDescPtr, outputf32, dstDescPtr, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_crop_mirror_normalize_host(inputi8, srcDescPtr, outputi8, dstDescPtr, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 70:
            {
                test_case_name = "copy";

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_copy_host(input, srcDescPtr, output, dstDescPtr, handle);
                else if (ip_bitDepth == 1)
                    rppt_copy_host(inputf16, srcDescPtr, outputf16, dstDescPtr, handle);
                else if (ip_bitDepth == 2)
                    rppt_copy_host(inputf32, srcDescPtr, outputf32, dstDescPtr, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_copy_host(inputi8, srcDescPtr, outputi8, dstDescPtr, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 80:
            {
                test_case_name = "resize_mirror_normalize";

                if (interpolationType != RpptInterpolationType::BILINEAR)
                {
                    missingFuncFlag = 1;
                    break;
                }

                for (i = 0; i < images; i++)
                {
                    dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
                    dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
                }
                int size_mean;
                if (layout_type == 0 || layout_type == 1)
                    size_mean = images * 3;

                else
                    size_mean = images;

                Rpp32f mean[size_mean];
                Rpp32f stdDev[size_mean];

                Rpp32u mirror[images];
                for (i = 0, j = 0; i < images; i++, j += 3)
                {
                    if (layout_type == 2)
                    {
                        mean[i] = 100.0;
                        stdDev[i] = 1.0;
                    }
                    else
                    {
                        mean[j] = 60.0;
                        stdDev[j] = 1.0;

                        mean[j + 1] = 80.0;
                        stdDev[j + 1] = 1.0;

                        mean[j + 2] = 100.0;
                        stdDev[j + 2] = 1.0;
                    }

                    mirror[i] = 1;
                }

                // Uncomment to run test case with an xywhROI override
                // for (i = 0; i < images; i++)
                // {
                //     roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                //     roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                //     dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                //     dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                // }

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_resize_mirror_normalize_host(input, srcDescPtr, output, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_resize_mirror_normalize_host(inputf16, srcDescPtr, outputf16, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_resize_mirror_normalize_host(inputf32, srcDescPtr, outputf32, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    rppt_resize_mirror_normalize_host(input, srcDescPtr, outputf16, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 4)
                    rppt_resize_mirror_normalize_host(input, srcDescPtr, outputf32, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 5)
                    rppt_resize_mirror_normalize_host(inputi8, srcDescPtr, outputi8, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 81:
            {
                test_case_name = "color_jitter";

                Rpp32f brightness[images];
                Rpp32f contrast[images];
                Rpp32f hue[images];
                Rpp32f saturation[images];
                for (i = 0; i < images; i++)
                {
                    brightness[i] = 1.02;
                    contrast[i] = 1.1;
                    hue[i] = 0.02;
                    saturation[i] = 1.3;
                }

                // Uncomment to run test case with an xywhROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }*/

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_color_jitter_host(input, srcDescPtr, output, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_color_jitter_host(inputf16, srcDescPtr, outputf16, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_color_jitter_host(inputf32, srcDescPtr, outputf32, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_color_jitter_host(inputi8, srcDescPtr, outputi8, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 83:
            {
                test_case_name = "gridmask";

                Rpp32u tileWidth = 40;
                Rpp32f gridRatio = 0.6;
                Rpp32f gridAngle = 0.5;
                RpptUintVector2D translateVector;
                translateVector.x = 0.0;
                translateVector.y = 0.0;

                // Uncomment to run test case with an xywhROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }*/

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_gridmask_host(input, srcDescPtr, output, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_gridmask_host(inputf16, srcDescPtr, outputf16, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_gridmask_host(inputf32, srcDescPtr, outputf32, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_gridmask_host(inputi8, srcDescPtr, outputi8, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 84:
            {
                test_case_name = "spatter";

                RpptRGB spatterColor;

                // Mud Spatter
                spatterColor.R = 65;
                spatterColor.G = 50;
                spatterColor.B = 23;

                // Blood Spatter
                // spatterColor.R = 98;
                // spatterColor.G = 3;
                // spatterColor.B = 3;

                // Ink Spatter
                // spatterColor.R = 5;
                // spatterColor.G = 20;
                // spatterColor.B = 64;

                // Uncomment to run test case with an xywhROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                }*/

                // Uncomment to run test case with an ltrbROI override
                /*for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                    roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                    roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                    roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                }
                roiTypeSrc = RpptRoiType::LTRB;
                roiTypeDst = RpptRoiType::LTRB;*/

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_spatter_host(input, srcDescPtr, output, dstDescPtr, spatterColor, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 1)
                    rppt_spatter_host(inputf16, srcDescPtr, outputf16, dstDescPtr, spatterColor, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 2)
                    rppt_spatter_host(inputf32, srcDescPtr, outputf32, dstDescPtr, spatterColor, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_spatter_host(inputi8, srcDescPtr, outputi8, dstDescPtr, spatterColor, roiTensorPtrSrc, roiTypeSrc, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 85:
            {
                test_case_name = "swap_channels";

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_swap_channels_host(input, srcDescPtr, output, dstDescPtr, handle);
                else if (ip_bitDepth == 1)
                    rppt_swap_channels_host(inputf16, srcDescPtr, outputf16, dstDescPtr, handle);
                else if (ip_bitDepth == 2)
                    rppt_swap_channels_host(inputf32, srcDescPtr, outputf32, dstDescPtr, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_swap_channels_host(inputi8, srcDescPtr, outputi8, dstDescPtr, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }

            case 86:
            {
                test_case_name = "color_to_greyscale";

                RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 0)
                    rppt_color_to_greyscale_host(input, srcDescPtr, output, dstDescPtr, srcSubpixelLayout, handle);
                else if (ip_bitDepth == 1)
                    rppt_color_to_greyscale_host(inputf16, srcDescPtr, outputf16, dstDescPtr, srcSubpixelLayout, handle);
                else if (ip_bitDepth == 2)
                    rppt_color_to_greyscale_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcSubpixelLayout, handle);
                else if (ip_bitDepth == 3)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 4)
                    missingFuncFlag = 1;
                else if (ip_bitDepth == 5)
                    rppt_color_to_greyscale_host(inputi8, srcDescPtr, outputi8, dstDescPtr, srcSubpixelLayout, handle);
                else if (ip_bitDepth == 6)
                    missingFuncFlag = 1;
                else
                    missingFuncFlag = 1;

                break;
            }
            default:
                missingFuncFlag = 1;
                break;
        }

        end = clock();
        end_omp = omp_get_wtime();

        if (missingFuncFlag == 1)
        {
            printf("\nThe functionality %s doesn't yet exist in RPP\n", func.c_str());
            return -1;
        }

        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        omp_time_used = end_omp - start_omp;

        if (cpu_time_used > max_time_used)
            max_time_used = cpu_time_used;
        if (cpu_time_used < min_time_used)
            min_time_used = cpu_time_used;
        avg_time_used += cpu_time_used;
    }

    if (test_type == 0)
    {
        cpu_time_used = cpu_time_used * 1000;
        omp_time_used = omp_time_used * 1000;
        cout << "\nCPU Time - BatchPD : " << cpu_time_used;
        cout << "\nOMP Time - BatchPD : " << omp_time_used;
        printf("\n");
    }
    else
    {
        avg_time_used /= num_iterations;
        max_time_used = max_time_used * 1000;
        min_time_used = min_time_used * 1000;
        avg_time_used = avg_time_used * 1000;
        // Display measured times

        cout << fixed << "\nmax,min,avg in ms = " << max_time_used << "," << min_time_used << "," << avg_time_used << endl;
    }

    if (test_type == 0)
    {
        // Reconvert other bit depths to 8u for output display purposes
        string fileName = std::to_string(ip_bitDepth);
        ofstream outputFile(fileName + ".csv");

        if (ip_bitDepth == 0)
        {
            Rpp8u *outputTemp;
            outputTemp = output;

            if (outputFile.is_open())
            {
                for (int i = 0; i < oBufferSize; i++)
                {
                    outputFile << (Rpp32u)*outputTemp << ",";
                    outputTemp++;
                }
                outputFile.close();
            }
            else
                cout << "Unable to open file!";
        }
        else if ((ip_bitDepth == 1) || (ip_bitDepth == 3))
        {
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
            Rpp8u *outputTemp;
            outputTemp = output;
            Rpp8s *outputi8Temp;
            outputi8Temp = outputi8;

            if (outputFile.is_open())
            {
                for (int i = 0; i < oBufferSize; i++)
                {
                    outputFile << (Rpp32s)*outputi8Temp << ",";
                    *outputTemp = (Rpp8u)RPPPIXELCHECK(((Rpp32s)*outputi8Temp) + 128);
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

        // Convert any PLN3 outputs to the corresponding PKD3 version for OpenCV dump
        if (layout_type == 0 || layout_type == 1)
        {
            if ((dstDescPtr->c == 3) && (dstDescPtr->layout == RpptLayout::NCHW))
                convert_pln3_to_pkd3(output, dstDescPtr);
        }
        rppDestroyHost(handle);

        // OpenCV dump (if test_type is unit test)
        mkdir(dst.c_str(), 0700);
        dst += "/";

        count = 0;
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        for (j = 0; j < dstDescPtr->n; j++)
        {
            int height = dstImgSizes[j].height;
            int width = dstImgSizes[j].width;
            int op_size;
            op_size = height * width * dstDescPtr->c;

            Rpp8u *temp_output = (Rpp8u *)calloc(op_size, sizeof(Rpp8u));
            Rpp8u *temp_output_row;
            temp_output_row = temp_output;
            elementsInRow = width * dstDescPtr->c;
            Rpp8u *output_row = output + count;

            for (int k = 0; k < height; k++)
            {
                memcpy(temp_output_row, output_row, elementsInRow * sizeof(Rpp8u));
                temp_output_row += elementsInRow;
                output_row += elementsInRowMax;
            }
            count += dstDescPtr->strides.nStride;

            string temp;
            temp = dst;
            temp += imageNames[j];

            Mat mat_op_image;
            if (layout_type == 0 || layout_type == 1)
                mat_op_image = (pln1OutTypeCase) ? Mat(height, width, CV_8UC1, temp_output) : Mat(height, width, CV_8UC3, temp_output);
            else if (layout_type == 2)
                mat_op_image = Mat(height, width, CV_8UC1, temp_output);

            imwrite(temp, mat_op_image);
            free(temp_output);
        }
    }

    // Free memory

    free(roiTensorPtrSrc);
    free(roiTensorPtrDst);
    free(input);
    free(input_second);
    free(output);
    free(inputf16);
    free(inputf16_second);
    free(outputf16);
    free(inputf32);
    free(inputf32_second);
    free(outputf32);
    free(inputi8);
    free(inputi8_second);
    free(outputi8);

    return 0;
}
