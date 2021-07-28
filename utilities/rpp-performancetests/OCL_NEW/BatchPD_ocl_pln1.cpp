#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "/opt/rocm/rpp/include/rppi.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half.hpp>
#include <fstream>
#include <algorithm>
#include <iterator>
#include "helpers/testSuite_helper.hpp"
#include </opt/rocm/opencl/include/CL/cl.h>

using namespace cv;
using namespace std;
using half_float::half;

typedef half Rpp16f;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))

int main(int argc, char **argv)
{
    const int MIN_ARG_COUNT = 7;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./BatchPD_ocl_pln1 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:81> <verbosity = 0/1>\n");
        return -1;
    }
    if (atoi(argv[4]) != 0)
    {
        printf("\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = 0\n");
        return -1;
    }

    if (atoi(argv[6]) == 1)
    {
        printf("\nInputs for this test case are:");
        printf("\nsrc1 = %s", argv[1]);
        printf("\nsrc2 = %s", argv[2]);
        printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[3]);
        printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[4]);
        printf("\ncase number (0:81) = %s", argv[5]);
    }

    char *src = argv[1];
    char *src_second = argv[2];
    int ip_bitDepth = atoi(argv[3]);
    unsigned int outputFormatToggle = atoi(argv[4]);
    int test_case = atoi(argv[5]);

    int ip_channel = 1;

    char funcType[1000] = {"BatchPD_OCL_PLN1_toPLN1"};

    char funcName[1000];
    switch (test_case)
    {
    case 0:
        strcpy(funcName, "brightness");
        outputFormatToggle = 0;
        break;
    case 1:
        strcpy(funcName, "gamma_correction");
        outputFormatToggle = 0;
        break;
    case 2:
        strcpy(funcName, "blend");
        outputFormatToggle = 0;
        break;
    case 3:
        strcpy(funcName, "blur");
        outputFormatToggle = 0;
        break;
    case 4:
        strcpy(funcName, "contrast");
        outputFormatToggle = 0;
        break;
    case 5:
        strcpy(funcName, "pixelate");
        outputFormatToggle = 0;
        break;
    case 6:
        strcpy(funcName, "jitter");
        outputFormatToggle = 0;
        break;
    case 7:
        strcpy(funcName, "snow");
        outputFormatToggle = 0;
        break;
    case 8:
        strcpy(funcName, "noise");
        outputFormatToggle = 0;
        break;
    case 9:
        strcpy(funcName, "random_shadow");
        outputFormatToggle = 0;
        break;
    case 10:
        strcpy(funcName, "fog");
        outputFormatToggle = 0;
        break;
    case 11:
        strcpy(funcName, "rain");
        outputFormatToggle = 0;
        break;
    case 12:
        strcpy(funcName, "random_crop_letterbox");
        outputFormatToggle = 0;
        break;
    case 13:
        strcpy(funcName, "exposure");
        outputFormatToggle = 0;
        break;
    case 14:
        strcpy(funcName, "histogram_balance");
        outputFormatToggle = 0;
        break;
    case 15:
        strcpy(funcName, "thresholding");
        outputFormatToggle = 0;
        break;
    case 16:
        strcpy(funcName, "min");
        outputFormatToggle = 0;
        break;
    case 17:
        strcpy(funcName, "max");
        outputFormatToggle = 0;
        break;
    case 18:
        strcpy(funcName, "integral");
        outputFormatToggle = 0;
        break;
    case 19:
        strcpy(funcName, "histogram_equalization");
        outputFormatToggle = 0;
        break;
    case 20:
        strcpy(funcName, "flip");
        outputFormatToggle = 0;
        break;
    case 21:
        strcpy(funcName, "resize");
        break;
    case 22:
        strcpy(funcName, "resize_crop");
        break;
    case 23:
        strcpy(funcName, "rotate");
        break;
    case 24:
        strcpy(funcName, "warp_affine");
        break;
    case 25:
        strcpy(funcName, "fisheye");
        outputFormatToggle = 0;
        break;
    case 26:
        strcpy(funcName, "lens_correction");
        outputFormatToggle = 0;
        break;
    case 27:
        strcpy(funcName, "scale");
        outputFormatToggle = 0;
        break;
    case 28:
        strcpy(funcName, "warp_perspective");
        outputFormatToggle = 0;
        break;
    case 29:
        strcpy(funcName, "water");
        break;
    case 30:
        strcpy(funcName, "non_linear_blend");
        break;
    case 31:
        strcpy(funcName, "color_cast");
        break;
    case 32:
        strcpy(funcName, "erase");
        break;
    case 33:
        strcpy(funcName, "crop_and_patch");
        break;
    case 34:
        strcpy(funcName, "lut");
        break;
    case 35:
        strcpy(funcName, "glitch");
        break;
    case 36:
        strcpy(funcName, "color_twist");
        break;
    case 37:
        strcpy(funcName, "crop");
        break;
    case 38:
        strcpy(funcName, "crop_mirror_normalize");
        break;
    case 39:
        strcpy(funcName, "resize_crop_mirror");
        break;
    case 40:
        strcpy(funcName, "erode");
        outputFormatToggle = 0;
        break;
    case 41:
        strcpy(funcName, "dilate");
        outputFormatToggle = 0;
        break;
    case 42:
        strcpy(funcName, "hueRGB");
        outputFormatToggle = 0;
        break;
    case 43:
        strcpy(funcName, "saturationRGB");
        outputFormatToggle = 0;
        break;
    case 44:
        strcpy(funcName, "color_convert");
        outputFormatToggle = 0;
        break;
    case 45:
        strcpy(funcName, "color_temperature");
        outputFormatToggle = 0;
        break;
    case 46:
        strcpy(funcName, "vignette");
        outputFormatToggle = 0;
        break;
    case 47:
        strcpy(funcName, "channel_combine and channel_extract");
        outputFormatToggle = 0;
        break;
    case 48:
        strcpy(funcName, "look_up_table");
        outputFormatToggle = 0;
        break;
    case 49:
        strcpy(funcName, "box_filter");
        outputFormatToggle = 0;
        break;
    case 50:
        strcpy(funcName, "sobel_filter");
        outputFormatToggle = 0;
        break;
    case 51:
        strcpy(funcName, "median_filter");
        outputFormatToggle = 0;
        break;
    case 52:
        strcpy(funcName, "custom_convolution");
        outputFormatToggle = 0;
        break;
    case 53:
        strcpy(funcName, "non_max_suppression");
        outputFormatToggle = 0;
        break;
    case 54:
        strcpy(funcName, "gaussian_filter");
        outputFormatToggle = 0;
        break;
    case 55:
        strcpy(funcName, "nonlinear_filter");
        outputFormatToggle = 0;
        break;
    case 56:
        strcpy(funcName, "absolute_difference");
        outputFormatToggle = 0;
        break;
    case 57:
        strcpy(funcName, "accumulate_weighted");
        outputFormatToggle = 0;
        break;
    case 58:
        strcpy(funcName, "accumulate");
        outputFormatToggle = 0;
        break;
    case 59:
        strcpy(funcName, "add");
        outputFormatToggle = 0;
        break;
    case 60:
        strcpy(funcName, "subtract");
        outputFormatToggle = 0;
        break;
    case 61:
        strcpy(funcName, "magnitude");
        outputFormatToggle = 0;
        break;
    case 62:
        strcpy(funcName, "multiply");
        outputFormatToggle = 0;
        break;
    case 63:
        strcpy(funcName, "phase");
        outputFormatToggle = 0;
        break;
    case 64:
        strcpy(funcName, "accumulate_squared");
        outputFormatToggle = 0;
        break;
    case 65:
        strcpy(funcName, "bitwise_AND");
        outputFormatToggle = 0;
        break;
    case 66:
        strcpy(funcName, "bitwise_NOT");
        outputFormatToggle = 0;
        break;
    case 67:
        strcpy(funcName, "exclusive_OR");
        outputFormatToggle = 0;
        break;
    case 68:
        strcpy(funcName, "inclusive_OR");
        outputFormatToggle = 0;
        break;
    case 69:
        strcpy(funcName, "local_binary_pattern");
        outputFormatToggle = 0;
        break;
    case 70:
        strcpy(funcName, "data_object_copy");
        outputFormatToggle = 0;
        break;
    case 71:
        strcpy(funcName, "gaussian_image_pyramid");
        outputFormatToggle = 0;
        break;
    case 72:
        strcpy(funcName, "laplacian_image_pyramid");
        outputFormatToggle = 0;
        break;
    case 73:
        strcpy(funcName, "canny_edge_detector");
        outputFormatToggle = 0;
        break;
    case 74:
        strcpy(funcName, "harris_corner_detector");
        outputFormatToggle = 0;
        break;
    case 75:
        strcpy(funcName, "fast_corner_detector");
        outputFormatToggle = 0;
        break;
    case 76:
        strcpy(funcName, "reconstruction_laplacian_image_pyramid");
        outputFormatToggle = 0;
        break;
    case 77:
        strcpy(funcName, "hough_lines");
        outputFormatToggle = 0;
        break;
    case 78:
        strcpy(funcName, "hog");
        outputFormatToggle = 0;
        break;
    case 79:
        strcpy(funcName, "remap");
        outputFormatToggle = 0;
        break;
    }

    if (ip_bitDepth == 0)
    {
        strcat(funcName, "_u8_");
    }
    else if (ip_bitDepth == 1)
    {
        strcat(funcName, "_f16_");
    }
    else if (ip_bitDepth == 2)
    {
        strcat(funcName, "_f32_");
    }
    else if (ip_bitDepth == 3)
    {
        strcat(funcName, "_u8_f16_");
    }
    else if (ip_bitDepth == 4)
    {
        strcat(funcName, "_u8_f32_");
    }
    else if (ip_bitDepth == 5)
    {
        strcat(funcName, "_i8_");
    }
    else if (ip_bitDepth == 6)
    {
        strcat(funcName, "_u8_i8_");
    }

    char func[1000];
    strcpy(func, funcName);
    strcat(func, funcType);

    int ip_bitDepth_1_cases[14] = {21, 22, 23, 24, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39};
    int ip_bitDepth_2_cases[14] = {21, 22, 23, 24, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39};
    int ip_bitDepth_3_cases[3]  = {21, 37, 38};
    int ip_bitDepth_4_cases[3]  = {21, 37, 38};
    int ip_bitDepth_5_cases[15] = {21, 22, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
    int ip_bitDepth_6_cases[3]  = {21, 37, 38};

    bool functionality_existence;

    if (ip_bitDepth == 0)
        functionality_existence = 1;
    else if (ip_bitDepth == 1)
        functionality_existence = std::any_of(std::begin(ip_bitDepth_1_cases), std::end(ip_bitDepth_1_cases), [&](int i) {return i == test_case;});
    else if (ip_bitDepth == 2)
        functionality_existence = std::any_of(std::begin(ip_bitDepth_2_cases), std::end(ip_bitDepth_2_cases), [&](int i) {return i == test_case;});
    else if (ip_bitDepth == 3)
        functionality_existence = std::any_of(std::begin(ip_bitDepth_3_cases), std::end(ip_bitDepth_3_cases), [&](int i) {return i == test_case;});
    else if (ip_bitDepth == 4)
        functionality_existence = std::any_of(std::begin(ip_bitDepth_4_cases), std::end(ip_bitDepth_4_cases), [&](int i) {return i == test_case;});
    else if (ip_bitDepth == 5)
        functionality_existence = std::any_of(std::begin(ip_bitDepth_5_cases), std::end(ip_bitDepth_5_cases), [&](int i) {return i == test_case;});
    else if (ip_bitDepth == 6)
        functionality_existence = std::any_of(std::begin(ip_bitDepth_6_cases), std::end(ip_bitDepth_6_cases), [&](int i) {return i == test_case;});

    if (functionality_existence == 0)
    {
        printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
        return -1;
    }

    int missingFuncFlag = 0;

    int i = 0, j = 0;
    int minHeight = 30000, minWidth = 30000, maxHeight = 0, maxWidth = 0;
    int minDstHeight = 30000, minDstWidth = 30000, maxDstHeight = 0, maxDstWidth = 0;
    unsigned long long count = 0;
    unsigned long long ioBufferSize = 0;
    unsigned long long oBufferSize = 0;
    static int noOfImages = 0;

    Mat image, image_second;

    struct dirent *de;
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");
    char src1_second[1000];
    strcpy(src1_second, src_second);
    strcat(src1_second, "/");
    strcat(funcName, funcType);

    DIR *dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfImages += 1;
    }
    closedir(dr);

    printf("\nRunning %s 100 times (each time with a batch size of %d images) and computing mean statistics...", func, noOfImages);

    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
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

        srcSize[count].height = image.rows;
        srcSize[count].width = image.cols;
        if (maxHeight < srcSize[count].height)
            maxHeight = srcSize[count].height;
        if (maxWidth < srcSize[count].width)
            maxWidth = srcSize[count].width;
        if (minHeight > srcSize[count].height)
            minHeight = srcSize[count].height;
        if (minWidth > srcSize[count].width)
            minWidth = srcSize[count].width;

        dstSize[count].height = image.rows;
        dstSize[count].width = image.cols;
        if (maxDstHeight < dstSize[count].height)
            maxDstHeight = dstSize[count].height;
        if (maxDstWidth < dstSize[count].width)
            maxDstWidth = dstSize[count].width;
        if (minDstHeight > dstSize[count].height)
            minDstHeight = dstSize[count].height;
        if (minDstWidth > dstSize[count].width)
            minDstWidth = dstSize[count].width;

        count++;
    }
    closedir(dr1);

    ioBufferSize = (unsigned long long)maxHeight * (unsigned long long)maxWidth * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
    oBufferSize = (unsigned long long)maxDstHeight * (unsigned long long)maxDstWidth * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(oBufferSize, sizeof(Rpp8u));

    RppiSize maxSize, maxDstSize;
    maxSize.height = maxHeight;
    maxSize.width = maxWidth;
    maxDstSize.height = maxDstHeight;
    maxDstSize.width = maxDstWidth;

    DIR *dr2 = opendir(src);
    DIR *dr2_second = opendir(src_second);
    count = 0;
    i = 0;
    unsigned long long imageDimMax = (unsigned long long)maxHeight * (unsigned long long)maxWidth * (unsigned long long)ip_channel;
    Rpp32u elementsInRowMax = maxWidth * ip_channel;

    while ((de = readdir(dr2)) != NULL)
    {
        Rpp8u *input_temp, *input_second_temp;
        input_temp = input + (i * imageDimMax);
        input_second_temp = input_second + (i * imageDimMax);
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
        Rpp32u elementsInRow = srcSize[i].width * ip_channel;
        for (j = 0; j < srcSize[i].height; j++)
        {
            memcpy(input_temp, ip_image, elementsInRow * sizeof (Rpp8u));
            memcpy(input_second_temp, ip_image_second, elementsInRow * sizeof (Rpp8u));
            ip_image += elementsInRow;
            ip_image_second += elementsInRow;
            input_temp += elementsInRowMax;
            input_second_temp += elementsInRowMax;
        }
        i++;
        count += imageDimMax;
    }
    closedir(dr2);

    Rpp16f *inputf16, *inputf16_second, *outputf16;
    Rpp32f *inputf32, *inputf32_second, *outputf32;
    Rpp8s *inputi8, *inputi8_second, *outputi8;

    cl_mem d_input, d_input_second, d_inputf16, d_inputf16_second, d_inputf32, d_inputf32_second, d_inputi8, d_inputi8_second;
    cl_mem d_output, d_outputf16, d_outputf32, d_outputi8;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context theContext;
    cl_command_queue theQueue;
    cl_int err;
    err = clGetPlatformIDs(1, &platform_id, NULL);
    err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    theContext = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    theQueue = clCreateCommandQueueWithProperties(theContext, device_id, 0, &err);

    if (ip_bitDepth == 0)
    {
        d_input = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
        d_input_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
        d_output = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp8u), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_input, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_input_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input_second, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_output, CL_TRUE, 0, oBufferSize * sizeof(Rpp8u), output, 0, NULL, NULL);
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

        d_inputf16 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp16f), NULL, NULL);
        d_inputf16_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp16f), NULL, NULL);
        d_outputf16 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp16f), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_inputf16, CL_TRUE, 0, ioBufferSize * sizeof(Rpp16f), inputf16, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_inputf16_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp16f), inputf16_second, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_outputf16, CL_TRUE, 0, oBufferSize * sizeof(Rpp16f), outputf16, 0, NULL, NULL);
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

        d_inputf32 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp32f), NULL, NULL);
        d_inputf32_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp32f), NULL, NULL);
        d_outputf32 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp32f), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_inputf32, CL_TRUE, 0, ioBufferSize * sizeof(Rpp32f), inputf32, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_inputf32_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp32f), inputf32_second, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_outputf32, CL_TRUE, 0, oBufferSize * sizeof(Rpp32f), outputf32, 0, NULL, NULL);
    }
    else if (ip_bitDepth == 3)
    {
        outputf16 = (Rpp16f *)calloc(oBufferSize, sizeof(Rpp16f));
        d_input = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
        d_input_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
        d_outputf16 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp16f), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_input, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_input_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input_second, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_outputf16, CL_TRUE, 0, oBufferSize * sizeof(Rpp16f), outputf16, 0, NULL, NULL);
    }
    else if (ip_bitDepth == 4)
    {
        outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));
        d_input = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
        d_input_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
        d_outputf32 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp32f), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_input, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_input_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input_second, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_outputf32, CL_TRUE, 0, oBufferSize * sizeof(Rpp32f), outputf32, 0, NULL, NULL);
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

        d_inputi8 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8s), NULL, NULL);
        d_inputi8_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8s), NULL, NULL);
        d_outputi8 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp8s), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_inputi8, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8s), inputi8, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_inputi8_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8s), inputi8_second, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_outputi8, CL_TRUE, 0, oBufferSize * sizeof(Rpp8s), outputi8, 0, NULL, NULL);
    }
    else if (ip_bitDepth == 6)
    {
        outputi8 = (Rpp8s *)calloc(oBufferSize, sizeof(Rpp8s));
        d_input = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
        d_input_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
        d_outputi8 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp8s), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_input, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_input_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input_second, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_outputi8, CL_TRUE, 0, oBufferSize * sizeof(Rpp8s), outputi8, 0, NULL, NULL);
    }

    rppHandle_t handle;

    rppCreateWithStreamAndBatchSize(&handle, theQueue, noOfImages);

    clock_t start, end;
    double max_time_used = 0, min_time_used = 500, avg_time_used = 0;

    string test_case_name;

    for (int perfRunCount = 0; perfRunCount < 100; perfRunCount++)
    {
        double gpu_time_used;
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

            start = clock();

            if (ip_bitDepth == 0)
                rppi_brightness_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, alpha, beta, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 1:
        {
            test_case_name = "gamma_correction";

            Rpp32f gamma[images];
            for (i = 0; i < images; i++)
            {
                gamma[i] = 1.9;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_gamma_correction_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, gamma, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 2:
        {
            test_case_name = "blend";

            Rpp32f alpha[images];
            for (i = 0; i < images; i++)
            {
                alpha[i] = 0.5;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_blend_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, alpha, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 3:
        {
            test_case_name = "blur";

            Rpp32u kernelSize[images];
            for (i = 0; i < images; i++)
            {
                kernelSize[i] = 5;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_blur_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 4:
        {
            test_case_name = "contrast";

            Rpp32u newMin[images];
            Rpp32u newMax[images];
            for (i = 0; i < images; i++)
            {
                newMin[i] = 30;
                newMax[i] = 100;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_contrast_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, newMin, newMax, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 5:
        {
            test_case_name = "pixelate";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_pixelate_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 6:
        {
            test_case_name = "jitter";

            Rpp32u kernelSize[images];
            for (i = 0; i < images; i++)
            {
                kernelSize[i] = 5;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_jitter_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 7:
        {
            test_case_name = "snow";

            Rpp32f snowPercentage[images];
            for (i = 0; i < images; i++)
            {
                snowPercentage[i] = 0.6;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_snow_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, snowPercentage, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 8:
        {
            test_case_name = "noise";

            Rpp32f noiseProbability[images];
            for (i = 0; i < images; i++)
            {
                noiseProbability[i] = 0.2;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_noise_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, noiseProbability, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 9:
        {
            test_case_name = "random_shadow";

            Rpp32u x1[images];
            Rpp32u y1[images];
            Rpp32u x2[images];
            Rpp32u y2[images];
            Rpp32u numbeoOfShadows[images];
            Rpp32u maxSizeX[images];
            Rpp32u maxSizey[images];
            for (i = 0; i < images; i++)
            {
                x1[i] = 0;
                y1[i] = 0;
                x2[i] = 100;
                y2[i] = 100;
                numbeoOfShadows[i] = 10;
                maxSizeX[i] = 12;
                maxSizey[i] = 15;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_random_shadow_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, x1, y1, x2, y2, numbeoOfShadows, maxSizeX, maxSizey, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 10:
        {
            test_case_name = "fog";

            Rpp32f fogValue[images];
            for (i = 0; i < images; i++)
            {
                fogValue[i] = 0.2;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_fog_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, fogValue, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 11:
        {
            test_case_name = "rain";

            Rpp32f rainPercentage[images];
            Rpp32u rainWidth[images];
            Rpp32u rainHeight[images];
            Rpp32f transparency[images];
            for (i = 0; i < images; i++)
            {
                rainPercentage[i] = 0.75;
                rainWidth[i] = 1;
                rainHeight[i] = 12;
                transparency[i] = 0.3;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_rain_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, rainPercentage, rainWidth, rainHeight, transparency, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 12:
        {
            test_case_name = "random_crop_letterbox";

            Rpp32u x1[images];
            Rpp32u y1[images];
            Rpp32u x2[images];
            Rpp32u y2[images];
            for (i = 0; i < images; i++)
            {
                x1[i] = 0;
                y1[i] = 0;
                x2[i] = 99;
                y2[i] = 99;
                dstSize[i].height = 150;
                dstSize[i].width = 150;
                if (maxDstHeight < dstSize[i].height)
                    maxDstHeight = dstSize[i].height;
                if (maxDstWidth < dstSize[i].width)
                    maxDstWidth = dstSize[i].width;
                if (minDstHeight > dstSize[i].height)
                    minDstHeight = dstSize[i].height;
                if (minDstWidth > dstSize[i].width)
                    minDstWidth = dstSize[i].width;
            }
            maxDstSize.height = maxDstHeight;
            maxDstSize.width = maxDstWidth;

            start = clock();

            if (ip_bitDepth == 0)
                rppi_random_crop_letterbox_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, x1, x2, y1, y2, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

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

            start = clock();

            if (ip_bitDepth == 0)
                rppi_exposure_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, exposureFactor, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 14:
        {
            test_case_name = "histogram_balance";
            missingFuncFlag = 1;

            break;
        }
        case 15:
        {
            test_case_name = "thresholding";

            Rpp8u min[images];
            Rpp8u max[images];
            for (i = 0; i < images; i++)
            {
                min[i] = 30;
                max[i] = 100;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_thresholding_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, min, max, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 16:
        {
            test_case_name = "min";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_min_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 17:
        {
            test_case_name = "max";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_max_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 18:
        {
            test_case_name = "integral";
            missingFuncFlag = 1;

            break;
        }
        case 19:
        {
            test_case_name = "histogram_equalization";
            missingFuncFlag = 1;

            break;
        }
        case 20:
        {
            test_case_name = "flip";

            Rpp32u flipAxis[images];
            for (i = 0; i < images; i++)
            {
                flipAxis[i] = 1;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_flip_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, flipAxis, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 21:
        {
            test_case_name = "resize";

            for (i = 0; i < images; i++)
            {
                dstSize[i].height = image.rows / 3;
                dstSize[i].width = image.cols / 1.1;
                if (maxDstHeight < dstSize[i].height)
                    maxDstHeight = dstSize[i].height;
                if (maxDstWidth < dstSize[i].width)
                    maxDstWidth = dstSize[i].width;
                if (minDstHeight > dstSize[i].height)
                    minDstHeight = dstSize[i].height;
                if (minDstWidth > dstSize[i].width)
                    minDstWidth = dstSize[i].width;
            }
            maxDstSize.height = maxDstHeight;
            maxDstSize.width = maxDstWidth;

            start = clock();

            if (ip_bitDepth == 0)
                rppi_resize_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 1)
                rppi_resize_f16_pln1_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 2)
                rppi_resize_f32_pln1_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 3)
                rppi_resize_u8_f16_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 4)
                rppi_resize_u8_f32_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 5)
                rppi_resize_i8_pln1_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 6)
                rppi_resize_u8_i8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 22:
        {
            test_case_name = "resize_crop";

            Rpp32u x1[images];
            Rpp32u y1[images];
            Rpp32u x2[images];
            Rpp32u y2[images];
            for (i = 0; i < images; i++)
            {
                x1[i] = 0;
                y1[i] = 0;
                x2[i] = 50;
                y2[i] = 50;
                dstSize[i].height = image.rows / 3;
                dstSize[i].width = image.cols / 1.1;
                if (maxDstHeight < dstSize[i].height)
                    maxDstHeight = dstSize[i].height;
                if (maxDstWidth < dstSize[i].width)
                    maxDstWidth = dstSize[i].width;
                if (minDstHeight > dstSize[i].height)
                    minDstHeight = dstSize[i].height;
                if (minDstWidth > dstSize[i].width)
                    minDstWidth = dstSize[i].width;
            }
            maxDstSize.height = maxDstHeight;
            maxDstSize.width = maxDstWidth;

            start = clock();

            if (ip_bitDepth == 0)
                rppi_resize_crop_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, x1, x2, y1, y2, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 1)
                rppi_resize_crop_f16_pln1_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, x1, x2, y1, y2, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 2)
                rppi_resize_crop_f32_pln1_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, x1, x2, y1, y2, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppi_resize_crop_i8_pln1_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, x1, x2, y1, y2, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 23:
        {
            test_case_name = "rotate";

            Rpp32f angle[images];
            for (i = 0; i < images; i++)
            {
                angle[i] = 50;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_rotate_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, angle, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 1)
                rppi_rotate_f16_pln1_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, angle, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 2)
                rppi_rotate_f32_pln1_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, angle, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppi_rotate_i8_pln1_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, angle, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 24:
        {
            test_case_name = "warp_affine";

            Rpp32f affine_array[6 * images];
            for (i = 0; i < 6 * images; i = i + 6)
            {
                affine_array[i] = 1.23;
                affine_array[i + 1] = 0.5;
                affine_array[i + 2] = 0.0;
                affine_array[i + 3] = -0.8;
                affine_array[i + 4] = 0.83;
                affine_array[i + 5] = 0.0;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_warp_affine_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, affine_array, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 1)
                rppi_warp_affine_f16_pln1_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, affine_array, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 2)
                rppi_warp_affine_f32_pln1_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, affine_array, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppi_warp_affine_i8_pln1_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, affine_array, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 25:
        {
            test_case_name = "fisheye";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_fisheye_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 26:
        {
            test_case_name = "lens_correction";

            Rpp32f strength[images];
            Rpp32f zoom[images];
            for (i = 0; i < images; i++)
            {
                strength[i] = 0.8;
                zoom[i] = 1;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_lens_correction_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, strength, zoom, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 27:
        {
            test_case_name = "scale";

            Rpp32f percentage[images];
            for (i = 0; i < images; i++)
            {
                percentage[i] = 75;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_scale_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, percentage, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 28:
        {
            test_case_name = "warp_perspective";

            const int size_perspective = images * 9;
            Rpp32f perspective[64 * 9];
            for (i = 0; i < images; i++)
            {
                perspective[0 + i * 9] = 0.93;
                perspective[1 + i * 9] = 0.5;
                perspective[2 + i * 9] = 0.0;
                perspective[3 + i * 9] = -0.5;
                perspective[4 + i * 9] = 0.93;
                perspective[5 + i * 9] = 0.0;
                perspective[6 + i * 9] = 0.005;
                perspective[7 + i * 9] = 0.005;
                perspective[8 + i * 9] = 1;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_warp_perspective_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, perspective, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 29:
        {
            test_case_name = "water";

            Rpp32f ampl_x[images];
            Rpp32f ampl_y[images];
            Rpp32f freq_x[images];
            Rpp32f freq_y[images];
            Rpp32f phase_x[images];
            Rpp32f phase_y[images];

            for (i = 0; i < images; i++)
            {
                ampl_x[i] = 2.0;
                ampl_y[i] = 5.0;
                freq_x[i] = 5.8;
                freq_y[i] = 1.2;
                phase_x[i] = 10.0;
                phase_y[i] = 15;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_water_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 1)
                rppi_water_f16_pln1_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 2)
                rppi_water_f32_pln1_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppi_water_i8_pln1_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 30:
        {
            test_case_name = "non_linear_blend";

            Rpp32f std_dev[images];
            for (i = 0; i < images; i++)
            {
                std_dev[i] = 50.0;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_non_linear_blend_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, std_dev, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 1)
                rppi_non_linear_blend_f16_pln1_batchPD_gpu(d_inputf16, d_inputf16_second, srcSize, maxSize, d_outputf16, std_dev, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 2)
                rppi_non_linear_blend_f32_pln1_batchPD_gpu(d_inputf32, d_inputf32_second, srcSize, maxSize, d_outputf32, std_dev, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppi_non_linear_blend_i8_pln1_batchPD_gpu(d_inputi8, d_inputi8_second, srcSize, maxSize, d_outputi8, std_dev, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 31:
        {
            test_case_name = "color_cast";
            missingFuncFlag = 1;

            break;
        }
        // case 32:
        // {
        //     test_case_name = "erase";

        //     Rpp32u boxesInEachImage = 3;

        //     Rpp32u anchor_box_info[images * boxesInEachImage * 4];
        //     Rpp32u box_offset[images];
        //     Rpp32u num_of_boxes[images];
        //     Rpp8u colorsu8[images * boxesInEachImage];
        //     Rpp32f colorsf32[images * boxesInEachImage];
        //     Rpp16f colorsf16[images * boxesInEachImage];
        //     Rpp8s colorsi8[images * boxesInEachImage];

        //     for (i = 0; i < images; i++)
        //     {
        //         box_offset[i] = i * boxesInEachImage;
        //         num_of_boxes[i] = boxesInEachImage;

        //         anchor_box_info[(boxesInEachImage * 4 * i)] = 0.125 * srcSize[i].width;
        //         anchor_box_info[(boxesInEachImage * 4 * i) + 1] = 0.125 * srcSize[i].height;
        //         anchor_box_info[(boxesInEachImage * 4 * i) + 2] = 0.375 * srcSize[i].width;
        //         anchor_box_info[(boxesInEachImage * 4 * i) + 3] = 0.375 * srcSize[i].height;

        //         anchor_box_info[(boxesInEachImage * 4 * i) + 4] = 0.125 * srcSize[i].width;
        //         anchor_box_info[(boxesInEachImage * 4 * i) + 5] = 0.625 * srcSize[i].height;
        //         anchor_box_info[(boxesInEachImage * 4 * i) + 6] = 0.875 * srcSize[i].width;
        //         anchor_box_info[(boxesInEachImage * 4 * i) + 7] = 0.875 * srcSize[i].height;

        //         anchor_box_info[(boxesInEachImage * 4 * i) + 8] = 0.75 * srcSize[i].width;
        //         anchor_box_info[(boxesInEachImage * 4 * i) + 9] = 0.125 * srcSize[i].height;
        //         anchor_box_info[(boxesInEachImage * 4 * i) + 10] = 0.875 * srcSize[i].width;
        //         anchor_box_info[(boxesInEachImage * 4 * i) + 11] = 0.5 * srcSize[i].height;

        //         colorsu8[(boxesInEachImage * i)] = (Rpp8u) 240;
        //         colorsu8[(boxesInEachImage * i) + 1] = (Rpp8u) 120;
        //         colorsu8[(boxesInEachImage * i) + 2] = (Rpp8u) 60;

        //         colorsf32[(boxesInEachImage * i)] = (Rpp32f) (240.0 / 255.0);
        //         colorsf32[(boxesInEachImage * i) + 1] = (Rpp32f) (120.0 / 255.0);
        //         colorsf32[(boxesInEachImage * i) + 2] = (Rpp32f) (60.0 / 255.0);

        //         colorsf16[(boxesInEachImage * i)] = (Rpp16f) (240.0 / 255.0);
        //         colorsf16[(boxesInEachImage * i) + 1] = (Rpp16f) (120.0 / 255.0);
        //         colorsf16[(boxesInEachImage * i) + 2] = (Rpp16f) (60.0 / 255.0);

        //         colorsi8[(boxesInEachImage * i)] = (Rpp8s) (240 - 128);
        //         colorsi8[(boxesInEachImage * i) + 1] = (Rpp8s) (120 - 128);
        //         colorsi8[(boxesInEachImage * i) + 2] = (Rpp8s) (60 - 128);
        //     }

        //     cl_mem d_anchor_box_info, d_box_offset, d_colorsu8, d_colorsf16, d_colorsf32, d_colorsi8;
        //     d_anchor_box_info = clCreateBuffer(theContext, CL_MEM_READ_ONLY,  images * boxesInEachImage * 4 * sizeof(Rpp32u), NULL, NULL);
        //     d_box_offset = clCreateBuffer(theContext, CL_MEM_READ_ONLY,  images * sizeof(Rpp32u), NULL, NULL);
        //     d_colorsu8 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, images * boxesInEachImage * 3 * sizeof(Rpp8u), NULL, NULL);
        //     d_colorsf16 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, images * boxesInEachImage * 3 * sizeof(Rpp16f), NULL, NULL);
        //     d_colorsf32 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, images * boxesInEachImage * 3 * sizeof(Rpp32f), NULL, NULL);
        //     d_colorsi8 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, images * boxesInEachImage * 3 * sizeof(Rpp8s), NULL, NULL);
        //     err |= clEnqueueWriteBuffer(theQueue, d_anchor_box_info, CL_TRUE, 0, images * boxesInEachImage * 4 * sizeof(Rpp32u), anchor_box_info, 0, NULL, NULL);
        //     err |= clEnqueueWriteBuffer(theQueue, d_box_offset, CL_TRUE, 0, images * sizeof(Rpp32u), box_offset, 0, NULL, NULL);
        //     err |= clEnqueueWriteBuffer(theQueue, d_colorsu8, CL_TRUE, 0, images * boxesInEachImage * 3 * sizeof(Rpp8u), d_colorsu8, 0, NULL, NULL);
        //     err |= clEnqueueWriteBuffer(theQueue, d_colorsf16, CL_TRUE, 0, images * boxesInEachImage * 3 * sizeof(Rpp8u), d_colorsf16, 0, NULL, NULL);
        //     err |= clEnqueueWriteBuffer(theQueue, d_colorsf32, CL_TRUE, 0, images * boxesInEachImage * 3 * sizeof(Rpp8u), d_colorsf32, 0, NULL, NULL);
        //     err |= clEnqueueWriteBuffer(theQueue, d_colorsi8, CL_TRUE, 0, images * boxesInEachImage * 3 * sizeof(Rpp8u), d_colorsi8, 0, NULL, NULL);

        //     start = clock();

        //     if (ip_bitDepth == 0)
        //         rppi_erase_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, d_anchor_box_info, d_colorsu8, d_box_offset, num_of_boxes, outputFormatToggle, noOfImages, handle);
        //     else if (ip_bitDepth == 1)
        //         rppi_erase_f16_pln1_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, d_anchor_box_info, d_colorsf16, d_box_offset, num_of_boxes, outputFormatToggle, noOfImages, handle);
        //     else if (ip_bitDepth == 2)
        //         rppi_erase_f32_pln1_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, d_anchor_box_info, d_colorsf32, d_box_offset, num_of_boxes, outputFormatToggle, noOfImages, handle);
        //     else if (ip_bitDepth == 3)
        //         missingFuncFlag = 1;
        //     else if (ip_bitDepth == 4)
        //         missingFuncFlag = 1;
        //     else if (ip_bitDepth == 5)
        //         rppi_erase_i8_pln1_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, d_anchor_box_info, d_colorsi8, d_box_offset, num_of_boxes, outputFormatToggle, noOfImages, handle);
        //     else if (ip_bitDepth == 6)
        //         missingFuncFlag = 1;
        //     else
        //         missingFuncFlag = 1;

        //     end = clock();

        //     clReleaseMemObject(d_anchor_box_info);
        //     clReleaseMemObject(d_box_offset);
        //     clReleaseMemObject(d_colorsu8);
        //     clReleaseMemObject(d_colorsf16);
        //     clReleaseMemObject(d_colorsf32);
        //     clReleaseMemObject(d_colorsi8);

        //     break;
        // }
        case 33:
        {
            test_case_name = "crop_and_patch";

            Rpp32u x11[images];
            Rpp32u y11[images];
            Rpp32u x12[images];
            Rpp32u y12[images];
            Rpp32u x21[images];
            Rpp32u y21[images];
            Rpp32u x22[images];
            Rpp32u y22[images];
            for (i = 0; i < images; i++)
            {
                x11[i] = (Rpp32u) (((Rpp32f) srcSize[i].width) * 0.25);
                y11[i] = (Rpp32u) (((Rpp32f) srcSize[i].height) * 0.25);
                x12[i] = (Rpp32u) (((Rpp32f) srcSize[i].width) * 0.5);
                y12[i] = (Rpp32u) (((Rpp32f) srcSize[i].height) * 0.5);

                x21[i] = (Rpp32u) (((Rpp32f) srcSize[i].width) * 0.5);
                y21[i] = (Rpp32u) (((Rpp32f) srcSize[i].height) * 0.5);
                x22[i] = (Rpp32u) (((Rpp32f) srcSize[i].width) * 0.75);
                y22[i] = (Rpp32u) (((Rpp32f) srcSize[i].height) * 0.75);
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_crop_and_patch_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, x11, y11, x12, y12, x21, y21, x22, y22, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 1)
                rppi_crop_and_patch_f16_pln1_batchPD_gpu(d_inputf16, d_inputf16_second, srcSize, maxSize, d_outputf16, x11, y11, x12, y12, x21, y21, x22, y22, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 2)
                rppi_crop_and_patch_f32_pln1_batchPD_gpu(d_inputf32, d_inputf32_second, srcSize, maxSize, d_outputf32, x11, y11, x12, y12, x21, y21, x22, y22, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppi_crop_and_patch_i8_pln1_batchPD_gpu(d_inputi8, d_inputi8_second, srcSize, maxSize, d_outputi8, x11, y11, x12, y12, x21, y21, x22, y22, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 34:
        {
            test_case_name = "lut";

            Rpp8u lut8u[images * 256];
            Rpp8s lut8s[images * 256];

            for (i = 0; i < images; i++)
            {
                for (j = 0; j < 256; j++)
                {
                    lut8u[(i * 256) + j] = (Rpp8u)(255 - j);
                    lut8s[(i * 256) + j] = (Rpp8u)(255 - j - 128);
                }
            }

            cl_mem d_lut8u, d_lut8s;
            d_lut8u = clCreateBuffer(theContext, CL_MEM_READ_ONLY, images * 256 * sizeof(Rpp8u), NULL, NULL);
            d_lut8s = clCreateBuffer(theContext, CL_MEM_READ_ONLY, images * 256 * sizeof(Rpp8s), NULL, NULL);
            err |= clEnqueueWriteBuffer(theQueue, d_lut8u, CL_TRUE, 0, images * 256 * sizeof(Rpp8u), lut8u, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(theQueue, d_lut8s, CL_TRUE, 0, images * 256 * sizeof(Rpp8s), lut8s, 0, NULL, NULL);

            start = clock();

            if (ip_bitDepth == 0)
                rppi_lut_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, d_lut8u, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppi_lut_i8_pln1_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, d_lut8s, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            clReleaseMemObject(d_lut8u);
            clReleaseMemObject(d_lut8s);

            break;
        }
        case 35:
        {
            test_case_name = "glitch";
            missingFuncFlag = 1;

            break;
        }
        case 36:
        {
            test_case_name = "color_twist";
            missingFuncFlag = 1;

            break;
        }
        case 37:
        {
            test_case_name = "crop";

            Rpp32u crop_pos_x[images];
            Rpp32u crop_pos_y[images];
            for (i = 0; i < images; i++)
            {
                dstSize[i].height = 100;
                dstSize[i].width = 100;
                if (maxDstHeight < dstSize[i].height)
                    maxDstHeight = dstSize[i].height;
                if (maxDstWidth < dstSize[i].width)
                    maxDstWidth = dstSize[i].width;
                if (minDstHeight > dstSize[i].height)
                    minDstHeight = dstSize[i].height;
                if (minDstWidth > dstSize[i].width)
                    minDstWidth = dstSize[i].width;
                crop_pos_x[i] = 50;
                crop_pos_y[i] = 50;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_crop_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 1)
                rppi_crop_f16_pln1_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 2)
                rppi_crop_f32_pln1_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 3)
                rppi_crop_u8_f16_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 4)
                rppi_crop_u8_f32_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 5)
                rppi_crop_i8_pln1_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 6)
                rppi_crop_u8_i8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, noOfImages, handle);
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 38:
        {
            test_case_name = "crop_mirror_normalize";

            Rpp32u crop_pos_x[images];
            Rpp32u crop_pos_y[images];
            Rpp32f mean[images];
            Rpp32f stdDev[images];
            Rpp32u mirrorFlag[images];
            for (i = 0; i < images; i++)
            {
                dstSize[i].height = 100;
                dstSize[i].width = 100;
                if (maxDstHeight < dstSize[i].height)
                    maxDstHeight = dstSize[i].height;
                if (maxDstWidth < dstSize[i].width)
                    maxDstWidth = dstSize[i].width;
                if (minDstHeight > dstSize[i].height)
                    minDstHeight = dstSize[i].height;
                if (minDstWidth > dstSize[i].width)
                    minDstWidth = dstSize[i].width;
                crop_pos_x[i] = 50;
                crop_pos_y[i] = 50;
                mean[i] = 0.0;
                stdDev[i] = 1.0;
                mirrorFlag[i] = 1;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_crop_mirror_normalize_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 1)
                rppi_crop_mirror_normalize_f16_pln1_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 2)
                rppi_crop_mirror_normalize_f32_pln1_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 3)
                rppi_crop_mirror_normalize_u8_f16_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 4)
                rppi_crop_mirror_normalize_u8_f32_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 5)
                rppi_crop_mirror_normalize_i8_pln1_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 6)
                rppi_crop_mirror_normalize_u8_i8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, noOfImages, handle);
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 39:
        {
            test_case_name = "resize_crop_mirror";

            Rpp32u x1[images];
            Rpp32u y1[images];
            Rpp32u x2[images];
            Rpp32u y2[images];
            Rpp32u mirrorFlag[images];
            for (i = 0; i < images; i++)
            {
                x1[i] = 0;
                y1[i] = 0;
                x2[i] = 50;
                y2[i] = 50;
                dstSize[i].height = image.rows / 3;
                dstSize[i].width = image.cols / 1.1;
                if (maxDstHeight < dstSize[i].height)
                    maxDstHeight = dstSize[i].height;
                if (maxDstWidth < dstSize[i].width)
                    maxDstWidth = dstSize[i].width;
                if (minDstHeight > dstSize[i].height)
                    minDstHeight = dstSize[i].height;
                if (minDstWidth > dstSize[i].width)
                    minDstWidth = dstSize[i].width;
                mirrorFlag[i] = 1;
            }
            maxDstSize.height = maxDstHeight;
            maxDstSize.width = maxDstWidth;

            start = clock();

            if (ip_bitDepth == 0)
                rppi_resize_crop_mirror_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, x1, x2, y1, y2, mirrorFlag, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 1)
                rppi_resize_crop_mirror_f16_pln1_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, x1, x2, y1, y2, mirrorFlag, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 2)
                rppi_resize_crop_mirror_f32_pln1_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, x1, x2, y1, y2, mirrorFlag, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppi_resize_crop_mirror_i8_pln1_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, x1, x2, y1, y2, mirrorFlag, outputFormatToggle, noOfImages, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 40:
        {
            test_case_name = "erode";

            Rpp32u kernelSize[images];
            for (i = 0; i < images; i++)
            {
                kernelSize[i] = 5;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_erode_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 41:
        {
            test_case_name = "dilate";

            Rpp32u kernelSize[images];
            for (i = 0; i < images; i++)
            {
                kernelSize[i] = 5;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_dilate_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 42:
        {
            test_case_name = "hueRGB";
            missingFuncFlag = 1;

            break;
        }
        case 43:
        {
            test_case_name = "saturationRGB";
            missingFuncFlag = 1;

            break;
        }
        case 44:
        {
            test_case_name = "color_convert";
            missingFuncFlag = 1;

            break;
        }
        case 45:
        {
            test_case_name = "color_temperature";

            Rpp32s adjustmentValue[images];
            for (i = 0; i < images; i++)
            {
                adjustmentValue[i] = 70;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_color_temperature_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, adjustmentValue, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 46:
        {
            test_case_name = "vignette";

            Rpp32f stdDev[images];
            for (i = 0; i < images; i++)
            {
                stdDev[i] = 75.0;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_vignette_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, stdDev, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 47:
        {
            test_case_name = "channel_combine and channel_extract";
            missingFuncFlag = 1;

            break;
        }
        case 48:
        {
            test_case_name = "look_up_table";

            Rpp8u lookUpTableU8Pln[images * ip_channel * 256];
            Rpp8u *lookUpTableU8PlnTemp;
            lookUpTableU8PlnTemp = lookUpTableU8Pln;

            for (i = 0; i < images; i++)
            {
                for (int c = 0; c < ip_channel; c++)
                {
                    for (j = 0; j < 256; j++)
                    {
                        if (c == 0)
                            *lookUpTableU8PlnTemp = (Rpp8u)(255 - j);
                        else
                            *lookUpTableU8PlnTemp = (Rpp8u)(j);
                        lookUpTableU8PlnTemp++;
                    }
                }
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_look_up_table_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, lookUpTableU8Pln, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 49:
        {
            test_case_name = "box_filter";

            Rpp32u kernelSize[images];
            for (i = 0; i < images; i++)
            {
                kernelSize[i] = 5;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_box_filter_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 50:
        {
            test_case_name = "sobel_filter";

            Rpp32u sobelType[images];
            for (i = 0; i < images; i++)
            {
                sobelType[i] = 1;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_sobel_filter_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, sobelType, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 51:
        {
            test_case_name = "median_filter";

            Rpp32u kernelSize[images];
            for (i = 0; i < images; i++)
            {
                kernelSize[i] = 5;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_median_filter_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 52:
        {
            test_case_name = "custom_convolution";
            missingFuncFlag = 1;

            break;
        }
        case 53:
        {
            test_case_name = "non_max_suppression";

            Rpp32u kernelSize[images];
            for (i = 0; i < images; i++)
            {
                kernelSize[i] = 5;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_non_max_suppression_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 54:
        {
            test_case_name = "gaussian_filter";

            Rpp32u kernelSize[images];
            Rpp32f stdDev[images];
            for (i = 0; i < images; i++)
            {
                kernelSize[i] = 5;
                stdDev[i] = 5.0;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_gaussian_filter_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, stdDev, kernelSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 55:
        {
            test_case_name = "nonlinear_filter";

            Rpp32u kernelSize[images];
            for (i = 0; i < images; i++)
            {
                kernelSize[i] = 5;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_nonlinear_filter_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 56:
        {
            test_case_name = "absolute_difference";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_absolute_difference_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 57:
        {
            test_case_name = "accumulate_weighted";

            Rpp32f alpha[images];
            for (i = 0; i < images; i++)
            {
                alpha[i] = 0.5;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_accumulate_weighted_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, alpha, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            err |= clEnqueueCopyBuffer(theQueue, d_input, d_output, 0, 0, oBufferSize * sizeof(Rpp8u), 0, NULL, NULL);

            break;
        }
        case 58:
        {
            test_case_name = "accumulate";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_accumulate_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            err |= clEnqueueCopyBuffer(theQueue, d_input, d_output, 0, 0, oBufferSize * sizeof(Rpp8u), 0, NULL, NULL);

            break;
        }
        case 59:
        {
            test_case_name = "add";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_add_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 60:
        {
            test_case_name = "subtract";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_subtract_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 61:
        {
            test_case_name = "magnitude";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_magnitude_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 62:
        {
            test_case_name = "multiply";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_multiply_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 63:
        {
            test_case_name = "phase";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_phase_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 64:
        {
            test_case_name = "accumulate_squared";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_accumulate_squared_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            err |= clEnqueueCopyBuffer(theQueue, d_input, d_output, 0, 0, oBufferSize * sizeof(Rpp8u), 0, NULL, NULL);

            break;
        }
        case 65:
        {
            test_case_name = "bitwise_AND";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_bitwise_AND_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 66:
        {
            test_case_name = "bitwise_NOT";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_bitwise_NOT_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 67:
        {
            test_case_name = "exclusive_OR";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_exclusive_OR_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 68:
        {
            test_case_name = "inclusive_OR";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_inclusive_OR_u8_pln1_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 69:
        {
            test_case_name = "local_binary_pattern";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_local_binary_pattern_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 70:
        {
            test_case_name = "data_object_copy";

            start = clock();

            if (ip_bitDepth == 0)
                rppi_data_object_copy_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 71:
        {
            test_case_name = "gaussian_image_pyramid";

            Rpp32u kernelSize[images];
            Rpp32f stdDev[images];
            for (i = 0; i < images; i++)
            {
                kernelSize[i] = 5;
                stdDev[i] = 5.0;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_gaussian_image_pyramid_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, stdDev, kernelSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 72:
        {
            test_case_name = "laplacian_image_pyramid";

            Rpp32u kernelSize[images];
            Rpp32f stdDev[images];
            for (i = 0; i < images; i++)
            {
                kernelSize[i] = 5;
                stdDev[i] = 5.0;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_laplacian_image_pyramid_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, stdDev, kernelSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 73:
        {
            test_case_name = "canny_edge_detector";

            Rpp8u minThreshold[images];
            Rpp8u maxThreshold[images];
            for (i = 0; i < images; i++)
            {
                minThreshold[i] = 10;
                maxThreshold[i] = 30;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_canny_edge_detector_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, minThreshold, maxThreshold, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 74:
        {
            test_case_name = "harris_corner_detector";

            Rpp32u gaussianKernelSize[images];
            Rpp32f stdDev[images];
            Rpp32u kernelSize[images];
            Rpp32f kValue[images];
            Rpp32f threshold[images];
            Rpp32u nonmaxKernelSize[images];
            for (i = 0; i < images; i++)
            {
                gaussianKernelSize[i] = 3;
                stdDev[i] = 0.75;
                kernelSize[i] = 3;
                kValue[i] = 0.04;
                threshold[i] = 4000000000;
                nonmaxKernelSize[i] = 3;
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppi_harris_corner_detector_u8_pln1_batchPD_gpu(d_input, srcSize, maxSize, d_output, gaussianKernelSize, stdDev, kernelSize, kValue, threshold, nonmaxKernelSize, noOfImages, handle);
            else if (ip_bitDepth == 1)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 2)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            end = clock();

            break;
        }
        case 75:
        {
            test_case_name = "fast_corner_detector";
            missingFuncFlag = 1;

            break;
        }
        case 76:
        {
            test_case_name = "reconstruction_laplacian_image_pyramid";
            missingFuncFlag = 1;

            break;
        }
        case 77:
        {
            test_case_name = "hough_lines";
            missingFuncFlag = 1;

            break;
        }
        case 78:
        {
            test_case_name = "hog";
            missingFuncFlag = 1;

            break;
        }
        case 79:
        {
            test_case_name = "remap";
            missingFuncFlag = 1;

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

        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        if (gpu_time_used > max_time_used)
            max_time_used = gpu_time_used;
        if (gpu_time_used < min_time_used)
            min_time_used = gpu_time_used;
        avg_time_used += gpu_time_used;
    }

    avg_time_used /= 100;
    cout << fixed << "\nmax,min,avg = " << max_time_used << "," << min_time_used << "," << avg_time_used << endl;

    rppDestroyGPU(handle);

    free(srcSize);
    free(dstSize);
    free(input);
    free(input_second);
    free(output);

    if (ip_bitDepth == 0)
    {
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_input_second);
        clReleaseMemObject(d_output);
    }
    else if (ip_bitDepth == 1)
    {
        free(inputf16);
        free(inputf16_second);
        free(outputf16);
        clReleaseMemObject(d_inputf16);
        clReleaseMemObject(d_inputf16_second);
        clReleaseMemObject(d_outputf16);
    }
    else if (ip_bitDepth == 2)
    {
        free(inputf32);
        free(inputf32_second);
        free(outputf32);
        clReleaseMemObject(d_inputf32);
        clReleaseMemObject(d_inputf32_second);
        clReleaseMemObject(d_outputf32);
    }
    else if (ip_bitDepth == 3)
    {
        free(outputf16);
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_input_second);
        clReleaseMemObject(d_outputf16);
    }
    else if (ip_bitDepth == 4)
    {
        free(outputf32);
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_input_second);
        clReleaseMemObject(d_outputf32);
    }
    else if (ip_bitDepth == 5)
    {
        free(inputi8);
        free(inputi8_second);
        free(outputi8);
        clReleaseMemObject(d_inputi8);
        clReleaseMemObject(d_inputi8_second);
        clReleaseMemObject(d_outputi8);
    }
    else if (ip_bitDepth == 6)
    {
        free(outputi8);
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_input_second);
        clReleaseMemObject(d_outputi8);
    }

    return 0;
}
