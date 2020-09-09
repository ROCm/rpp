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
#include <CL/cl.hpp>
#include <half.hpp>
#include <fstream>

using namespace cv;
using namespace std;
using half_float::half;

typedef half Rpp16f;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))

int main(int argc, char **argv)
{
    const int MIN_ARG_COUNT = 7;
    printf("\nUsage: ./BatchPD_gpu <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 1:7>\n");
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        return -1;
    }

    printf("\nsrc1 = %s", argv[1]);
    printf("\nsrc2 = %s", argv[2]);
    printf("\ndst = %s", argv[3]);
    printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[4]);
    printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[5]);
    printf("\ncase number (1:7) = %s", argv[6]);

    char *src = argv[1];
    char *src_second = argv[2];
    char *dst = argv[3];
    int ip_bitDepth = atoi(argv[4]);
    unsigned int outputFormatToggle = atoi(argv[5]);
    int test_case = atoi(argv[6]);
    int ip_channel = 3;

    char funcType[1000] = {"BatchPD_GPU_PKD3"};

    if (outputFormatToggle == 0)
    {
        strcat(funcType, "_toPKD3");
    }
    else if (outputFormatToggle == 1)
    {
        strcat(funcType, "_toPLN3");
    }


    char funcName[1000];
    switch (test_case)
    {
    case 1:
        strcpy(funcName, "rotate");
        break;
    case 2:
        strcpy(funcName, "resize");
        break;
    case 3:
        strcpy(funcName, "resize_crop");
        break;
    case 4:
        strcpy(funcName, "resize_crop_mirror");
        break;
    case 5:
        strcpy(funcName, "crop");
        break;
    case 6:
        strcpy(funcName, "crop_mirror_normalize");
        break;
    case 7:
        strcpy(funcName, "color_twist");
        break;
    case 8:
        strcpy(funcName, "non_linear_blend");
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
    printf("\n\nRunning %s...", func);

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
    strcat(dst, "/");
    strcat(dst, funcName);

    DIR *dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfImages += 1;
    }
    closedir(dr);

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
        // if (ip_channel == 3)
        // {
            image = imread(temp, 1);
        // }
        // else
        // {
        //     image = imread(temp, 0);
        // }
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
                         std::cout << "comes here"<< src_second << std::endl;

    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(oBufferSize, sizeof(Rpp8u));

    Rpp16f *inputf16 = (Rpp16f *)calloc(ioBufferSize, sizeof(Rpp16f));
    Rpp16f *inputf16_second = (Rpp16f *)calloc(ioBufferSize, sizeof(Rpp16f));
    Rpp16f *outputf16 = (Rpp16f *)calloc(ioBufferSize, sizeof(Rpp16f));

    Rpp32f *inputf32 = (Rpp32f *)calloc(ioBufferSize, sizeof(Rpp32f));
    Rpp32f *inputf32_second = (Rpp32f *)calloc(ioBufferSize, sizeof(Rpp32f));
    Rpp32f *outputf32 = (Rpp32f *)calloc(ioBufferSize, sizeof(Rpp32f));

    Rpp8s *inputi8 = (Rpp8s *)calloc(ioBufferSize, sizeof(Rpp8s));
    Rpp8s *inputi8_second = (Rpp8s *)calloc(ioBufferSize, sizeof(Rpp8s));
    Rpp8s *outputi8 = (Rpp8s *)calloc(ioBufferSize, sizeof(Rpp8s));

    RppiSize maxSize, maxDstSize;
    maxSize.height = maxHeight;
    maxSize.width = maxWidth;
    maxDstSize.height = maxDstHeight;
    maxDstSize.width = maxDstWidth;

    DIR *dr2 = opendir(src);
    DIR *dr2_second = opendir(src_second);
    count = 0;
    i = 0;
    while ((de = readdir(dr2)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;

        count = (unsigned long long)i * (unsigned long long)maxHeight * (unsigned long long)maxWidth * (unsigned long long)ip_channel;

        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, de->d_name);

        char temp_second[1000];
        strcpy(temp_second, src1_second);
        strcat(temp_second, de->d_name);

        // if (ip_channel == 3)
        // {
            image = imread(temp, 1);
            image_second = imread(temp_second, 1);
        // }
        // else
        // {
        //     image = imread(temp, 0);
        //     image_second = imread(temp_second, 0);
        // }

        Rpp8u *ip_image = image.data;
        Rpp8u *ip_image_second = image_second.data;
        for (j = 0; j < srcSize[i].height; j++)
        {
            for (int x = 0; x < srcSize[i].width; x++)
            {
                for (int y = 0; y < ip_channel; y++)
                {
                    input[count + ((j * maxWidth * ip_channel) + (x * ip_channel) + y)] = ip_image[(j * srcSize[i].width * ip_channel) + (x * ip_channel) + y];
                    input_second[count + ((j * maxWidth * ip_channel) + (x * ip_channel) + y)] = ip_image_second[(j * srcSize[i].width * ip_channel) + (x * ip_channel) + y];
                }
            }
        }
        i++;
    }
    closedir(dr2);

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
            *inputi8Temp = (Rpp8s) (((Rpp32s) *inputTemp) - 128);
            *inputi8_secondTemp = (Rpp8s) (((Rpp32s) *input_secondTemp) - 128);
            inputTemp++;
            inputi8Temp++;
            input_secondTemp++;
            inputi8_secondTemp++;
        }
    }

    cl_mem d_input, d_input2, d_inputf16, d_inputf32,d_inputi8, d_input_second, d_output, d_outputf16, d_outputf32,d_outputi8;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context theContext;
    cl_command_queue theQueue;
    cl_int err;
    err = clGetPlatformIDs(1, &platform_id, NULL);
    err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    theContext = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    theQueue = clCreateCommandQueue(theContext, device_id, 0, &err);
    d_input = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
    d_input2 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
    d_inputf16 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp16f), NULL, NULL);
    d_inputf32 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp32f), NULL, NULL);
    d_inputi8 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8s), NULL, NULL);
    d_input_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
    d_output = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp8u), NULL, NULL);
    d_outputf16 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp16f), NULL, NULL);
    d_outputf32 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp32f), NULL, NULL);
    d_outputi8 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp8s), NULL, NULL);
    err |= clEnqueueWriteBuffer(theQueue, d_input, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(theQueue, d_inputf16, CL_TRUE, 0, ioBufferSize * sizeof(Rpp16f), inputf16, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(theQueue, d_inputf32, CL_TRUE, 0, ioBufferSize * sizeof(Rpp32f), inputf32, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(theQueue, d_inputi8, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8s), inputi8, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(theQueue, d_input_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input_second, 0, NULL, NULL);
    rppHandle_t handle;

    rppCreateWithStreamAndBatchSize(&handle, theQueue, noOfImages);

    clock_t start, end;
    double cpu_time_used;

    string test_case_name;

    switch (test_case)
    {
    case 1:
    {
        test_case_name = "rotate";

        Rpp32f angle[images];
        for (i = 0; i < images; i++)
        {
            angle[i] = 50;
        }

        start = clock();
        if (ip_bitDepth == 0)
            rppi_rotate_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, angle, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_rotate_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, angle,outputFormatToggle, noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_rotate_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, angle,outputFormatToggle, noOfImages, handle);
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
            rppi_rotate_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, angle,outputFormatToggle, noOfImages, handle);
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        end = clock();

        break;
    }
    case 2:
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
            rppi_resize_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize,outputFormatToggle, noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_resize_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_resize_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 3)
            rppi_resize_u8_f16_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 4)
            rppi_resize_u8_f32_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 5)
            rppi_resize_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 6)
            rppi_resize_u8_i8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, outputFormatToggle,noOfImages, handle);
        end = clock();

        break;
    }
    case 3:
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
            rppi_resize_crop_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, x1, x2, y1, y2,outputFormatToggle, noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_resize_crop_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, x1, x2, y1, y2, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_resize_crop_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, x1, x2, y1, y2, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
           rppi_resize_crop_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, x1, x2, y1, y2, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        end = clock();

        break;
    }
    case 4:
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
            rppi_resize_crop_mirror_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, x1, x2, y1, y2, mirrorFlag,outputFormatToggle, noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_resize_crop_mirror_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, x1, x2, y1, y2, mirrorFlag, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_resize_crop_mirror_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, x1, x2, y1, y2, mirrorFlag, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
            rppi_resize_crop_mirror_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, x1, x2, y1, y2, mirrorFlag, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        end = clock();

        break;
    }
    case 5:
    {
        test_case_name = "crop";

        Rpp32u crop_pos_x[images];
        Rpp32u crop_pos_y[images];
        for (i = 0; i < images; i++)
        {
            
            dstSize[i].height = 100;
            dstSize[i].width =  150; 
            if (maxDstHeight < dstSize[i].height)
                maxDstHeight = dstSize[i].height;
            if (maxDstWidth < dstSize[i].width)
                maxDstWidth = dstSize[i].width;
            if (minDstHeight > dstSize[i].height)
                minDstHeight = dstSize[i].height;
            if (minDstWidth > dstSize[i].width)
                minDstWidth = dstSize[i].width;
            cout << maxDstHeight << maxDstWidth << endl;
            crop_pos_x[i] = 100;
            crop_pos_y[i] = 150;
            
        }
        start = clock();
        if (ip_bitDepth == 0)
            rppi_crop_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, crop_pos_x, crop_pos_y,outputFormatToggle, noOfImages, handle);
        if (ip_bitDepth == 1)
            rppi_crop_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize,
                            crop_pos_x, crop_pos_y, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_crop_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize,
                    maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 3)
            rppi_crop_u8_f16_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_outputf16, dstSize,
                        maxDstSize, crop_pos_x, crop_pos_y,  outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 4)
            rppi_crop_u8_f32_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_outputf32, dstSize,
                        maxDstSize, crop_pos_x, crop_pos_y,outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 5)
            rppi_crop_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize,
                                 maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 6)
            rppi_crop_u8_i8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_outputi8, dstSize, maxDstSize,
                                    crop_pos_x, crop_pos_y, outputFormatToggle,noOfImages, handle);        
        end = clock();
        break;
    }
    case 6:
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
        //Rpp32u outputFormatToggle = 0;

        start = clock();
        if (ip_bitDepth == 0)
            rppi_crop_mirror_normalize_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_crop_mirror_normalize_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_crop_mirror_normalize_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag,  outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 3)
            rppi_crop_mirror_normalize_u8_f16_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag,outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 4)
            rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag,  outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 5)
            rppi_crop_mirror_normalize_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag,  outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 6)
            rppi_crop_mirror_normalize_u8_i8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag,  outputFormatToggle,noOfImages, handle);
        end = clock();

        break;
    }
    case 7:
    {
        test_case_name = "color_twist";

        Rpp32f alpha[images];
        Rpp32f beta[images];
        Rpp32f hueShift[images];
        Rpp32f saturationFactor[images];
        for (i = 0; i < images; i++)
        {
            alpha[i] = 1.4;
            beta[i] = 0;
            hueShift[i] = 60;
            saturationFactor[i] = 1.9;
        }

        start = clock();
        if (ip_bitDepth == 0)
            rppi_color_twist_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, alpha, beta, hueShift, saturationFactor, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_color_twist_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, alpha, beta, hueShift, saturationFactor, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_color_twist_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, alpha, beta, hueShift, saturationFactor, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
            rppi_color_twist_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, alpha, beta, hueShift, saturationFactor, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        end = clock();
        break;
    }
    case 8:
    {
        test_case_name = "non_linear_blend";

        Rpp32f alpha[images];
        Rpp32f beta[images];
        Rpp32f hueShift[images];
        Rpp32f saturationFactor[images];
        for (i = 0; i < images; i++)
        {
            alpha[i] = 1.4;
        }

        start = clock();
        if (ip_bitDepth == 0)
            rppi_non_linear_blend_u8_pkd3_batchPD_gpu(d_input, d_input2, srcSize, maxSize, d_output, alpha, outputFormatToggle,noOfImages, handle);
        // else if (ip_bitDepth == 1)
        //     rppi_color_twist_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, alpha, beta, hueShift, saturationFactor, outputFormatToggle,noOfImages, handle);
        // else if (ip_bitDepth == 2)
        //     rppi_color_twist_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, alpha, beta, hueShift, saturationFactor, outputFormatToggle,noOfImages, handle);
        // else if (ip_bitDepth == 3)
        //     missingFuncFlag = 1;
        // else if (ip_bitDepth == 4)
        //     missingFuncFlag = 1;
        // else if (ip_bitDepth == 5)
        //     rppi_color_twist_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, alpha, beta, hueShift, saturationFactor, outputFormatToggle,noOfImages, handle);
        // else if (ip_bitDepth == 6)
        //     missingFuncFlag = 1;
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

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    cout << "\nGPU Time - BatchPD : " << cpu_time_used;
    printf("\n");
    
    clEnqueueReadBuffer(theQueue, d_output, CL_TRUE, 0, oBufferSize * sizeof(Rpp8u), output, 0, NULL, NULL);
    clEnqueueReadBuffer(theQueue, d_outputf16, CL_TRUE, 0, oBufferSize * sizeof(Rpp16f), outputf16, 0, NULL, NULL);
    clEnqueueReadBuffer(theQueue, d_outputf32, CL_TRUE, 0, oBufferSize * sizeof(Rpp32f), outputf32, 0, NULL, NULL);
    clEnqueueReadBuffer(theQueue, d_outputi8, CL_TRUE, 0, oBufferSize * sizeof(Rpp8s), outputi8, 0, NULL, NULL);

    string fileName = std::to_string(ip_bitDepth);
    ofstream outputFile (fileName + ".csv");

    if (ip_bitDepth == 0)
    {
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
        Rpp8u *outputCopy = (Rpp8u *)calloc(oBufferSize, sizeof(Rpp8u));

     if(outputFormatToggle == 1)
    {
        for(int i = 0; i < noOfImages; i++)
        {
            for(int j = 0; j < maxDstHeight; j++)
            {
                for(int k = 0; k < maxDstWidth ; k++)
                {
                    for(int c = 0; c < ip_channel; c++)
                    {
                        int dstpix = i* maxDstHeight * maxDstWidth * ip_channel + j * maxDstWidth * ip_channel + k * ip_channel + c;
                        int srcpix = i* maxDstHeight * maxDstWidth * ip_channel + (j * maxDstWidth + k)  + c * maxDstHeight * maxDstWidth;
                        if(j == 0 && k == 0)
                            cout << dstpix << " " << srcpix << endl;
                        outputCopy[dstpix] = output[srcpix];
                    }
                }
            }
        }
        output = outputCopy;
    }

    rppDestroyGPU(handle);

    mkdir(dst, 0700);
    strcat(dst, "/");
    count = 0;
    for (j = 0; j < noOfImages; j++)
    {
        int op_size = maxHeight * maxWidth * ip_channel;
        Rpp8u *temp_output = (Rpp8u *)calloc(op_size, sizeof(Rpp8u));
        for (i = 0; i < op_size; i++)
        {
            temp_output[i] = output[count];
            count++;
        }
        char temp[1000];
        strcpy(temp, dst);
        strcat(temp, imageNames[j]);
        Mat mat_op_image;
        // if (ip_channel == 3)
        // {
            mat_op_image = Mat(maxHeight, maxWidth, CV_8UC3, temp_output);
            imwrite(temp, mat_op_image);
        // }
        // if (ip_channel == 1)
        // {
        //     mat_op_image = Mat(maxHeight, maxWidth, CV_8UC1, temp_output);
        //     imwrite(temp, mat_op_image);
        // }
        free(temp_output);
    }

    free(srcSize);
    free(dstSize);
    free(input);
    free(input_second);
    free(output);
    free(inputf16);
    free(inputf16_second);
    free(outputf16);
    free(inputf32);
    free(inputf32_second);
    free(inputi8);
    free(inputi8_second);
    free(outputf32);
    free(outputi8);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_input_second);
    clReleaseMemObject(d_inputf16);
    clReleaseMemObject(d_outputf16);
    clReleaseMemObject(d_inputf32);
    clReleaseMemObject(d_outputf32);
    clReleaseMemObject(d_inputi8);
    clReleaseMemObject(d_outputi8);
    return 0;

}
