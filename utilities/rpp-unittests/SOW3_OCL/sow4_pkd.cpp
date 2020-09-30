
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
    printf("\nUsage: ./BatchPD_gpu <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / i8 = 3> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 1:7>\n");
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        return -1;
    }

    printf("\nsrc1 = %s", argv[1]);
    printf("\nsrc2 = %s", argv[2]);
    printf("\ndst = %s", argv[3]);
    printf("\nu8 / f16 / f32 / i8 (0/1/2/3) = %s", argv[4]);
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
        strcpy(funcName, "non_linear_blend");
        break;
    case 2:
        strcpy(funcName, "Water");
        break;
    case 3:
        strcpy(funcName, "look");
        break;
    case 4:
        strcpy(funcName, "erase");
        break;
    case 5:
        strcpy(funcName, "color_cast");
        break;
    case 6:
        strcpy(funcName, "crop_and_patch");
        break;
    case 7:
        strcpy(funcName, "warp_affine");
        break;
    case 8:
        strcpy(funcName, "glitch");
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
        strcat(funcName, "_i8_");
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
        if (ip_channel == 3)
        {
            image = imread(temp, 1);
        }
        else
        {
            image = imread(temp, 0);
        }
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

        if (ip_channel == 3)
        {
            image = imread(temp, 1);
            image_second = imread(temp_second, 1);
        }
        else
        {
            image = imread(temp, 0);
            image_second = imread(temp_second, 0);
        }

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
    else if (ip_bitDepth == 3)
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

    cl_mem d_input, d_inputf16, d_inputf32,d_inputi8, d_input_second, d_inputf16_second,d_inputf32_second,d_inputi8_second,d_output, d_outputf16, d_outputf32,d_outputi8;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context theContext;
    cl_command_queue theQueue;
    cl_int err;
    err = clGetPlatformIDs(1, &platform_id, NULL);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    theContext = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    theQueue = clCreateCommandQueue(theContext, device_id, 0, &err);
    d_input = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
    d_inputf16 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp16f), NULL, NULL);
    d_inputf32 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp32f), NULL, NULL);
    d_inputi8 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8s), NULL, NULL);
    d_input_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
    d_inputf16_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp16f), NULL, NULL);
    d_inputf32_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp32f), NULL, NULL);
    d_inputi8_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8s), NULL, NULL);
    d_output = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp8u), NULL, NULL);
    d_outputf16 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp16f), NULL, NULL);
    d_outputf32 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp32f), NULL, NULL);
    d_outputi8 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp8s), NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_input, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_inputf16, CL_TRUE, 0, ioBufferSize * sizeof(Rpp16f), inputf16, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_inputf32, CL_TRUE, 0, ioBufferSize * sizeof(Rpp32f), inputf32, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_inputi8, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8s), inputi8, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_input_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input_second, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_inputf16_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp16f), inputf16_second, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_inputf32_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp32f), inputf32_second, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_inputi8_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8s), inputi8_second, 0, NULL, NULL);
    rppHandle_t handle;

    rppCreateWithStreamAndBatchSize(&handle, theQueue, noOfImages);

    clock_t start, end;
    double cpu_time_used;

    string test_case_name;

    switch (test_case)
    {
//     case 1:
//     {
//         test_case_name = "Erase";

        
//         start = clock();
//         if (ip_bitDepth == 0)
//             rppi_erase_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, angle, outputFormatToggle,noOfImages, handle);
//         else if (ip_bitDepth == 1)
//             rppi_erase_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, angle,outputFormatToggle, noOfImages, handle);
//         else if (ip_bitDepth == 2)
//             rppi_erase_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, angle,outputFormatToggle, noOfImages, handle);
//         else if (ip_bitDepth == 3)
// 	   rppi_erase_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, angle,outputFormatToggle, noOfImages, handle);
// 	else
//             missingFuncFlag = 1;
//         end = clock();
// 	break;
//     }
   case 1:
    {
        test_case_name = "non-linear blend";
        Rpp32f stdDev[images];
    for (i = 0; i < images; i++)
    {
        stdDev[i] = 0.2;
    }
        
        start = clock();
        if (ip_bitDepth == 0)
            rppi_non_linear_blend_u8_pkd3_batchPD_gpu(d_input,d_input_second, srcSize, maxSize, d_output, stdDev, outputFormatToggle,noOfImages, handle);
         else if (ip_bitDepth == 1)
             rppi_non_linear_blend_f16_pkd3_batchPD_gpu(d_inputf16,d_inputf16_second, srcSize, maxSize, d_outputf16, stdDev, outputFormatToggle,noOfImages, handle);
         else if (ip_bitDepth == 2)
             rppi_non_linear_blend_f32_pkd3_batchPD_gpu(d_inputf32,d_inputf32_second, srcSize, maxSize, d_outputf32, stdDev, outputFormatToggle,noOfImages, handle);
         else if (ip_bitDepth == 3)
	    rppi_non_linear_blend_i8_pkd3_batchPD_gpu(d_inputi8,d_inputi8_second, srcSize, maxSize, d_outputi8, stdDev, outputFormatToggle,noOfImages, handle);
	else
            missingFuncFlag = 1;
        end = clock();
	break;
    }
    case 2:
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
        ampl_x[i] = 10.0;
        ampl_y[i] = 10.0;
        freq_x[i] = 0.049087; 
        freq_y[i] = 0.049087;
        phase_x[i] = 0.2;
        phase_y[i] = 0.2;

    }
        start = clock();
        if (ip_bitDepth == 0)
            rppi_water_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, ampl_x,ampl_y,freq_x,freq_y,phase_x,phase_y,outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_water_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, ampl_x,ampl_y,freq_x,freq_y,phase_x,phase_y,outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_water_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, ampl_x,ampl_y,freq_x,freq_y,phase_x,phase_y,outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 3)
	   rppi_water_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, ampl_x,ampl_y,freq_x,freq_y,phase_x,phase_y,outputFormatToggle,noOfImages, handle);
	else
            missingFuncFlag = 1;
        end = clock();
	break;
    }
   
    case 3:
    {
        test_case_name = "look_up_table";


	const int luptrCount = images * 256 * ip_channel;
    Rpp8u luptr[luptrCount];
	int counter = 0;
    for(i = 0 ; i < images ; i++)
	{
        for(int x = 0 ; x < ip_channel * 256 ; x++)
        {    
            luptr[counter] = 255 - (x % 256);
            counter++;
        }
    }
        start = clock();
        //if (ip_bitDepth == 0)
            //rppi_look_up_table_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, luptr, noOfImages, handle);
    //     else if (ip_bitDepth == 1)
    //         rppi_glitch_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, angle,outputFormatToggle, noOfImages, handle);
    //     else if (ip_bitDepth == 2)
    //         rppi_glitch_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, angle,outputFormatToggle, noOfImages, handle);
    //     else if (ip_bitDepth == 3)
	//    rppi_glitch_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, angle,outputFormatToggle, noOfImages, handle);
	//else
            missingFuncFlag = 1;
        end = clock();
	break;
    }
    case 4:
    {
        test_case_name = "Erase";
        Rpp32u crop_params[240];
        Rpp8u colors[180];
        Rpp32f colors_f32[180];
        half colors_f16[180];
        char colors_i8[180];
        Rpp32u no_of_boxes[20];
        Rpp32u box_offset[20];
        int offset = 0;
        box_offset[0] = 0;
       
        for (i = 0; i < images; i++)
        {
            no_of_boxes[i] = i % 3 + 3;
            if (i != images - 1) box_offset[i+1] = box_offset[i] + no_of_boxes[i]; 
            for(int j = 0; j < no_of_boxes[i]; j++)
            {
               crop_params[4 * offset] = 100  + j * 100;
               crop_params[4 * offset + 1] = 100 + j * 50;
               crop_params[4 * offset + 2] = 700 + j * 120;
               crop_params[4 * offset + 3]  = 200 + j * 60;
               colors[3 * offset] = 155 - j * 5;
               colors[3 * offset + 1] = 155 - j * 10;
               colors[3 * offset + 2] = j * 20;
               colors_f32[3 * offset] = (155 - j * 5) / 255.0;
               colors_f32[3 * offset + 1] = (155 - j * 10) / 255.0;
               colors_f32[3 * offset + 2] = (j * 20) / 255.0;
               colors_f16[3 * offset] = (155 - j * 5) / 255.0;
               colors_f16[3 * offset + 1] = (155 - j * 10) / 255.0;
               colors_f16[3 * offset + 2] = (j * 20) / 255.0;
               colors_i8[3 * offset] = (155 - j * 5) - 128;
               colors_i8[3 * offset + 1] = (155 - j * 10) - 128;
               colors_i8[3 * offset + 2] = (j * 20) - 128;
               offset++;
            }
            // no_of_boxes[i] = 2;
            // if (i != images - 1) box_offset[i+1] = box_offset[i] + no_of_boxes[i]; 
            // crop_params[4 * offset] = 100;
            // crop_params[4 * offset + 1] = 100;
            // crop_params[4 * offset + 2] = 700;
            // crop_params[4 * offset + 3]  = 200;
            // colors[3 * offset] = 121;
            // colors[3 * offset + 1] = 200;
            // colors[3 * offset + 2] = 99;
            // offset++;
            // crop_params[4 * offset] = 800;
            // crop_params[4 * offset + 1] = 300;
            // crop_params[4 * offset + 2] = 1000;
            // crop_params[4 * offset + 3]  = 800;
            // colors[3 * offset] = 155;
            // colors[3 * offset + 1] = 100;    
            // colors[3 * offset + 2] =  20;
            // offset++;
        }
        int offset2 = 0;
        for (i = 0; i < images; i++)
        {
            std::cout << i <<"\t" << "number of boxes" << no_of_boxes[i] << "\t" << box_offset[i] <<
              "images \t" <<  images <<  std::endl; 
            for(int j = 0; j < no_of_boxes[i]; j++)
            {
                std::cout << "crop_params" << crop_params[4 * offset2] << "\t" << 
                crop_params[4 * offset2 + 1]<< "\t" <<crop_params[4 * offset2 + 2]<< "\t" << 
                crop_params[4 * offset2 + 3]<< std::endl;
            
                std::cout << "colors" << (int)colors[3 * offset2] << "\t" << 
                (int)colors[3 * offset2 + 1] << "\t" <<(int)colors[3 * offset2 + 2] << "\t" << std::endl;
                offset2++;
            }
        }
    
        std::cout << "offset is" << offset << std::endl;
        cl_mem d_colors, d_offset, d_boxes, d_colors_f16, d_colors_f32, d_colors_i8;
        d_boxes =  clCreateBuffer(theContext, CL_MEM_READ_ONLY,  offset * 4 * sizeof(int), NULL, NULL);
        d_offset = clCreateBuffer(theContext, CL_MEM_READ_ONLY,  images * sizeof(int), NULL, NULL);
        d_colors = clCreateBuffer(theContext, CL_MEM_READ_ONLY,  offset * 3 * sizeof(Rpp8u), NULL, NULL);
        d_colors_f32 = clCreateBuffer(theContext, CL_MEM_READ_ONLY,  offset * 3 * sizeof(Rpp32f), NULL, NULL);
        d_colors_f16 = clCreateBuffer(theContext, CL_MEM_READ_ONLY,  offset * 3 * sizeof(half), NULL, NULL);
        d_colors_i8 = clCreateBuffer(theContext, CL_MEM_READ_ONLY,  offset * 3 * sizeof(char), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_boxes, CL_TRUE, 0, offset * 4 * sizeof(int), crop_params, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_offset, CL_TRUE, 0, images * sizeof(int), box_offset, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_colors, CL_TRUE, 0, offset * 3 * sizeof(Rpp8u), colors, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_colors_f32, CL_TRUE, 0, offset * 3 * sizeof(Rpp32f), colors, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_colors_f16, CL_TRUE, 0, offset * 3 * sizeof(half), colors, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_colors_i8, CL_TRUE, 0, offset * 3 * sizeof(char), colors, 0, NULL, NULL);
        std::cout << "err is " << err << std::endl;
        start = clock();
        if (ip_bitDepth == 0)
            rppi_erase_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, d_boxes, d_colors, d_offset, no_of_boxes, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_erase_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, d_boxes, d_colors_f16, d_offset, no_of_boxes, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_erase_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, d_boxes, d_colors_f16, d_offset, no_of_boxes, outputFormatToggle,noOfImages, handle);
        else if (ip_bitDepth == 3)
            rppi_erase_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, d_boxes, d_colors_i8, d_offset, no_of_boxes, outputFormatToggle,noOfImages, handle);
        else
            missingFuncFlag = 1;
        end = clock();
        clReleaseMemObject(d_boxes);
        clReleaseMemObject(d_offset);
        clReleaseMemObject(d_colors);
        clReleaseMemObject(d_colors_f32);
        clReleaseMemObject(d_colors_i8);
        clReleaseMemObject(d_colors_f16);

	break;
    }
    case 5:
    {
        test_case_name = "color_cast";
        Rpp8u r[images];
        Rpp8u g[images];
        Rpp8u b[images];
        Rpp32f alpha[images];
    for (i = 0; i < images; i++)
    {
        r[i] = 225;
        g[i] = 235;
        b[i] = 52;
        alpha[i] = 0.5;
    }
        
        start = clock();
        if (ip_bitDepth == 0)
            rppi_color_cast_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output,r, g, b, alpha, outputFormatToggle, noOfImages, handle);
         else if (ip_bitDepth == 1)
            rppi_color_cast_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16,r, g, b, alpha, outputFormatToggle, noOfImages, handle);
         else if (ip_bitDepth == 2)
            rppi_color_cast_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32,r, g, b, alpha, outputFormatToggle, noOfImages, handle);
         else if (ip_bitDepth == 3)
            rppi_color_cast_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8,r, g, b, alpha, outputFormatToggle, noOfImages, handle);
	else
            missingFuncFlag = 1;
        end = clock();
	break;
    }

    case 6:
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
        x11[i] = 10;
        y11[i] = 10;
        x12[i] = 700;
        y12[i] = 700;
        x21[i] = 100;
        y21[i] = 100;
        x22[i] = 500;
        y22[i] = 500;
       
    }
        
        start = clock();
        if (ip_bitDepth == 0)
            rppi_crop_and_patch_u8_pkd3_batchPD_gpu(d_input,d_input_second, srcSize, maxSize, d_output,
            x11, y11, x12, y12, x21, y21, x22, y22, outputFormatToggle, noOfImages, handle);
         else if (ip_bitDepth == 1)
            rppi_crop_and_patch_f16_pkd3_batchPD_gpu(d_inputf16,d_inputf16_second, srcSize, maxSize, d_outputf16,
            x11, y11, x12, y12, x21, y21, x22, y22, outputFormatToggle, noOfImages, handle);
         else if (ip_bitDepth == 2)
            rppi_crop_and_patch_f32_pkd3_batchPD_gpu(d_inputf32,d_inputf32_second, srcSize, maxSize, d_outputf32,
            x11, y11, x12, y12, x21, y21, x22, y22, outputFormatToggle, noOfImages, handle);
         else if (ip_bitDepth == 3)
            rppi_crop_and_patch_i8_pkd3_batchPD_gpu(d_inputi8,d_inputi8_second, srcSize, maxSize, d_outputi8,
            x11, y11, x12, y12, x21, y21, x22, y22, outputFormatToggle, noOfImages, handle);
	else
            missingFuncFlag = 1;
        end = clock();
	break;
    }
    case 7:
    {
        test_case_name = "warp-affine"; 
        Rpp32f affine_array[6 * images];
        for (i = 0; i < images; i = i + 6)
        {
            affine_array[i] = 0.83;
            affine_array[i + 1] = 0.5;
            affine_array[i + 2] = 0.0;
            affine_array[i + 3] = -0.5;
            affine_array[i + 4] = 0.83;
            affine_array[i + 5] = 0.0;
        }


        start = clock();
        if (ip_bitDepth == 0)
            rppi_warp_affine_u8_pkd3_batchPD_gpu( d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, affine_array, outputFormatToggle, noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_warp_affine_f16_pkd3_batchPD_gpu( d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, affine_array, outputFormatToggle, noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_warp_affine_f32_pkd3_batchPD_gpu( d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, affine_array, outputFormatToggle, noOfImages, handle);
        else if (ip_bitDepth == 3)
            rppi_warp_affine_i8_pkd3_batchPD_gpu( d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, affine_array, outputFormatToggle, noOfImages, handle);
        else
            missingFuncFlag = 1;

        end = clock();
        break;
    }

        case 8:
    {
        test_case_name = "glitch";
        Rpp32u x_offset_r[images];
        Rpp32u y_offset_r[images];
        Rpp32u x_offset_g[images];
        Rpp32u y_offset_g[images];
        Rpp32u x_offset_b[images];
        Rpp32u y_offset_b[images];
        


       
    for (i = 0; i < images; i++)
    {
        x_offset_r[i] = 0;
        y_offset_r[i] = 0;
        x_offset_g[i] = 10;
        y_offset_g[i] = 10;
        x_offset_b[i] = 30;
        y_offset_b[i] = 30;
       
       
    }
        
        start = clock();
        if (ip_bitDepth == 0)
            rppi_glitch_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output,
            x_offset_r, y_offset_r,
            x_offset_g, y_offset_g,
            x_offset_b, y_offset_b, 
            outputFormatToggle, noOfImages, handle);
         else if (ip_bitDepth == 1)
            rppi_glitch_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16,
            x_offset_r, y_offset_r,
            x_offset_g, y_offset_g,
            x_offset_b, y_offset_b, 
            outputFormatToggle, noOfImages, handle);
         else if (ip_bitDepth == 2)
            rppi_glitch_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32,
            x_offset_r, y_offset_r,
            x_offset_g, y_offset_g,
            x_offset_b, y_offset_b, 
            outputFormatToggle, noOfImages, handle);
         else if (ip_bitDepth == 3)
            rppi_glitch_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8,
            x_offset_r, y_offset_r,
            x_offset_g, y_offset_g,
            x_offset_b, y_offset_b,
            outputFormatToggle, noOfImages, handle);
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
    else if ((ip_bitDepth == 1))
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
    else if ((ip_bitDepth == 2))
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
    else if ((ip_bitDepth == 3))
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
        if (ip_channel == 3)
        {
            mat_op_image = Mat(maxHeight, maxWidth, CV_8UC3, temp_output);
            imwrite(temp, mat_op_image);
        }
        if (ip_channel == 1)
        {
            mat_op_image = Mat(maxHeight, maxWidth, CV_8UC1, temp_output);
            imwrite(temp, mat_op_image);
        }
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
    clReleaseMemObject(d_inputf16_second);
    clReleaseMemObject(d_inputf32_second);
    clReleaseMemObject(d_inputi8_second);
    clReleaseMemObject(d_inputf16);
    clReleaseMemObject(d_outputf16);
    clReleaseMemObject(d_inputf32);
    clReleaseMemObject(d_outputf32);
    clReleaseMemObject(d_inputi8);
    clReleaseMemObject(d_outputi8);
    return 0;

}

