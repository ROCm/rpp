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

using namespace cv;
using namespace std;
using half_float::half;

typedef half Rpp16f;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))

int main(int argc, char **argv)
{
    const int MIN_ARG_COUNT = 6;
    printf("\nUsage: ./BatchPD_host <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2> <case number = 0:64>\n");
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        return -1;
    }

    printf("\nsrc1 = %s", argv[1]);
    printf("\nsrc2 = %s", argv[2]);
    printf("\ndst = %s", argv[3]);
    printf("\nu8/f16/f32 (0/1/2) = %s", argv[4]);
    printf("\ncase number (1:64) = %s", argv[5]);

    char *src = argv[1];
    char *src_second = argv[2];
    char *dst = argv[3];
    int ip_bitDepth = atoi(argv[4]);
    int test_case = atoi(argv[5]);

    int ip_channel = 1;

    char funcType[1000] = {"BatchPD_HOST_PLN1"};

    char funcName[1000];
    switch (test_case)
    {
    case 0:
        strcpy(funcName, "brightness");
        break;
    case 1:
        strcpy(funcName, "contrast");
        break;
    case 2:
        strcpy(funcName, "blur");
        break;
    case 3:
        strcpy(funcName, "jitter");
        break;
    case 4:
        strcpy(funcName, "blend");
        break;
    case 5:
        strcpy(funcName, "color_temperature");
        break;
    case 6:
        strcpy(funcName, "gamma_correction");
        break;
    case 7:
        strcpy(funcName, "fog");
        break;
    case 8:
        strcpy(funcName, "snow");
        break;
    case 9:
        strcpy(funcName, "lens_correction");
        break;
    case 10:
        strcpy(funcName, "noise");
        break;
    case 11:
        strcpy(funcName, "pixelate");
        break;
    case 12:
        strcpy(funcName, "exposure");
        break;
    case 13:
        strcpy(funcName, "fisheye");
        break;
    case 14:
        strcpy(funcName, "vignette");
        break;
    case 15:
        strcpy(funcName, "flip");
        break;
    case 16:
        strcpy(funcName, "rain");
        break;
    case 17:
        strcpy(funcName, "rotate");
        break;
    case 18:
        strcpy(funcName, "warp_affine");
        break;
    case 19:
        strcpy(funcName, "resize");
        break;
    case 20:
        strcpy(funcName, "resize_crop");
        break;
    case 21:
        strcpy(funcName, "hueRGB");
        break;
    case 22:
        strcpy(funcName, "saturationRGB");
        break;
    case 23:
        strcpy(funcName, "histogram_balance");
        break;
    case 24:
        strcpy(funcName, "random_shadow");
        break;
    case 25:
        strcpy(funcName, "random_crop_letterbox");
        break;
    case 26:
        strcpy(funcName, "absolute_difference");
        break;
    case 27:
        strcpy(funcName, "accumulate");
        break;
    case 28:
        strcpy(funcName, "accumulate_squared");
        break;
    case 29:
        strcpy(funcName, "accumulate_weighted");
        break;
    case 30:
        strcpy(funcName, "add");
        break;
    case 31:
        strcpy(funcName, "subtract");
        break;
    case 32:
        strcpy(funcName, "bitwise_AND");
        break;
    case 33:
        strcpy(funcName, "exclusive_OR");
        break;
    case 34:
        strcpy(funcName, "inclusive_OR");
        break;
    case 35:
        strcpy(funcName, "bitwise_NOT");
        break;
    case 36:
        strcpy(funcName, "box_filter");
        break;
    case 37:
        strcpy(funcName, "canny_edge_detector");
        break;
    case 38:
        strcpy(funcName, "channel_extract");
        break;
    case 39:
        strcpy(funcName, "data_object_copy");
        break;
    case 40:
        strcpy(funcName, "dilate");
        break;
    case 41:
        strcpy(funcName, "histogram_equalization");
        break;
    case 42:
        strcpy(funcName, "erode");
        break;
    case 43:
        strcpy(funcName, "fast_corner_detector");
        break;
    case 44:
        strcpy(funcName, "gaussian_filter");
        break;
    case 45:
        strcpy(funcName, "gaussian_image_pyramid");
        break;
    case 46:
        strcpy(funcName, "harris_corner_detector");
        break;
    case 47:
        strcpy(funcName, "local_binary_pattern");
        break;
    case 48:
        strcpy(funcName, "laplacian_image_pyramid");
        break;
    case 49:
        strcpy(funcName, "magnitude");
        break;
    case 50:
        strcpy(funcName, "max");
        break;
    case 51:
        strcpy(funcName, "median_filter");
        break;
    case 52:
        strcpy(funcName, "min");
        break;
    case 53:
        strcpy(funcName, "nonlinear_filter");
        break;
    case 54:
        strcpy(funcName, "non_max_suppression");
        break;
    case 55:
        strcpy(funcName, "phase");
        break;
    case 56:
        strcpy(funcName, "multiply");
        break;
    case 57:
        strcpy(funcName, "scale");
        break;
    case 58:
        strcpy(funcName, "sobel_filter");
        break;
    case 59:
        strcpy(funcName, "thresholding");
        break;
    case 60:
        strcpy(funcName, "warp_perspective");
        break;
    case 61:
        strcpy(funcName, "resize_crop_mirror");
        break;
    case 62:
        strcpy(funcName, "crop");
        break;
    case 63:
        strcpy(funcName, "crop_mirror_normalize");
        break;
    case 64:
        strcpy(funcName, "color_twist");
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
            *inputf16Temp = (Rpp16f)*inputTemp;
            *inputf16_secondTemp = (Rpp16f)*input_secondTemp;
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
            *inputf32Temp = (Rpp32f)*inputTemp;
            *inputf32_secondTemp = (Rpp32f)*input_secondTemp;
            inputTemp++;
            inputf32Temp++;
            input_secondTemp++;
            inputf32_secondTemp++;
        }
    }

    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, noOfImages);

    clock_t start, end;
    double start_omp, end_omp;
    double cpu_time_used, omp_time_used;

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
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_brightness_u8_pln1_batchPD_host(input, srcSize, maxSize, output, alpha, beta, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 1:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_contrast_u8_pln1_batchPD_host(input, srcSize, maxSize, output, newMin, newMax, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 2:
    {
        test_case_name = "blur";

        Rpp32u kernelSize[images];
        for (i = 0; i < images; i++)
        {
            kernelSize[i] = 5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_blur_u8_pln1_batchPD_host(input, srcSize, maxSize, output, kernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 3:
    {
        test_case_name = "jitter";

        Rpp32u kernelSize[images];
        for (i = 0; i < images; i++)
        {
            kernelSize[i] = 5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_jitter_u8_pln1_batchPD_host(input, srcSize, maxSize, output, kernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 4:
    {
        test_case_name = "blend";

        Rpp32f alpha[images];
        for (i = 0; i < images; i++)
        {
            alpha[i] = 0.5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_blend_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, output, alpha, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 5:
    {
        test_case_name = "color_temperature";

        Rpp32s adjustmentValue[images];
        for (i = 0; i < images; i++)
        {
            adjustmentValue[i] = 70;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_color_temperature_u8_pln1_batchPD_host(input, srcSize, maxSize, output, adjustmentValue, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 6:
    {
        test_case_name = "gamma_correction";

        Rpp32f gamma[images];
        for (i = 0; i < images; i++)
        {
            gamma[i] = 1.9;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_gamma_correction_u8_pln1_batchPD_host(input, srcSize, maxSize, output, gamma, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 7:
    {
        test_case_name = "fog";

        Rpp32f fogValue[images];
        for (i = 0; i < images; i++)
        {
            fogValue[i] = 0.2;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_fog_u8_pln1_batchPD_host(input, srcSize, maxSize, output, fogValue, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 8:
    {
        test_case_name = "snow";

        Rpp32f snowPercentage[images];
        for (i = 0; i < images; i++)
        {
            snowPercentage[i] = 0.6;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_snow_u8_pln1_batchPD_host(input, srcSize, maxSize, output, snowPercentage, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 9:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_lens_correction_u8_pln1_batchPD_host(input, srcSize, maxSize, output, strength, zoom, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 10:
    {
        test_case_name = "noise";

        Rpp32f noiseProbability[images];
        for (i = 0; i < images; i++)
        {
            noiseProbability[i] = 0.2;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_noise_u8_pln1_batchPD_host(input, srcSize, maxSize, output, noiseProbability, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 11:
    {
        test_case_name = "pixelate";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_pixelate_u8_pln1_batchPD_host(input, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 12:
    {
        test_case_name = "exposure";

        Rpp32f exposureFactor[images];
        for (i = 0; i < images; i++)
        {
            exposureFactor[i] = 1.4;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_exposure_u8_pln1_batchPD_host(input, srcSize, maxSize, output, exposureFactor, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 13:
    {
        test_case_name = "fisheye";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_fisheye_u8_pln1_batchPD_host(input, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 14:
    {
        test_case_name = "vignette";

        Rpp32f stdDev[images];
        for (i = 0; i < images; i++)
        {
            stdDev[i] = 75.0;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_vignette_u8_pln1_batchPD_host(input, srcSize, maxSize, output, stdDev, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 15:
    {
        test_case_name = "flip";

        Rpp32u flipAxis[images];
        for (i = 0; i < images; i++)
        {
            flipAxis[i] = 1;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_flip_u8_pln1_batchPD_host(input, srcSize, maxSize, output, flipAxis, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 16:
    {
        test_case_name = "rain";

        Rpp32f rainPercentage[images];
        Rpp32u rainWidth[images];
        Rpp32u rainHeight[images];
        Rpp32f transparency[images];
        for (i = 0; i < images; i++)
        {
            rainPercentage[i] = 0.8;
            rainWidth[i] = 5;
            rainHeight[i] = 12;
            transparency[i] = 0.5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_rain_u8_pln1_batchPD_host(input, srcSize, maxSize, output, rainPercentage, rainWidth, rainHeight, transparency, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 17:
    {
        test_case_name = "rotate";

        Rpp32f angle[images];
        for (i = 0; i < images; i++)
        {
            angle[i] = 50;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_rotate_u8_pln1_batchPD_host(input, srcSize, maxSize, output, dstSize, maxDstSize, angle, noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_rotate_f16_pln1_batchPD_host(inputf16, srcSize, maxSize, outputf16, dstSize, maxDstSize, angle, noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_rotate_f32_pln1_batchPD_host(inputf32, srcSize, maxSize, outputf32, dstSize, maxDstSize, angle, noOfImages, handle);
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 18:
    {
        test_case_name = "warp_affine";

        Rpp32f affine_array[6 * images];
        for (i = 0; i < 6 * images; i = i + 6)
        {
            affine_array[i] = 0.83;
            affine_array[i + 1] = 0.5;
            affine_array[i + 2] = 0.0;
            affine_array[i + 3] = -0.5;
            affine_array[i + 4] = 0.83;
            affine_array[i + 5] = 0.0;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_warp_affine_u8_pln1_batchPD_host(input, srcSize, maxSize, output, dstSize, maxDstSize, affine_array, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 19:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_resize_u8_pln1_batchPD_host(input, srcSize, maxSize, output, dstSize, maxDstSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_resize_f16_pln1_batchPD_host(inputf16, srcSize, maxSize, outputf16, dstSize, maxDstSize, noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_resize_f32_pln1_batchPD_host(inputf32, srcSize, maxSize, outputf32, dstSize, maxDstSize, noOfImages, handle);
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 20:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_resize_crop_u8_pln1_batchPD_host(input, srcSize, maxSize, output, dstSize, maxDstSize, x1, x2, y1, y2, noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_resize_crop_f16_pln1_batchPD_host(inputf16, srcSize, maxSize, outputf16, dstSize, maxDstSize, x1, x2, y1, y2, noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_resize_crop_f32_pln1_batchPD_host(inputf32, srcSize, maxSize, outputf32, dstSize, maxDstSize, x1, x2, y1, y2, noOfImages, handle);
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 21:
    {
        test_case_name = "hueRGB";

        Rpp32f hueShift[images];
        for (i = 0; i < images; i++)
        {
            hueShift[i] = 60;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_hueRGB_u8_pln1_batchPD_host(input, srcSize, maxSize, output, hueShift, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 22:
    {
        test_case_name = "saturationRGB";

        Rpp32f saturationFactor[images];
        for (i = 0; i < images; i++)
        {
            saturationFactor[i] = 5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_saturationRGB_u8_pln1_batchPD_host(input, srcSize, maxSize, output, saturationFactor, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 23:
    {
        test_case_name = "histogram_balance";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_histogram_balance_u8_pln1_batchPD_host(input, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 24:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_random_shadow_u8_pln1_batchPD_host(input, srcSize, maxSize, output, x1, y1, x2, y2, numbeoOfShadows, maxSizeX, maxSizey, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 25:
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
            x2[i] = 100;
            y2[i] = 100;
            dstSize[i].height = 140;
            dstSize[i].width = 140;
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_random_crop_letterbox_u8_pln1_batchPD_host(input, srcSize, maxSize, output, dstSize, maxDstSize, x1, x2, y1, y2, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 26:
    {
        test_case_name = "absolute_difference";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_absolute_difference_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 27:
    {
        test_case_name = "accumulate";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_accumulate_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 28:
    {
        test_case_name = "accumulate_squared";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_accumulate_squared_u8_pln1_batchPD_host(input, srcSize, maxSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 29:
    {
        test_case_name = "accumulate_weighted";

        Rpp32f alpha[images];
        for (i = 0; i < images; i++)
        {
            alpha[i] = 0.5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_accumulate_weighted_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, alpha, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 30:
    {
        test_case_name = "add";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_add_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 31:
    {
        test_case_name = "subtract";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_subtract_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 32:
    {
        test_case_name = "bitwise_AND";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_bitwise_AND_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 33:
    {
        test_case_name = "exclusive_OR";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_exclusive_OR_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 34:
    {
        test_case_name = "inclusive_OR";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_inclusive_OR_u8_pln1_batchPD_gpu(input, input_second, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 35:
    {
        test_case_name = "bitwise_NOT";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_bitwise_NOT_u8_pln1_batchPD_host(input, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 36:
    {
        test_case_name = "box_filter";

        Rpp32u kernelSize[images];
        for (i = 0; i < images; i++)
        {
            kernelSize[i] = 5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_box_filter_u8_pln1_batchPD_host(input, srcSize, maxSize, output, kernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 37:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_canny_edge_detector_u8_pln1_batchPD_host(input, srcSize, maxSize, output, minThreshold, maxThreshold, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 38:
    {
        test_case_name = "channel_extract";

        Rpp32u extractChannelNumber[images];
        for (i = 0; i < images; i++)
        {
            extractChannelNumber[i] = 0;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_channel_extract_u8_pln1_batchPD_host(input, srcSize, maxSize, output, extractChannelNumber, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 39:
    {
        test_case_name = "data_object_copy";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_data_object_copy_u8_pln1_batchPD_host(input, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 40:
    {
        test_case_name = "dilate";

        Rpp32u kernelSize[images];
        for (i = 0; i < images; i++)
        {
            kernelSize[i] = 5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_dilate_u8_pln1_batchPD_host(input, srcSize, maxSize, output, kernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 41:
    {
        test_case_name = "histogram_equalization";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_histogram_equalization_u8_pln1_batchPD_host(input, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 42:
    {
        test_case_name = "erode";

        Rpp32u kernelSize[images];
        for (i = 0; i < images; i++)
        {
            kernelSize[i] = 5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_erode_u8_pln1_batchPD_host(input, srcSize, maxSize, output, kernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 43:
    {
        test_case_name = "fast_corner_detector";

        Rpp32u numOfPixels[images];
        Rpp8u threshold[images];
        Rpp32u nonmaxKernelSize[images];
        for (i = 0; i < images; i++)
        {
            numOfPixels[i] = 4;
            threshold[i] = 15;
            nonmaxKernelSize[i] = 5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_fast_corner_detector_u8_pln1_batchPD_host(input, srcSize, maxSize, output, numOfPixels, threshold, nonmaxKernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 44:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_gaussian_filter_u8_pln1_batchPD_host(input, srcSize, maxSize, output, stdDev, kernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 45:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_gaussian_image_pyramid_u8_pln1_batchPD_host(input, srcSize, maxSize, output, stdDev, kernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 46:
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
            gaussianKernelSize[i] = 7;
            stdDev[i] = 5.0;
            kernelSize[i] = 5;
            kValue[i] = 1;
            threshold[i] = 10.0;
            nonmaxKernelSize[i] = 5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_harris_corner_detector_u8_pln1_batchPD_host(input, srcSize, maxSize, output, gaussianKernelSize, stdDev, kernelSize, kValue, threshold, nonmaxKernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 47:
    {
        test_case_name = "local_binary_pattern";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_local_binary_pattern_u8_pln1_batchPD_host(input, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 48:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_laplacian_image_pyramid_u8_pln1_batchPD_host(input, srcSize, maxSize, output, stdDev, kernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 49:
    {
        test_case_name = "magnitude";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_magnitude_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 50:
    {
        test_case_name = "max";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_max_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_median_filter_u8_pln1_batchPD_host(input, srcSize, maxSize, output, kernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 52:
    {
        test_case_name = "min";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_min_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 53:
    {
        test_case_name = "nonlinear_filter";

        Rpp32u kernelSize[images];
        for (i = 0; i < images; i++)
        {
            kernelSize[i] = 5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_nonlinear_filter_u8_pln1_batchPD_host(input, srcSize, maxSize, output, kernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 54:
    {
        test_case_name = "non_max_suppression";

        Rpp32u kernelSize[images];
        for (i = 0; i < images; i++)
        {
            kernelSize[i] = 5;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_non_max_suppression_u8_pln1_batchPD_host(input, srcSize, maxSize, output, kernelSize, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 55:
    {
        test_case_name = "phase";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_phase_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 56:
    {
        test_case_name = "multiply";

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_multiply_u8_pln1_batchPD_host(input, input_second, srcSize, maxSize, output, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 57:
    {
        test_case_name = "scale";

        Rpp32f percentage[images];
        for (i = 0; i < images; i++)
        {
            percentage[i] = 100;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_scale_u8_pln1_batchPD_host(input, srcSize, maxSize, output, dstSize, maxDstSize, percentage, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 58:
    {
        test_case_name = "sobel_filter";

        Rpp32u sobelType[images];
        for (i = 0; i < images; i++)
        {
            sobelType[i] = 1;
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_sobel_filter_u8_pln1_batchPD_host(input, srcSize, maxSize, output, sobelType, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 59:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_thresholding_u8_pln1_batchPD_host(input, srcSize, maxSize, output, min, max, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 60:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_warp_perspective_u8_pln1_batchPD_host(input, srcSize, maxSize, output, dstSize, maxDstSize, perspective, noOfImages, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 61:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_resize_crop_mirror_u8_pln1_batchPD_host(input, srcSize, maxSize, output, dstSize, maxDstSize, x1, x2, y1, y2, mirrorFlag, noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_resize_crop_mirror_f16_pln1_batchPD_host(inputf16, srcSize, maxSize, outputf16, dstSize, maxDstSize, x1, x2, y1, y2, mirrorFlag, noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_resize_crop_mirror_f32_pln1_batchPD_host(inputf32, srcSize, maxSize, outputf32, dstSize, maxDstSize, x1, x2, y1, y2, mirrorFlag, noOfImages, handle);
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 62:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_crop_u8_pln1_batchPD_host(input, srcSize, maxSize, output, dstSize, maxDstSize, crop_pos_x, crop_pos_y, noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_crop_f16_pln1_batchPD_host(inputf16, srcSize, maxSize, outputf16, dstSize, maxDstSize, crop_pos_x, crop_pos_y, noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_crop_f32_pln1_batchPD_host(inputf32, srcSize, maxSize, outputf32, dstSize, maxDstSize, crop_pos_x, crop_pos_y, noOfImages, handle);
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 63:
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
        Rpp32u outputFormatToggle = 0;

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_crop_mirror_normalize_u8_pln1_batchPD_host(input, srcSize, maxSize, output, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, noOfImages, handle);
        else if (ip_bitDepth == 1)
            rppi_crop_mirror_normalize_f16_pln1_batchPD_host(inputf16, srcSize, maxSize, outputf16, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, noOfImages, handle);
        else if (ip_bitDepth == 2)
            rppi_crop_mirror_normalize_f32_pln1_batchPD_host(inputf32, srcSize, maxSize, outputf32, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, noOfImages, handle);
        end_omp = omp_get_wtime();
        end = clock();

        break;
    }
    case 64:
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
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
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
    omp_time_used = end_omp - start_omp;
    cout << "\nCPU Time - BatchPD : " << cpu_time_used;
    cout << "\nOMP Time - BatchPD : " << omp_time_used;
    printf("\n");

    if (ip_bitDepth == 1)
    {
        int valCount = 0;
        Rpp8u *outputTemp;
        outputTemp = output;
        Rpp16f *outputf16Temp;
        outputf16Temp = outputf16;
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = (Rpp8u)RPPPIXELCHECK(*outputf16Temp);
            outputf16Temp++;
            outputTemp++;
        }
    }
    else if (ip_bitDepth == 2)
    {
        int valCount = 0;
        Rpp8u *outputTemp;
        outputTemp = output;
        Rpp32f *outputf32Temp;
        outputf32Temp = outputf32;
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = (Rpp8u)RPPPIXELCHECK(*outputf32Temp);
            outputf32Temp++;
            outputTemp++;
        }
    }

    rppDestroyHost(handle);

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
    free(outputf32);

    return 0;
}
