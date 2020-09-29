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

using namespace cv;
using namespace std;
using half_float::half;

typedef half Rpp16f;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))

template <typename T>
void displayTensor(T *pArr, Rpp32u size)
{
    int p = 0, count = 0;
    for (int i = 0; i < size; i++, count++)
    {
        printf("%d\t", (int) *(pArr + p));
        p++;
        if (count == 19)
        {
            printf("\n");
            count = 0;
        }
    }
    printf("\n");
}

template <typename T>
void displayTensorF(T *pArr, Rpp32u size)
{
    int p = 0, count = 0;
    for (int i = 0; i < size; i++, count++)
    {
        printf("%0.2f\t", (Rpp32f) *(pArr + p));
        p++;
        if (count == 19)
        {
            printf("\n");
            count = 0;
        }
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    const int MIN_ARG_COUNT = 3;
    printf("\nUsage: ./uniqueFunctionalities_host <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 1:2>\n");
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        return -1;
    }

    int ip_bitDepth = atoi(argv[1]);
    int test_case = atoi(argv[2]);

    clock_t start, end;
    double start_omp, end_omp;
    double cpu_time_used, omp_time_used;

    int missingFuncFlag = 0;
    string test_case_name;

    switch (test_case)
    {
    case 1:
    {
        test_case_name = "tensor_transpose";

        // Test Case 1
        Rpp32u totalNumberOfElements = 36;
        Rpp32u tensorDimension = 3;
        Rpp32u tensorDimensionValues[3] = {3, 3, 4};
        Rpp32u dimension1 = 0, dimension2 = 1;
        Rpp8u srcPtr[36] = {255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 130, 129, 128, 127, 126, 117, 113, 121, 127, 111, 100, 108, 65, 66, 67, 68, 69, 70, 71, 72, 13, 24, 15, 16};
        Rpp8u dstPtr[36] = {0};

        // Test Case 2
        // Rpp32u totalNumberOfElements = 48;
        // Rpp32u tensorDimension = 3;
        // Rpp32u tensorDimensionValues[3] = {4, 4, 3};
        // Rpp32u dimension1 = 0, dimension2 = 1;
        // Rpp8u srcPtr[48] = {255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 130, 129, 128, 127, 126, 117, 113, 121, 127, 111, 100, 108, 91, 95, 92, 98, 65, 66, 67, 68, 69, 70, 71, 72, 49, 47, 55, 51, 41, 39, 38, 34, 13, 24, 15, 16};
        // Rpp8u dstPtr[48] = {0};

        start = clock();
        start_omp = omp_get_wtime();
        rppi_tensor_transpose_u8_host(srcPtr, dstPtr, dimension1, dimension2, tensorDimension, tensorDimensionValues);
        end_omp = omp_get_wtime();
        end = clock();

        printf("\n\nInput:\n");
        displayTensor(srcPtr, totalNumberOfElements);
        printf("\n\nOutput of tensor_transpose:\n");
        displayTensor(dstPtr, totalNumberOfElements);

        break;
    }
    case 2:
    {
        test_case_name = "transpose";

        // Test Case 1
        // Rpp32u totalNumberOfElements = 24;
        // Rpp32u perm[4] = {0, 3, 1, 2};
        // Rpp32u shape[4] = {2, 2, 2, 3};
        // Rpp8u srcPtr[24] = {255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119};
        // Rpp8u dstPtr[24] = {0};
        // Rpp16f srcPtr16f[24], dstPtr16f[24];
        // Rpp32f srcPtr32f[24], dstPtr32f[24];
        // Rpp8s srcPtr8s[24], dstPtr8s[24];

        // Test Case 2
        Rpp32u totalNumberOfElements = 120;
        Rpp32u perm[4] = {0, 3, 1, 2};
        Rpp32u shape[4] = {2, 4, 5, 3};
        Rpp8u srcPtr[120] = {
            255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 5, 4, 3, 2, 1, 0, 
            27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 115, 114, 113, 112, 111, 110, 
            240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 140, 139, 138, 137, 136, 135, 
            70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 15, 14, 13, 12, 11, 10
        };
        Rpp8u dstPtr[120] = {0};
        Rpp16f srcPtr16f[120], dstPtr16f[120];
        Rpp32f srcPtr32f[120], dstPtr32f[120];
        Rpp8s srcPtr8s[120], dstPtr8s[120];


        for (int i = 0; i < totalNumberOfElements; i++)
        {
            srcPtr16f[i] = (Rpp16f) srcPtr[i];
            srcPtr32f[i] = (Rpp32f) srcPtr[i];
            srcPtr8s[i] = (Rpp8s) (((Rpp32s) srcPtr[i]) - 128);
        }

        start = clock();
        start_omp = omp_get_wtime();
        if (ip_bitDepth == 0)
            rppi_transpose_u8_host(srcPtr, dstPtr, perm, shape);
        else if (ip_bitDepth == 1)
            rppi_transpose_f16_host(srcPtr16f, dstPtr16f, perm, shape);
        else if (ip_bitDepth == 2)
            rppi_transpose_f32_host(srcPtr32f, dstPtr32f, perm, shape);
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
            rppi_transpose_i8_host(srcPtr8s, dstPtr8s, perm, shape);
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        else
            missingFuncFlag = 1;
        end_omp = omp_get_wtime();
        end = clock();

        if (ip_bitDepth == 0)
        {
            printf("\n\nInput:\n");
            displayTensor(srcPtr, totalNumberOfElements);
            printf("\n\nOutput of tensor_transpose:\n");
            displayTensor(dstPtr, totalNumberOfElements);
        }
        else if (ip_bitDepth == 1)
        {
            printf("\n\nInput:\n");
            displayTensorF(srcPtr16f, totalNumberOfElements);
            printf("\n\nOutput of tensor_transpose:\n");
            displayTensorF(dstPtr16f, totalNumberOfElements);
        }
        else if (ip_bitDepth == 2)
        {
            printf("\n\nInput:\n");
            displayTensorF(srcPtr32f, totalNumberOfElements);
            printf("\n\nOutput of tensor_transpose:\n");
            displayTensorF(dstPtr32f, totalNumberOfElements);
        }
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
        {
            printf("\n\nInput:\n");
            displayTensor(srcPtr8s, totalNumberOfElements);
            printf("\n\nOutput of tensor_transpose:\n");
            displayTensor(dstPtr8s, totalNumberOfElements);
        }
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

    if (missingFuncFlag == 1)
    {
        std::cout << "\nThe functionality " << test_case_name << " doesn't yet exist in RPP\n";
        return -1;
    }

    return 0;
}