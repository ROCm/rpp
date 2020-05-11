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
using namespace cv;
using namespace std;
#include <CL/cl.hpp>
#define images 100
int G_IP_CHANNEL = 3;
int G_MODE = 1;
char src[1000] = {"/home/neel/Ulagammai/Input_Images_16/RGB"};
char src_second[1000] = {"/home/neel/Ulagammai/Input_Images_16/RGB1"};
char dst[1000] = {"/home/neel/Ulagammai/Output"};
char funcType[1000] = {"Single"};


int main(int argc, char **argv)
{
    int ip_channel = G_IP_CHANNEL;
    int mode = G_MODE;
    char *funcName = argv[1]; 
    if(mode == 0)
    {
        strcat(funcType,"_CPU");
    }
    else if (mode == 1)
    {
        strcat(funcType,"_GPU");
    }
    else
    {
        strcat(funcType,"_HIP");
    }
    if(ip_channel == 1)
    {
        strcat(funcType,"_PLN");
    }
    else
    {
        strcat(funcType,"_PKD");
    }
    
    int i = 0, j = 0;
    int minHeight = 30000, minWidth = 30000, maxHeight = 0, maxWidth = 0;
    unsigned long long count = 0;
    unsigned long long ioBufferSize = 0;
    
    static int noOfImages = 0;


    Mat image,image_second;

    struct dirent *de;
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");
    char src1_second[1000];
    strcpy(src1_second, src_second);
    strcat(src1_second, "/");
    strcat(funcName,funcType);
    strcat(dst,"/");
    strcat(dst,funcName);
    mkdir(dst, 0700);
    strcat(dst,"/");

    DIR *dr = opendir(src); 
    while ((de = readdir(dr)) != NULL) 
    {
        if(strcmp(de->d_name,".") == 0 || strcmp(de->d_name,"..") == 0)
            continue;
        noOfImages += 1;
        break;
    }
    closedir(dr);

    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    //const int images = noOfImages;
    char imageNames[images][1000];

    DIR *dr1 = opendir(src);
    while ((de = readdir(dr1)) != NULL) 
    {
        if(strcmp(de->d_name,".") == 0 || strcmp(de->d_name,"..") == 0) 
            continue;
        strcpy(imageNames[count],de->d_name);
        char temp[1000];
        strcpy(temp,src1);
        strcat(temp, imageNames[count]);
        if(ip_channel == 3)
        {
            image = imread(temp, 1);
        }
        else
        {
            image = imread(temp, 0);
        }
        srcSize[count].height = image.rows;
        srcSize[count].width = image.cols;
        ioBufferSize += (unsigned long long)srcSize[count].height * (unsigned long long)srcSize[count].width * (unsigned long long)ip_channel;

        count++;
        break;
    }
    closedir(dr1); 

    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));

    /* Read the input image */
    DIR *dr2 = opendir(src);
    DIR *dr2_second = opendir(src_second);
    count = 0;
    i = 0;
    while ((de = readdir(dr2)) != NULL) 
    {
        if(strcmp(de->d_name,".") == 0 || strcmp(de->d_name,"..") == 0) 
            continue;
        char temp[1000];
        strcpy(temp,src1);
        strcat(temp, de->d_name);
        char temp_second[1000];
        strcpy(temp_second,src1_second);
        strcat(temp_second, de->d_name); 
        if(ip_channel == 3)
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
        for(j = 0 ; j < srcSize[i].height * srcSize[i].width * ip_channel ; j++)
        {
            input[count] = ip_image[j];
            input_second[count] = ip_image_second[j];
            count++;
        }
        i++;
        break;
    }
    
    closedir(dr2); 
    rppHandle_t handle;
	rppCreateWithBatchSize(&handle, noOfImages);
 
	
	clock_t start, end;   
	double cpu_time_used;
	start = clock();


string test_case_name;
    int test_case = atoi(argv[1]);
    Rpp32u newMin = 30;
Rpp32u newMax = 100;
   uint kernelSize = 3;
    Rpp32f alpha = 0.5;
    Rpp32f beta = 100;
    Rpp32s adjustmentValue = 100;
    Rpp32f exposureFactor = 0.5;
    Rpp32u flipAxis = 0;
    Rpp32f fogValue = 1;
    Rpp32f gamma = 0.5;
    Rpp32f strength = 1.5;
    Rpp32f zoom = 0;
    Rpp32f noiseProbability = 0.5;
    Rpp32f rainPercentage = 0.5; 
    Rpp32u rainWidth = 3; 
    Rpp32u rainHeight = 6;
    Rpp32f transparency = 0.5;
    Rpp32f stdDev = 20;
    Rpp32f snowPercentage = 0.4;
    Rpp32f angle = 135.0;
    Rpp32f affine[6] = {1.0, 2.0, 1.0, 1.0, 1.0, 2.0};
    Rpp32f coordinates[4] = {100, 200, 200, 400};
    Rpp32f hueShift = 10;
    Rpp32f saturationFactor = 10;
    Rpp8u minThreshold = 10;
    Rpp8u maxThreshold = 30;
    Rpp32u numOfPixels = 4;
		Rpp32u gaussianKernelSize = 7;
		Rpp32f kValue = 1;
		Rpp32f threshold = 15;
		Rpp32u nonmaxKernelSize = 5;
     	Rpp32u sobelType = 1;
		Rpp8u min = 10;
		Rpp8u max = 30;
        Rpp32u crop_pos_x = 100;
	Rpp32u crop_pos_y = 100;
        Rpp32u xRoiBegin = 50;
		Rpp32u yRoiBegin = 50;
        	Rpp32u xRoiEnd  = 200;
        	Rpp32u yRoiEnd  = 200;
        	Rpp32u mirrorFlag = 0;

	
        Rpp32f perspective[9];

        perspective[0] = 1;
        perspective[1] = 0;
        perspective[2] = 0.5;
        perspective[3] = 0;
        perspective[4] = 1;
        perspective[5] = 0.5;  
        perspective[6] = 1;
        perspective[7] = 0;
        perspective[8] = 0.5;
        Rpp32f percentage = 100;
        Rpp32u numbeoOfShadows =12;
        Rpp32u maxSizeX = 12;
	Rpp32u maxSizey = 15;
        Rpp32u extractChannelNumber = 1;

    switch (test_case)
    {
    case 0:
        test_case_name = "contrast";
        rppi_contrast_u8_pkd3_host(input, srcSize[0], output, newMin, newMax, handle);
        break;
    case 1:
        test_case_name = "jitter";
        rppi_jitter_u8_pkd3_host(input, srcSize[0], output, kernelSize, handle);
        break;
    case 2:
        test_case_name = "blur";
        rppi_blur_u8_pkd3_host(input, srcSize[0], output, kernelSize, handle);
	break;
    case 3:
        test_case_name = "brightness";
        rppi_brightness_u8_pkd3_host(input, srcSize[0], output, alpha, beta, handle);
        break;
    case 4:
        test_case_name = "blend";
	rppi_blend_u8_pkd3_host(input, input_second, srcSize[0], output, alpha, handle);
        break;
    case 5:
        test_case_name = "color_temperature";
	rppi_color_temperature_u8_pkd3_host(input, srcSize[0], output, adjustmentValue, handle);
        break;
    case 6:
        test_case_name = "gamma_correction";
        rppi_gamma_correction_u8_pkd3_host(input, srcSize[0], output, gamma, handle);
        break;
    case 7:
        test_case_name = "fog";
        rppi_fog_u8_pkd3_host(input, srcSize[0], output, fogValue, handle);
        break;
    case 8:
        test_case_name = "snow";
        rppi_snow_u8_pkd3_host(input, srcSize[0], output, snowPercentage, handle);
        break;
    case 9:
        test_case_name = "lens_correction";
        rppi_lens_correction_u8_pkd3_host(input, srcSize[0], output, strength, zoom, handle);
        break;
    case 10:
        test_case_name = "noise";
        rppi_noise_u8_pkd3_host(input, srcSize[0], output, noiseProbability, handle);
        break;
    case 11:
        test_case_name = "pixelate";
        rppi_pixelate_u8_pkd3_host(input, srcSize[0], output, handle);
        break;
    case 12:
        test_case_name = "exposure";
        rppi_exposure_u8_pkd3_host(input, srcSize[0], output, exposureFactor, handle);
        break;
    case 13:
        test_case_name = "fisheye";
	rppi_fisheye_u8_pkd3_host(input, srcSize[0], output, handle);
        break;
    case 14:
        test_case_name = "vignette";
        rppi_vignette_u8_pkd3_host(input, srcSize[0], output, stdDev, handle);
        break;
    case 15:
        test_case_name = "flip";
        rppi_flip_u8_pkd3_host(input, srcSize[0], output, flipAxis, handle);
        break;
    case 16:
        test_case_name = "rain";
        rppi_rain_u8_pkd3_host(input, srcSize[0], output, rainPercentage, rainWidth, rainHeight, transparency, handle);
        break;
    case 17:
        test_case_name = "rotate";
        rppi_rotate_u8_pkd3_host(input, srcSize[0], output, dstSize[0], angle, handle);
        break;
    case 18:
        test_case_name = "warp-affine";
        rppi_warp_affine_u8_pkd3_host(input, srcSize[0], output, dstSize[0], affine, handle);
        break;
    case 19:
        test_case_name = "resize";
        rppi_resize_u8_pkd3_host(input, srcSize[0], output, dstSize[0], handle);
    case 20:
        test_case_name = "resize_crop";
        rppi_resize_crop_u8_pkd3_host(input, srcSize[0], output, dstSize[0], coordinates[0], coordinates[1],
        coordinates[2], coordinates[3],  handle);
   case 21:
        test_case_name = "Hue modification";
	rppi_hueRGB_u8_pkd3_host(input, srcSize[0], output, hueShift, handle);
        break;
    case 22:
        test_case_name = "Saturation";
	rppi_saturationRGB_u8_pkd3_host(input, srcSize[0], output, saturationFactor, handle);
        break;
    case 23:
        test_case_name = "Histogram Balance";
	rppi_histogram_balance_u8_pkd3_host(input, srcSize[0], output, handle);
        break;
    case 24:
        test_case_name = "RandomShadow";
	rppi_random_shadow_u8_pkd3_host(input, srcSize[0], output, coordinates[0], coordinates[1],
        coordinates[2], coordinates[3], numbeoOfShadows, maxSizeX, maxSizey, handle);        break;
    case 25:
        test_case_name = "RandomCropLetterBox";
	rppi_random_crop_letterbox_u8_pkd3_host(input, srcSize[0], output, dstSize[0], coordinates[0], coordinates[1],
        coordinates[2], coordinates[3],  handle);

        break;
    case 26:
        test_case_name = "Absolute Difference";
	rppi_absolute_difference_u8_pkd3_host(input, input_second, srcSize[0], output, handle);
        break;
    case 27:
        test_case_name = "Accumulate";
	rppi_accumulate_u8_pkd3_host(input, input_second, srcSize[0], handle);
        break;
    case 28:
        test_case_name = "Accumulate Squared";
	rppi_accumulate_squared_u8_pkd3_host(input, srcSize[0],  handle);
        break;
    case 29:
        test_case_name = "Accumulate Weighted";
	rppi_accumulate_weighted_u8_pkd3_host(input, input_second, srcSize[0], alpha, handle);
        break;
    case 30:
        test_case_name = "Arithmetic Addition";
	rppi_add_u8_pkd3_host(input, input_second, srcSize[0], output, handle);
        break;
    case 31:
        test_case_name = "Arithmetic Subtraction";
	rppi_subtract_u8_pkd3_host(input, input_second, srcSize[0], output, handle);
        break;
    case 32:
        test_case_name = "Bitwise AND";
	rppi_bitwise_AND_u8_pkd3_host(input, input_second, srcSize[0], output, handle);
        break;
    case 33:
        test_case_name = "Bitwise EXCLUSIVE OR";
	rppi_exclusive_OR_u8_pkd3_host(input, input_second, srcSize[0], output, handle);
        break;
    case 34:
        test_case_name = "Bitwise INCLUSIVE OR";
	rppi_inclusive_OR_u8_pkd3_host(input, input_second, srcSize[0], output, handle);
        break;
    case 35:
        test_case_name = "Bitwise NOT";
	rppi_bitwise_NOT_u8_pkd3_host(input, srcSize[0], output, handle);
        break;
    case 36:
        test_case_name = "Box Filter";
	rppi_box_filter_u8_pkd3_host(input, srcSize[0], output, kernelSize, handle);
        break;
    case 37:
        test_case_name = "Canny Edge Detector";
	rppi_canny_edge_detector_u8_pkd3_host(input, srcSize[0], output, minThreshold, maxThreshold, handle);
        break;
    case 38:
        test_case_name = "Channel Extract";
	rppi_channel_extract_u8_pkd3_host(input, srcSize[0], output, extractChannelNumber, handle);
        break;
    case 39:
        test_case_name = "Data Object Copy";
	rppi_data_object_copy_u8_pkd3_host(input, srcSize[0], output, handle);
        break;
    case 40:
        test_case_name = "Dilate Image";
	rppi_dilate_u8_pkd3_host(input, srcSize[0], output, kernelSize, handle);
        break;
    case 41:
        test_case_name = "Equalize Histogram";
	rppi_histogram_equalization_u8_pkd3_host(input, srcSize[0], output, handle);
        break;
    case 42:
        test_case_name = "Erode Image";
	rppi_erode_u8_pkd3_host(input, srcSize[0], output, kernelSize, handle);
        break;
    case 43:
        test_case_name = "Fast Corners";
	rppi_fast_corner_detector_u8_pkd3_host(input, srcSize[0], output, numOfPixels, threshold, nonmaxKernelSize, handle);
        break;
    case 44:
        test_case_name = "Gaussian Filter";
	rppi_gaussian_filter_u8_pkd3_host(input, srcSize[0], output, stdDev, kernelSize, handle);
        break;
    case 45:
        test_case_name = "Gaussian Image Pyramid";
rppi_gaussian_image_pyramid_u8_pkd3_host(input, srcSize[0], output, stdDev, kernelSize, handle);
        break;
    case 46:
        test_case_name = "Harris Corners";
	rppi_harris_corner_detector_u8_pkd3_host(input, srcSize[0], output, gaussianKernelSize, stdDev, kernelSize, kValue, threshold, nonmaxKernelSize, handle);

        break;
    case 47:
        test_case_name = "LBP";
	rppi_local_binary_pattern_u8_pkd3_host(input, srcSize[0], output, handle);
        break;
    case 48:
        test_case_name = "Laplacian Image Pyramid";
	rppi_laplacian_image_pyramid_u8_pkd3_host(input, srcSize[0], output, stdDev, kernelSize, handle);
        break;
    case 49:
        test_case_name = "Magnitude";
	rppi_magnitude_u8_pkd3_host(input, input_second, srcSize[0], output, handle);
        break;
    case 50:
        test_case_name = "Max";
	rppi_max_u8_pkd3_host(input, input_second, srcSize[0], output, handle);
        break;
    case 51:
        test_case_name = "Median Filter";
	rppi_median_filter_u8_pkd3_host(input, srcSize[0], output, kernelSize, handle);
        break;
   case 52:
        test_case_name = "Min";
	rppi_min_u8_pkd3_host(input, input_second, srcSize[0], output, handle);
        break;
    case 53:
        test_case_name = "Non Linear Filter";
	rppi_nonlinear_filter_u8_pkd3_host(input, srcSize[0], output, kernelSize, handle);
        break;
    case 54:
        test_case_name = "Non-Maxima Suppression";
	rppi_non_max_suppression_u8_pkd3_host(input, srcSize[0], output, kernelSize, handle);
        break;
    case 55:
        test_case_name = "Phase";
	rppi_phase_u8_pkd3_host(input, input_second, srcSize[0], output, handle);
        break;
    case 56:
        test_case_name = "Pixel-wise Multiplication";
	rppi_multiply_u8_pkd3_host(input, input_second, srcSize[0], output, handle);
        break;
    case 57:
        test_case_name = "Scale Image";
	rppi_scale_u8_pkd3_gpu(input, srcSize[0], output, dstSize[0], percentage, handle);
        break;
    case 58:
        test_case_name = "Sobel 3x3";
      rppi_sobel_filter_u8_pkd3_gpu(input, srcSize[0], output, sobelType, handle);
        break;
    case 59:
        test_case_name = "Thresholding";
	rppi_thresholding_u8_pkd3_host(input, srcSize[0], output, min, max, handle);
        break;
   case 60:
        test_case_name = "Warp Perspective";
	rppi_warp_perspective_u8_pkd3_host(input, srcSize[0], output, dstSize[0], perspective, handle);
        break;
   
    
    default:
        break;
    }
    

    end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	cout<<" Single : "<<cpu_time_used<<endl;  

    	rppDestroyHost(handle);


    count = 0;
    for(j = 0 ; j < noOfImages ; j++)
    {
        int op_size = srcSize[j].height * srcSize[j].width * ip_channel;
        Rpp8u *temp_output = (Rpp8u *)calloc(op_size, sizeof(Rpp8u));
        for(i = 0 ; i < op_size ; i++)
        {
            temp_output[i] = output[count];
            count++;
        }
        char temp[1000];
        strcpy(temp,dst);
        strcat(temp, imageNames[j]);
        Mat mat_op_image;
        if(ip_channel == 3)
        {
            mat_op_image = Mat(srcSize[j].height, srcSize[j].width, CV_8UC3, temp_output);
            imwrite(temp, mat_op_image);
        }
        if(ip_channel == 1)
        {
            mat_op_image = Mat(srcSize[j].height, srcSize[j].width, CV_8UC1, temp_output);
            imwrite(temp, mat_op_image);
        }
        free(temp_output);
    }

    free(srcSize);
    free(input);
    free(output);
    free(input_second);
	 return 0; 
}
