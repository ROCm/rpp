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
int main(int argc, char **argv)
{
    const int MIN_ARG_COUNT = 5;
    printf("\nUsage: ./batchPD_ocl <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <case number = 0:64>\n");
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        return -1;
    }

    printf("\nsrc1 = %s", argv[1]);
    printf("\nsrc2 = %s", argv[2]);
    printf("\ndst = %s", argv[3]);
    printf("\ncase number (1:64) = %s", argv[4]);

    char *src = argv[1];
    char *src_second = argv[2];
    char *dst = argv[3];
    int test_case = atoi(argv[4]);
    int ip_channel = 3;

    char funcType[1000] = {"batchSD_OCL_PKD3"};

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


    char func[1000];
    strcpy(func, funcName);
    strcat(func, funcType);
    printf("\n\nRunning %s...", func);

	int i = 0, j = 0;
	int minHeight = 30000, minWidth = 30000, maxHeight = 0, maxWidth = 0;
	
	unsigned long long ioBufferSize = 0;

	static int noOfImages = 128;

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
	mkdir(dst, 0700);
	strcat(dst, "/");

	DIR *dr = opendir(src);
	while ((de = readdir(dr)) != NULL)
	{
		if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
			continue;
		noOfImages += 1;
		break;
	}
	closedir(dr);

	RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
	RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
	const int images = noOfImages;
	char imageNames[images][1000];

	unsigned long long count1 = 0;

	DIR *dr1 = opendir(src);
	while ((de = readdir(dr1)) != NULL)
	{
		if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
			continue;
		strcpy(imageNames[count1], de->d_name);
		char temp[1000];
		strcpy(temp, src1);
		strcat(temp, imageNames[count1]);
		if (ip_channel == 3)
		{
			image = imread(temp, 1);
		}
		else
		{
			image = imread(temp, 0);
		}
		srcSize[count1].height = image.rows;
		srcSize[count1].width = image.cols;
		ioBufferSize += (unsigned long long)srcSize[count1].height * (unsigned long long)srcSize[count1].width * (unsigned long long)ip_channel;

		count1++;
		break;
	}
	closedir(dr1);

	Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
	Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
	Rpp8u *output = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));

	/* Read the input image */
	DIR *dr2 = opendir(src);
	DIR *dr2_second = opendir(src_second);
	unsigned long long count2 = 0;
	i = 0;
	while ((de = readdir(dr2)) != NULL)
	{
		if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
			continue;
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
		for (j = 0; j < srcSize[i].height * srcSize[i].width * ip_channel; j++)
		{
			input[count2] = ip_image[j];
			input_second[count2] = ip_image_second[j];
			count2++;
		}
		i++;
		break;
	}

	closedir(dr2);



        if(minHeight > srcSize[0].height)
        {
            minHeight = srcSize[0].height;
        }
        if(minWidth > srcSize[0].width)
        {
            minWidth = srcSize[0].width;
        }
    RppiROI roiPoints; 
    while (1)
    {
        roiPoints.x = rand() % minWidth; 
        roiPoints.y = rand() % minHeight; 
        roiPoints.roiHeight = (rand() % minHeight) * 3; 
        roiPoints.roiWidth = (rand() % minWidth) * 3; 
        roiPoints.roiHeight -= roiPoints.y; 
        roiPoints.roiWidth -= roiPoints.x;
        if((roiPoints.y + roiPoints.roiHeight > roiPoints.y && roiPoints.x + roiPoints.roiWidth > roiPoints.x) && (roiPoints.y + roiPoints.roiHeight < minHeight && roiPoints.x + roiPoints.roiWidth < minWidth))
            break;
    }
	cl_mem d_input, d_input_second, d_output;
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
	d_input_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
	d_output = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
	err |= clEnqueueWriteBuffer(theQueue, d_input, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(theQueue, d_input_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input_second, 0, NULL, NULL);
	rppHandle_t handle;

	rppCreateWithStreamAndBatchSize(&handle, theQueue, noOfImages);

	clock_t start, end;
	double cpu_time_used;
	start = clock();

	
	
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
	Rpp32u xRoiEnd = 200;
	Rpp32u yRoiEnd = 200;
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
	Rpp32u numbeoOfShadows = 12;
	Rpp32u maxSizeX = 12;
	Rpp32u maxSizey = 15;
	Rpp32u extractChannelNumber = 1;

	switch (test_case)
	{
		case 0:
			
			rppi_contrast_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, newMin, newMax, roiPoints,handle);
			break;
		case 1:
			rppi_jitter_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, kernelSize, roiPoints,handle);
			break;
		case 2:
			rppi_blur_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, kernelSize, roiPoints,handle);
			break;
		case 3:
			rppi_brightness_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, alpha, beta, roiPoints,handle);
			break;
		case 4:
			rppi_blend_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], d_output, alpha, roiPoints,handle);
			break;
		case 5:
			rppi_color_temperature_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, adjustmentValue, roiPoints,handle);
			break;
		case 6:
			rppi_gamma_correction_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, gamma, roiPoints,handle);
			break;
		case 7:
			rppi_fog_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, fogValue, roiPoints,handle);
			break;
		case 8:
			rppi_snow_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, snowPercentage, roiPoints,handle);
			break;
		case 9:
			rppi_lens_correction_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, strength, zoom, roiPoints,handle);
			break;
		case 10:
			rppi_noise_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, noiseProbability, roiPoints,handle);
			break;
		case 11:
			rppi_pixelate_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, roiPoints,handle);
			break;
		case 12:
			rppi_exposure_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, exposureFactor, roiPoints,handle);
			break;
		case 13:
			rppi_fisheye_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, roiPoints,handle);
			break;
		case 14:
			rppi_vignette_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, stdDev, roiPoints,handle);
			break;
		case 15:
			rppi_flip_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, flipAxis, roiPoints,handle);
			break;
		case 16:
			rppi_rain_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, rainPercentage, rainWidth, rainHeight, transparency, roiPoints,handle);
			break;
		case 17:
			rppi_rotate_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, dstSize[0], angle, roiPoints,handle);
			break;
		case 18:
			rppi_warp_affine_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, dstSize[0], affine, roiPoints,handle);
			break;
		case 19:
			rppi_resize_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, dstSize[0], roiPoints,handle);
			break;
		case 20:
			rppi_resize_crop_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, dstSize[0], coordinates[0], coordinates[1],
										coordinates[2], coordinates[3], roiPoints,handle);
			break;
		case 21:
			rppi_hueRGB_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, hueShift, roiPoints,handle);
			break;
		case 22:
			rppi_saturationRGB_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, saturationFactor, roiPoints,handle);
			break;
		case 23:
			rppi_histogram_balance_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, roiPoints,handle);
			break;
		case 24:
			rppi_random_shadow_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, coordinates[0], coordinates[1],
										coordinates[2], coordinates[3], numbeoOfShadows, maxSizeX, maxSizey, roiPoints,handle);
			break;
		case 25:
			rppi_random_crop_letterbox_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, dstSize[0], coordinates[0], coordinates[1],
												coordinates[2], coordinates[3], roiPoints,handle);
			break;
		case 26:
			rppi_absolute_difference_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], d_output, roiPoints,handle);
			break;
		case 27:
			rppi_accumulate_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], roiPoints,handle);
			break;
		case 28:
			rppi_accumulate_squared_u8_pkd3_ROI_gpu(d_input, srcSize[0], roiPoints,handle);
			break;
		case 29:
			rppi_accumulate_weighted_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], alpha, roiPoints,handle);
			break;
		case 30:
			rppi_add_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], d_output, roiPoints,handle);
			break;
		case 31:
			rppi_subtract_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], d_output, roiPoints,handle);
			break;
		case 32:
			rppi_bitwise_AND_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], d_output, roiPoints,handle);
			break;
		case 33:
			rppi_exclusive_OR_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], d_output, roiPoints,handle);
			break;
		case 34:
			rppi_inclusive_OR_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], d_output, roiPoints,handle);
			break;
		case 35:
			rppi_bitwise_NOT_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, roiPoints,handle);
			break;
		case 36:
			rppi_box_filter_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, kernelSize, roiPoints,handle);
			break;
		case 37:
			rppi_canny_edge_detector_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, minThreshold, maxThreshold, roiPoints,handle);
			break;
		case 38:
			rppi_channel_extract_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, extractChannelNumber, roiPoints,handle);
			break;
		case 39:
			rppi_data_object_copy_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, roiPoints,handle);
			break;
		case 40:
			rppi_dilate_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, kernelSize, roiPoints,handle);
			break;
		case 41:
			rppi_histogram_equalization_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, roiPoints,handle);
			break;
		case 42:
			rppi_erode_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, kernelSize, roiPoints,handle);
			break;
		case 43:
			rppi_fast_corner_detector_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, numOfPixels, threshold, nonmaxKernelSize, roiPoints,handle);
			break;
		case 44:
			rppi_gaussian_filter_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, stdDev, kernelSize, roiPoints,handle);
			break;
		case 45:
			rppi_gaussian_image_pyramid_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, stdDev, kernelSize, roiPoints,handle);
			break;
		case 46:
			rppi_harris_corner_detector_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, gaussianKernelSize, stdDev, kernelSize, kValue, threshold, nonmaxKernelSize, roiPoints,handle);
			break;
		case 47:
			rppi_local_binary_pattern_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, roiPoints,handle);
			break;
		case 48:
			rppi_laplacian_image_pyramid_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, stdDev, kernelSize, roiPoints,handle);
			break;
		case 49:
			rppi_magnitude_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], d_output, roiPoints,handle);
			break;
		case 50:
			ppi_max_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], d_output, roiPoints,handle);
			break;
		case 51:rppi_median_filter_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, kernelSize, roiPoints,handle);
			break;
		case 52:
			rppi_min_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], d_output, roiPoints,handle);
			break;
		case 53:
			rppi_nonlinear_filter_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, kernelSize, roiPoints,handle);
			break;
		case 54:
			rppi_non_max_suppression_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, kernelSize, roiPoints,handle);
			break;
		case 55:
			rppi_phase_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], d_output, roiPoints,handle);
			break;
		case 56:
			rppi_multiply_u8_pkd3_ROI_gpu(d_input, d_input_second, srcSize[0], d_output, roiPoints,handle);
			break;
		case 57:
			rppi_scale_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, dstSize[0], percentage, roiPoints,handle);
			break;
		case 58:
			rppi_sobel_filter_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, sobelType, roiPoints,handle);
			break;
		case 59:
			rppi_thresholding_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, min, max, roiPoints,handle);
			break;
		case 60:
			rppi_warp_perspective_u8_pkd3_ROI_gpu(d_input, srcSize[0], d_output, dstSize[0], perspective, roiPoints,handle);
			break;
		default:
			break;
	}

	end = clock();
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
	cout << " ROI : " << cpu_time_used << endl;

	clEnqueueReadBuffer(theQueue, d_output, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), output, 0, NULL, NULL);

	rppDestroyGPU(handle);

	unsigned long long count3 = 0;
	for (j = 0; j < noOfImages; j++)
	{
		int op_size = srcSize[j].height * srcSize[j].width * ip_channel;
		Rpp8u *temp_output = (Rpp8u *)calloc(op_size, sizeof(Rpp8u));
		for (i = 0; i < op_size; i++)
		{
			temp_output[i] = output[count3];
			count3++;
		}
		char temp[1000];
		strcpy(temp, dst);
		strcat(temp, imageNames[j]);
		Mat mat_op_image;
		if (ip_channel == 3)
		{
			mat_op_image = Mat(srcSize[j].height, srcSize[j].width, CV_8UC3, temp_output);
			imwrite(temp, mat_op_image);
		}
		if (ip_channel == 1)
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
