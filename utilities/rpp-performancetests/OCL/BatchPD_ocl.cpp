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
#include <CL/cl.hpp>
#include <half.hpp>

using namespace cv;
using namespace std;
using half_float::half;

#define images 100

int G_IP_CHANNEL = 3;

int main(int argc, char **argv)
{
	int ip_channel = G_IP_CHANNEL;

	const int MIN_ARG_COUNT = 5;
	
	if (argc < MIN_ARG_COUNT)
	{
		printf("\nImproper Usage! Needs all arguments!\n");
		printf("\nUsage: ./BatchPD_hip <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <case number = 0:64> <verbosity = 0/1>\n");
		printf("\nAdditional Note - There are no output images for rpp-performancetests - Only max, min, avg performance times for 1000 batches - A destination folder path however needs to be passed\n");
		return -1;
	}

	if (atoi(argv[5]) == 1)
	{
		printf("\nInputs for this test case are:");
		printf("\nsrc1 = %s", argv[1]);
		printf("\nsrc2 = %s", argv[2]);
		printf("\ndst = %s", argv[3]);
		printf("\ncase number (0:64) = %s\n", argv[4]);
	}

	char *src = argv[1];
	char *src_second = argv[2];
	char *dst = argv[3];
	char *funcName = argv[4];
	char *funcNameNumber = argv[4];
	int test_case = atoi(argv[4]);

	char funcType[1000] = {"BatchPD_OCL_PKD3"};

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
	int minDstHeight = 30000, minDstWidth = 30000, maxDstHeight = 0, maxDstWidth = 0;

	unsigned long long count = 0;
	unsigned long long ioBufferSize = 0;
	unsigned long long oBufferSize = 0;
	static int noOfImages = 0;
	Mat image,image_second;
	
	struct dirent *de;
	char src1[1000];
	strcpy(src1, src);
	strcat(src1, "/");
	char src1_second[1000];
	strcpy(src1_second, src_second);
	strcat(src1_second, "/");
	strcat(funcType, "_");
	strcat(funcType, funcName);
	strcat(dst,"/");
	strcat(dst,funcType);
	

	DIR *dr = opendir(src); 
	while ((de = readdir(dr)) != NULL) 
	{
		if(strcmp(de->d_name,".") == 0 || strcmp(de->d_name,"..") == 0)
			continue;
		noOfImages += 1;
	}
	closedir(dr);

	RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
	RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));

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
		if(maxHeight < srcSize[count].height)
			maxHeight = srcSize[count].height;
		if(maxWidth < srcSize[count].width)
			maxWidth = srcSize[count].width;
		if(minHeight > srcSize[count].height)
			minHeight = srcSize[count].height;
		if(minWidth > srcSize[count].width)
			minWidth = srcSize[count].width;
		
		dstSize[count].height = image.rows;
		dstSize[count].width = image.cols;
		if(maxDstHeight < dstSize[count].height)
			maxDstHeight = dstSize[count].height;
		if(maxDstWidth < dstSize[count].width)
			maxDstWidth = dstSize[count].width;
		if(minDstHeight > dstSize[count].height)
			minDstHeight = dstSize[count].height;
		if(minDstWidth > dstSize[count].width)
			minDstWidth = dstSize[count].width;

		count++;
	}
	closedir(dr1); 

	ioBufferSize = (unsigned long long)maxHeight * (unsigned long long)maxWidth * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
	oBufferSize = (unsigned long long)maxDstHeight * (unsigned long long)maxDstWidth * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

	Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
	Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
	Rpp8u *output = (Rpp8u *)calloc(oBufferSize, sizeof(Rpp8u));
	RppiSize maxSize,maxDstSize;
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
		if(strcmp(de->d_name,".") == 0 || strcmp(de->d_name,"..") == 0) 
			continue;
		count = (unsigned long long)i * (unsigned long long)maxHeight * (unsigned long long)maxWidth * (unsigned long long)ip_channel;
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
		for(j = 0 ; j < srcSize[i].height; j++)
		{
			for(int x = 0 ; x < srcSize[i].width ; x++)
			{
				for(int y = 0 ; y < ip_channel ; y ++)
				{
					input[count + ((j * maxWidth * ip_channel) + (x * ip_channel) + y)] = ip_image[(j * srcSize[i].width * ip_channel) + (x * ip_channel) + y];
					input_second[count + ((j * maxWidth * ip_channel) + (x * ip_channel) + y)] = ip_image_second[(j * srcSize[i].width * ip_channel) + (x * ip_channel) + y];
				}
			}
		}
		i++;
	}
	closedir(dr2);
	Rpp32u newMin[images];
	Rpp32u newMax[images];
	for(i = 0 ; i < images ; i++)
	{
		newMin[i] = 30;
		newMax[i] = 100;
	}

	Rpp32u kernelSize[images];
	Rpp32u numbeoOfShadows[images];
	Rpp32u maxSizeX[images];
	Rpp32u maxSizey[images];
	for(i = 0 ; i < images ; i++)
	{
		kernelSize[i]  = 5;
			numbeoOfShadows[i] = 10;
		maxSizeX[i] = 12;
		maxSizey[i] = 15;
	}
	   Rpp32f stdDev[images];
	for(i = 0 ; i < images ; i++)
	{
		stdDev[i] = 75.0;
	}
	Rpp32f snowPercentage[images];
	for(i = 0 ; i < images ; i++)
	{
		snowPercentage[i] = 0.8;
	}
	Rpp32f rainPercentage[images];
	Rpp32u rainWidth[images];
	Rpp32u rainHeight[images];
	Rpp32f transparency[images];
	for (i = 0; i < images; i++)
	{
		rainPercentage[i] = 1.0;
		rainWidth[i] = 1;
		rainHeight[i] = 12;
		transparency[i] = 1;
	}
	Rpp32f noiseProbability[images];
	for(i = 0 ; i < images ; i++)
	{
		noiseProbability[i] = 0.2;
	}
	
	Rpp32f strength[images];
	Rpp32f zoom[images];
	for(i = 0 ; i < images ; i++)
	{
		strength[i] = 0.5;
		zoom[i] = 1;
	}
	Rpp32f gamma[images];
	for(i = 0 ; i < images ; i++)
	{
		gamma[i] = 1.9;
	}
	Rpp32f fogValue[images];
	for(i = 0 ; i < images ; i++)
	{
		fogValue[i] = 0.2;
	}
	Rpp32u flipAxis[images];
	for(i = 0 ; i < images ; i++)
	{
		flipAxis[i] = 2;

	}
	Rpp32f exposureFactor[images];
	for(i = 0 ; i < images ; i++)
	{
		exposureFactor[i] = 1.4;
	}
	Rpp32s adjustmentValue[images];
	Rpp32u extractChannelNumber[images];
	Rpp32u extractChannelNumber1[images];
	Rpp32u extractChannelNumber2[images];
	Rpp32u extractChannelNumber3[images];
	for(i = 0 ; i < images ; i++)
	{
		adjustmentValue[i] = 70;
		extractChannelNumber[i] = 1;
		
		extractChannelNumber1[i] = 0;
		extractChannelNumber2[i] = 1;
		extractChannelNumber3[i] = 2;
	}
	Rpp32f alpha[images];
	Rpp32f beta[images];
	for(i = 0 ; i < images ; i++)
	{
		alpha[i] = 0.7;
		beta[i] = 0;
	}
	Rpp32f angle[images];
	for(i = 0 ; i < images ; i++)
	{
		angle[i] = 50;
	}

	Rpp32f affine_array[6*images];
	for (i = 0; i < 6 * images; i = i + 6)
	{
		affine_array[i] = 1.23;
		affine_array[i + 1] = 0.5;
		affine_array[i + 2] = 0.0;
		affine_array[i + 3] = -0.8;
		affine_array[i + 4] = 0.83;
		affine_array[i + 5] = 0.0;
	}
	
	Rpp32u x1[images];
	Rpp32u x2[images];
	Rpp32u y1[images];
	Rpp32u y2[images];

	for(i = 0; i < images; i++)
	{
	   x1[i] = 0;
	   x2[i] = 50;
	   y1[i] = 0;
	   y2[i] = 50;
	}
	Rpp32u xRoiBegin[images];
	Rpp32u xRoiEnd[images];
	Rpp32u yRoiBegin[images];
	Rpp32u yRoiEnd[images];
	Rpp32u mirrorFlag[images];

	for(i = 0 ; i < images ; i++)
	{
		xRoiBegin[i] = 50;
		yRoiBegin[i] = 50;
		xRoiEnd[i]  = 150;
		yRoiEnd[i]  = 150;
		mirrorFlag[i] = 0;
	}
	Rpp32u crop_pos_x[images];
	Rpp32u crop_pos_y[images];
	for(i = 0 ; i < images ; i++)
	{
		crop_pos_x[i] = 50;
		crop_pos_y[i] = 50;
	}
	Rpp32f hueShift[images];
	Rpp32f saturationFactor[images];
	for(i = 0 ; i < images ; i++)
	{
		
		hueShift[i] = 60;
		saturationFactor[i] = 5;
	}
	Rpp8u minThreshold[images];
	Rpp8u maxThreshold[images];
	for(i = 0 ; i < images ; i++)
	{
		minThreshold[i] = 10;
		maxThreshold[i] = 30;
	}
	Rpp32u numOfPixels[images];
	Rpp32u gaussianKernelSize[images];
	Rpp32f kValue[images];
	Rpp8u threshold[images];
	Rpp32f threshold1[images];
	Rpp32u nonmaxKernelSize[images];
	for(i = 0 ; i < images ; i++)
	{
		numOfPixels[i] = 4;
		gaussianKernelSize[i] = 7;
		kValue[i] = 1;
		threshold[i] = 15;
				threshold1[i] = 10;
		nonmaxKernelSize[i] = 5;
	}
	Rpp32u sobelType[images];
	for(i = 0 ; i < images ; i++)
	{
		sobelType[i] = 1;
	}
	Rpp8u min[images];
	Rpp8u max[images];
	for(i = 0 ; i < images ; i++)
	{
		min[i] = 10;
		max[i] = 30;
	}
	const int size_perspective = images * 9;
	Rpp32f perspective[size_perspective];
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
	Rpp32f percentage[images];
	for(i = 0 ; i < images ; i++)
	{
		percentage[i] = 0.75;
	}
	Rpp32f cmn_mean[images];
	Rpp32f cmn_stddev[images];
	for(i = 0 ; i < images ; i++)
	{
		cmn_mean[i] = 0;
		cmn_stddev[i] = 1;
	}
	Rpp32u outputFomatToggle = 0;

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
	d_output = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp8u), NULL, NULL);
	err |= clEnqueueWriteBuffer(theQueue, d_input, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(theQueue, d_input_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input_second, 0, NULL, NULL);

	rppHandle_t handle;
	rppCreateWithStreamAndBatchSize(&handle, theQueue, noOfImages);
	string test_case_name;

	if (
		(test_case == 19) || (test_case == 20) || 
		(test_case == 62) || (test_case == 63)
		)
	{
		for(i = 0 ; i < noOfImages ; i++)
		{
			dstSize[i].height = srcSize[i].height / 2;
			dstSize[i].width = srcSize[i].width / 3;
			if(maxDstHeight < dstSize[i].height)
				maxDstHeight = dstSize[i].height;
			if(maxDstWidth < dstSize[i].width)
				maxDstWidth = dstSize[i].width;
			if(minDstHeight > dstSize[i].height)
				minDstHeight = dstSize[i].height;
			if(minDstWidth > dstSize[i].width)
				minDstWidth = dstSize[i].width;
		}
		maxDstSize.height = maxDstHeight;
		maxDstSize.width = maxDstWidth;
		oBufferSize = (unsigned long long)maxDstHeight * (unsigned long long)maxDstWidth * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
		output = (Rpp8u *)calloc(oBufferSize, sizeof(Rpp8u));
		d_output = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp8u), NULL, NULL);
	}

	clock_t start, end;   
	double cpu_time_used[1000];
	double max_time_used = 0, min_time_used = 500, avg_time_used = 0;
	
	for (int i = 0; i < 1000; i++)
	{
		start = clock();
		switch (test_case)
		{
			case 0:
				test_case_name = "contrast";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_contrast_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, newMin, newMax, noOfImages, handle);
				break;
			case 1:
				test_case_name = "jitter";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_jitter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
				break;
			case 2:
				test_case_name = "blur";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_blur_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
				break;
			case 3:
				test_case_name = "brightness";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_brightness_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, alpha, beta, noOfImages, handle);
				break;
			case 4:
				test_case_name = "blend";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_blend_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, alpha, noOfImages, handle);
				break;
			case 5:
				test_case_name = "color_temperature";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_color_temperature_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, adjustmentValue, noOfImages, handle);
				break;
			case 6:
				test_case_name = "gamma_correction";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_gamma_correction_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, gamma, noOfImages, handle);
				break;
			case 7:
				test_case_name = "fog";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_fog_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, fogValue, noOfImages, handle);
				break;
			case 8:
				test_case_name = "snow";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_snow_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, snowPercentage, noOfImages, handle);
				break;
			case 9:
				test_case_name = "lens_correction";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_lens_correction_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, strength, zoom, noOfImages, handle);
				break;
			case 10:
				test_case_name = "noise";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_noise_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, noiseProbability, noOfImages, handle);
				break;
			case 11:
				test_case_name = "pixelate";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_pixelate_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 12:
				test_case_name = "exposure";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_exposure_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, exposureFactor, noOfImages, handle);
				break;
			case 13:
				test_case_name = "fisheye";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_fisheye_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 14:
				test_case_name = "vignette";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_vignette_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, stdDev, noOfImages, handle);
				break;
			case 15:
				test_case_name = "flip";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_flip_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, flipAxis, noOfImages, handle);
				break;
			case 16:
				test_case_name = "rain";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_rain_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, rainPercentage, rainWidth, rainHeight, transparency, noOfImages, handle);
				break;
			case 17:
				test_case_name = "rotate";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_rotate_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxSize, angle,outputFomatToggle, noOfImages, handle);
				break;
			case 18:
				test_case_name = "warp-affine";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_warp_affine_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxSize, affine_array, noOfImages, handle);
				break;
			case 19:
				test_case_name = "resize";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_resize_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize,outputFomatToggle, noOfImages, handle);
				break;
			case 20:
				test_case_name = "resize-crop";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_resize_crop_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, x1, x2, y1, y2,outputFomatToggle, noOfImages, handle);
				break;
			case 21:
				test_case_name = "Hue modification";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_hueRGB_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, hueShift, noOfImages, handle);
				break;
			case 22:
				test_case_name = "Saturation";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_saturationRGB_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, saturationFactor, noOfImages, handle);
				break;
			case 23:
				test_case_name = "Histogram Balance";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_histogram_balance_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 24:
				test_case_name = "RandomShadow";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_random_shadow_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, x1, y1, x2, y2, numbeoOfShadows, maxSizeX, maxSizey, noOfImages, handle);
				break;
			case 25:
				test_case_name = "RandomCropLetterBox";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_random_crop_letterbox_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, x1, x2, y1, y2, noOfImages, handle);
				break;
			case 26:
				test_case_name = "Absolute Difference";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_absolute_difference_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 27:
				test_case_name = "Accumulate";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_accumulate_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, noOfImages, handle);
				err |= clEnqueueCopyBuffer(theQueue, d_input, d_output, 0, 0, oBufferSize * sizeof(Rpp8u), 0, NULL, NULL);
				break;
			case 28:
				test_case_name = "Accumulate Squared";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_accumulate_squared_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize,  noOfImages, handle);
				err |= clEnqueueCopyBuffer(theQueue, d_input, d_output, 0, 0, oBufferSize * sizeof(Rpp8u), 0, NULL, NULL);
				break;
			case 29:
				test_case_name = "Accumulate Weighted";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_accumulate_weighted_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize,alpha, noOfImages, handle);
				err |= clEnqueueCopyBuffer(theQueue, d_input, d_output, 0, 0, oBufferSize * sizeof(Rpp8u), 0, NULL, NULL);
				break;
			case 30:
				test_case_name = "Arithmetic Addition";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 31:
				test_case_name = "Arithmetic Subtraction";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_subtract_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 32:
				test_case_name = "Bitwise AND";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_bitwise_AND_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 33:
				test_case_name = "Bitwise EXCLUSIVE OR";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_exclusive_OR_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 34:
				test_case_name = "Bitwise INCLUSIVE OR";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_inclusive_OR_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 35:
				test_case_name = "Bitwise NOT";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_bitwise_NOT_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 36:
				test_case_name = "Box Filter";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_box_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
				break;
			case 37:
				test_case_name = "Canny Edge Detector";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_canny_edge_detector_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, minThreshold, maxThreshold, noOfImages, handle);
				break;
			case 38:
				test_case_name = "Channel Extract";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_channel_extract_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, extractChannelNumber, noOfImages, handle);
				break;
			case 39:
				test_case_name = "Data Object Copy";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_data_object_copy_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 40:
				test_case_name = "Dilate Image";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_dilate_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
				break;
			case 41:
				test_case_name = "Equalize Histogram";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_histogram_equalization_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 42:
				test_case_name = "Erode Image";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_erode_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
				break;
			case 43:
				test_case_name = "Fast Corners";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_fast_corner_detector_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, numOfPixels, threshold, nonmaxKernelSize, noOfImages, handle);
				break;
			case 44:
				test_case_name = "Gaussian Filter";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_gaussian_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, stdDev, kernelSize, noOfImages, handle);
				break;
			case 45:
				test_case_name = "Gaussian Image Pyramid";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_gaussian_image_pyramid_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, stdDev, kernelSize, noOfImages, handle);
				break;
			case 46:
				test_case_name = "Harris Corners";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_harris_corner_detector_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, gaussianKernelSize, stdDev, kernelSize, kValue, threshold1, nonmaxKernelSize, noOfImages, handle);
				break;
			case 47:
				test_case_name = "LBP";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_local_binary_pattern_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 48:
				test_case_name = "Laplacian Image Pyramid";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_laplacian_image_pyramid_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, stdDev, kernelSize, noOfImages, handle);
				break;
			case 49:
				test_case_name = "Magnitude";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_magnitude_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 50:
				test_case_name = "Max";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_max_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 51:
				test_case_name = "Median Filter";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_median_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
				break;
			case 52:
				test_case_name = "Min";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_min_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 53:
				test_case_name = "Non Linear Filter";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_nonlinear_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
				break;
			case 54:
				test_case_name = "Non-Maxima Suppression";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_non_max_suppression_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
				break;
			case 55:
				test_case_name = "Phase";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_phase_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 56:
				test_case_name = "Pixel-wise Multiplication";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_multiply_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
				break;
			case 57:
				test_case_name = "Scale Image";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_scale_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, percentage, noOfImages, handle);
				break;
			case 58:
				test_case_name = "Sobel 3x3";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_sobel_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, sobelType, noOfImages, handle);
				break;
			case 59:
				test_case_name = "Thresholding";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_thresholding_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, min, max, noOfImages, handle);
				break;
			case 60:
				test_case_name = "Warp Perspective";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_warp_perspective_u8_pkd3_batchPD_gpu(d_input, srcSize,maxSize, d_output, dstSize,maxDstSize, perspective,noOfImages, handle);
				break;
			case 61:
				test_case_name = "resize-crop-mirror";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_resize_crop_mirror_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, xRoiBegin,xRoiEnd, yRoiBegin,yRoiEnd,mirrorFlag,outputFomatToggle, noOfImages, handle);
				break;
			case 62:
				test_case_name = "crop";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_crop_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, xRoiBegin, yRoiBegin,outputFomatToggle, noOfImages, handle);
				break;
			case 63:
				test_case_name = "crop - mirror - normalize";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_crop_mirror_normalize_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, crop_pos_x, crop_pos_y, cmn_mean, cmn_stddev, mirrorFlag, outputFomatToggle, noOfImages, handle);
				break;
			case 64:
				test_case_name = "color-twist";
				// std::cout << "\n"<< test_case_name << "\n";
				rppi_color_twist_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, alpha, beta, hueShift, saturationFactor, outputFomatToggle,noOfImages, handle);
				break;
			default:
				break;
		
		}
		end = clock();
		cpu_time_used[i] = ((double) (end - start)) / CLOCKS_PER_SEC;
		if (cpu_time_used[i] > max_time_used)
			max_time_used = cpu_time_used[i];
		if (cpu_time_used[i] < min_time_used)
			min_time_used = cpu_time_used[i];
		avg_time_used += cpu_time_used[i];
	}

	avg_time_used /= 1000;
	cout << "\n BatchPD performance for processing 1000 batches:";
	cout << "\n max, min, avg = " << max_time_used << ", " << min_time_used << ", " << avg_time_used << endl;

	clEnqueueReadBuffer(theQueue, d_output, CL_TRUE, 0, oBufferSize * sizeof(Rpp8u), output, 0, NULL, NULL);

	rppDestroyGPU(handle);

	free(srcSize);
	free(input);
	free(output);
	free(input_second);
	return 0; 
}
