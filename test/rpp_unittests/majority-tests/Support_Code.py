non_roi_functions = {"snow","rain","fog","data_object_copy","harris_corner_detector","fast_corner_detector","histogram_balance","histogram_equalization","gaussian_image_pyramid","laplacian_image_pyramid","canny_edge_detector","channel_extract","random_shadow"}

files = {"BatchPD_ROID_Center","BatchPS_ROID_Center","BatchDD_ROID_Center","BatchDS_ROID_Center","BatchSD_ROIS_Center","BatchSS_ROIS_Center","BatchPD_ROID","BatchPS_ROID","BatchDD_ROID","BatchDS_ROID","BatchSD_ROID","BatchSS_ROID","BatchPD_ROIS","BatchPS_ROIS","BatchDD_ROIS","BatchDS_ROIS","BatchSD_ROIS","BatchSS_ROIS","BatchPD","BatchPS","BatchDD","BatchDS","BatchSD","BatchSS","ROI","Single"}

local_folder = '/home/ulagammai/ulagammai/TESTSUITE_RPP/AMDRPP/'

csv_name = '/home/ulagammai/ulagammai/TESTSUITE_RPP/AMDRPP/Test_CSV.csv'

code_folder = '/home/ulagammai/ulagammai/TESTSUITE_RPP/SCRIPT/'

cmake = """cmake_minimum_required (VERSION 2.8)
project (Rpp_test)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenCV REQUIRED )

include_directories (${OpenCV_INCLUDE_DIRS})
include_directories (/opt/rocm/opencl/include/)
include_directories (/opt/rocm/include)
include_directories (/opt/rocm/rpp/include)
link_directories    (/opt/rocm/lib)
link_directories    (/opt/rocm/rpp/lib/)"""

header = """#include <stdio.h> 
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
"""

hip_stater = """
void check_hip_error(void)
{
	hipError_t err = hipGetLastError();
	if (err != hipSuccess)
	{
 		cerr<< "Error: "<< hipGetErrorString(err)<<endl;
		exit(err);
	}
}
"""

main = """
int main(int argc, char **argv)
{
    int ip_channel = G_IP_CHANNEL;
    int mode = G_MODE;
    
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

"""

single_image_1 = """
    Mat image;
    
    struct dirent *de;
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");
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
    }
    closedir(dr);

    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    const int images = noOfImages;
    char imageNames[images][1000];
"""

single_image_2 = """
    Mat image;
    
    struct dirent *de;
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");
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
    const int images = noOfImages;
    char imageNames[images][1000];
"""

double_image_1 = """
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
    }
    closedir(dr);

    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    const int images = noOfImages;
    char imageNames[images][1000];
"""

double_image_2 = """
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
    const int images = noOfImages;
    char imageNames[images][1000];
"""

roi_buffer = """
    RppiROI *roiPoints = (RppiROI *)calloc(noOfImages, sizeof(RppiROI));
"""

non_padding_ioBuffer = """
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
"""
padding_ioBuffer ="""
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
"""

non_padding_io_ending = """
        count++;
    }
    closedir(dr1); 
"""

padding_io_ending = """
        count++;
    }
    closedir(dr1); 

    ioBufferSize = (unsigned long long)maxHeight * (unsigned long long)maxWidth * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
"""
single1 = """
        count++;
        break;
    }
    closedir(dr1); 
"""
center_roi = """
        roiPoints[count].x = srcSize[count].width / 4; 
        roiPoints[count].y = srcSize[count].height / 4; 
        roiPoints[count].roiHeight = srcSize[count].height / 2; 
        roiPoints[count].roiWidth = srcSize[count].width / 2;
"""

different_roi = """
        while(1)
        {
            roiPoints[count].x = rand() % srcSize[count].width; 
            roiPoints[count].y = rand() % srcSize[count].height; 
            roiPoints[count].roiHeight = (rand() % srcSize[count].height) * 3; 
            roiPoints[count].roiWidth = (rand() % srcSize[count].width) * 3;
            roiPoints[count].roiHeight -= roiPoints[count].y; 
            roiPoints[count].roiWidth -= roiPoints[count].x;
            if((roiPoints[count].y + roiPoints[count].roiHeight > roiPoints[count].y && roiPoints[count].x + roiPoints[count].roiWidth > roiPoints[count].x) && (roiPoints[count].y + roiPoints[count].roiHeight < srcSize[count].height && roiPoints[count].x + roiPoints[count].roiWidth < srcSize[count].width))
                break;
        }
"""

single_image_non_padding = """
    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));

    DIR *dr2 = opendir(src);
    count = 0;
    i = 0;
    while ((de = readdir(dr2)) != NULL) 
    {
        if(strcmp(de->d_name,".") == 0 || strcmp(de->d_name,"..") == 0) 
            continue;
        char temp[1000];
        strcpy(temp,src1);
        strcat(temp, de->d_name);        
        if(ip_channel == 3)
        {
            image = imread(temp, 1);
        }
        else
        {
            image = imread(temp, 0);
        }
        Rpp8u *ip_image = image.data;
        for(j = 0 ; j < srcSize[i].height * srcSize[i].width * ip_channel ; j++)
        {
            input[count] = ip_image[j];
            count++;
        }
        i++;
    }
    closedir(dr2); 
"""

double_image_non_padding = """
    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));

    DIR *dr2 = opendir(src);
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
    }
    closedir(dr2); 
"""

single_image_padding = """
    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    RppiSize maxSize;
    maxSize.height = maxHeight;
    maxSize.width = maxWidth;

    DIR *dr2 = opendir(src);
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
        if(ip_channel == 3)
        {
            image = imread(temp, 1);
        }
        else
        {
            image = imread(temp, 0);
        }
        Rpp8u *ip_image = image.data;
        for(j = 0 ; j < srcSize[i].height; j++)
        {
            for(int x = 0 ; x < srcSize[i].width ; x++)
            {
                for(int y = 0 ; y < ip_channel ; y ++)
                {
                    input[count + ((j * maxWidth * ip_channel) + (x * ip_channel) + y)] = ip_image[(j * srcSize[i].width * ip_channel) + (x * ip_channel) + y];
                }
            }
        }
        i++;
    }
    closedir(dr2); 
"""

double_image_padding = """
    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    RppiSize maxSize;
    maxSize.height = maxHeight;
    maxSize.width = maxWidth;

    /* Read the input image */
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
"""

single2 = """
    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));

    DIR *dr2 = opendir(src);
    count = 0;
    i = 0;
    while ((de = readdir(dr2)) != NULL) 
    {
        if(strcmp(de->d_name,".") == 0 || strcmp(de->d_name,"..") == 0) 
            continue;
        char temp[1000];
        strcpy(temp,src1);
        strcat(temp, de->d_name);
        if(ip_channel == 3)
        {
            image = imread(temp, 1);
        }
        else
        {
            image = imread(temp, 0);
        }
        Rpp8u *ip_image = image.data;
        for(j = 0 ; j < srcSize[i].height * srcSize[i].width * ip_channel ; j++)
        {
            input[count] = ip_image[j];
            
            count++;
        }
        i++;
        break;
    }
    closedir(dr2); 
"""

single3 = """
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
"""

same_roi = """
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
"""
roi_patch = """
        if(minHeight > srcSize[0].height)
        {
            minHeight = srcSize[0].height;
        }
        if(minWidth > srcSize[0].width)
        {
            minWidth = srcSize[0].width;
        }
"""
rois_patch = """
        if(minHeight > srcSize[count].height)
        {
            minHeight = srcSize[count].height;
        }
        if(minWidth > srcSize[count].width)
        {
            minWidth = srcSize[count].width;
        }
"""
same_center_roi = """
    RppiROI roiPoints;
    roiPoints.x = minWidth / 4; 
    roiPoints.y = minHeight / 4; 
    roiPoints.roiHeight = (minHeight) / 2; 
    roiPoints.roiWidth = (minWidth) / 2;
"""

hip_single = """
	int *d_input, *d_output;
	hipMalloc(&d_input, ioBufferSize * sizeof(Rpp8u));
	hipMalloc(&d_output, ioBufferSize * sizeof(Rpp8u));
	check_hip_error();
	hipMemcpy(d_input, input, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
	check_hip_error();

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

	clock_t start, end;   
	double cpu_time_used;
	start = clock();
 
"""

hip_double = """
 	int *d_input, *d_input_second, *d_output;
	hipMalloc(&d_input, ioBufferSize * sizeof(Rpp8u));
	hipMalloc(&d_input_second, ioBufferSize * sizeof(Rpp8u));	
	hipMalloc(&d_output, ioBufferSize * sizeof(Rpp8u));
	check_hip_error();
	hipMemcpy(d_input, input, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
	hipMemcpy(d_input_second, input_second, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
	check_hip_error();

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

	clock_t start, end;   
	double cpu_time_used;
	start = clock();
 
"""

ocl_single = """
	cl_mem d_input, d_output;
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
    d_output = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_input, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input, 0, NULL, NULL);
	rppHandle_t handle;
   
	rppCreateWithStreamAndBatchSize(&handle, theQueue, noOfImages);

	clock_t start, end;   
	double cpu_time_used;
	start = clock();

"""

ocl_double = """
	cl_mem d_input, d_input_second, d_output;
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
    d_input_second = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
    d_output = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_input, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_input_second, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input_second, 0, NULL, NULL);
	rppHandle_t handle;
   
	rppCreateWithStreamAndBatchSize(&handle, theQueue, noOfImages);
 
	clock_t start, end;   
	double cpu_time_used;
	start = clock();

"""

host_timing = """
	rppHandle_t handle;
	rppCreateWithBatchSize(&handle, noOfImages);
 
	clock_t start, end;   
	double cpu_time_used;
	start = clock();
"""

timing_end_1 = """
    end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	cout<<" """

timing_end_2 = """ : "<<cpu_time_used<<endl;  
"""

ocl_copy = """
	clEnqueueReadBuffer(theQueue, d_output, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), output, 0, NULL, NULL);

	rppDestroyGPU(handle);

"""

hip_copy = """
	hipMemcpy(output,d_output,ioBufferSize * sizeof(Rpp8u),hipMemcpyDeviceToHost);

	rppDestroyGPU(handle);
"""

host_copy = """
rppDestroyHost(handle);
"""

write_image = """
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
"""

write_image_padding = """
    count = 0;
    for(j = 0 ; j < noOfImages ; j++)
    {
        int op_size = maxHeight * maxWidth * ip_channel;
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
            mat_op_image = Mat(maxHeight, maxWidth, CV_8UC3, temp_output);
            imwrite(temp, mat_op_image);
        }
        if(ip_channel == 1)
        {
            mat_op_image = Mat(maxHeight, maxWidth, CV_8UC1, temp_output);
            imwrite(temp, mat_op_image);
        }
        free(temp_output);
    }
"""

free_mem = """
    free(srcSize);
    free(input);
    free(output);
"""

IP_cmake_1 = "add_executable("

IP_cmake_2 = ")\n"

IP_cmake_3 = "\ntarget_link_libraries("

IP_cmake_4 = " ${OpenCV_LIBS} -I/opt/rocm/rpp/include -L/opt/rocm/rpp/lib/ -lamd_rpp -L/opt/rocm/opencl/lib/x86_64/ -lOpenCL pthread  -lboost_filesystem -lboost_system )"

IP_cmake_3_hip = "\ntarget_link_libraries("

IP_cmake_4_hip = " ${OpenCV_LIBS} -I/opt/rocm/rpp/include/ -I/opt/rocm/opencl/include/ -I/opt/rocm/include/ -L/opt/rocm/rpp/lib/ -lamd_rpp -lboost_filesystem -lboost_system -L/opt/rocm/hip/lib/ -lhiprtc)"
