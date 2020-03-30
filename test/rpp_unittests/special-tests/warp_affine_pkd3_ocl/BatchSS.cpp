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
int G_IP_CHANNEL = 3;
int G_MODE = 1;
char src[1000] = {"/home/ulagammai/ulagammai/TESTSUITE_RPP/Input_Images/RGBS"};
char dst[1000] = {"/home/ulagammai/ulagammai/TESTSUITE_RPP/Output_Images_OCL"};
 char funcName[1000] = {"warp_affine"};
char funcType[1000] = {"BatchSS"};

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
    int minDstHeight = 30000, minDstWidth = 30000, maxDstHeight = 0, maxDstWidth = 0;

    unsigned long long count = 0;
    unsigned long long ioBufferSize = 0;
    unsigned long long oBufferSize = 0;
    
    static int noOfImages = 0;


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
    RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    const int images = noOfImages;
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
        dstSize[count].height = image.rows + 350;
        dstSize[count].width = image.cols + 250;
        ioBufferSize += (unsigned long long)srcSize[count].height * (unsigned long long)srcSize[count].width * (unsigned long long)ip_channel;
        oBufferSize += (unsigned long long)dstSize[count].height * (unsigned long long)dstSize[count].width * (unsigned long long)ip_channel;

        count++;
    }
    closedir(dr1); 

    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(oBufferSize, sizeof(Rpp8u));

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
	Rpp32f affine[64 * 6];
    for(i = 0 ; i < images ; i++)
	{
        affine[0 + i * 6] = 1;
        affine[1 + i * 6] = 0.5;
        affine[2 + i * 6] = 0;
        affine[3 + i * 6] = 0.5;
        affine[4 + i * 6] = 1;
        affine[5 + i * 6] = 0;
	}
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
    d_output = clCreateBuffer(theContext, CL_MEM_READ_ONLY, oBufferSize * sizeof(Rpp8u), NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_input, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input, 0, NULL, NULL);
	rppHandle_t handle;
   
	rppCreateWithStreamAndBatchSize(&handle, theQueue, noOfImages);

	clock_t start, end;   
	double cpu_time_used;
	start = clock();

	rppi_warp_affine_u8_pkd3_batchSS_gpu(d_input, srcSize[0], d_output, dstSize[0], affine,noOfImages, handle);

    end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	cout<<" BatchSS : "<<cpu_time_used<<endl;  

	clEnqueueReadBuffer(theQueue, d_output, CL_TRUE, 0, oBufferSize * sizeof(Rpp8u), output, 0, NULL, NULL);

	rppDestroyGPU(handle);


    count = 0;
    for(j = 0 ; j < noOfImages ; j++)
    {
        int op_size = dstSize[j].height * dstSize[j].width * ip_channel;
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
            mat_op_image = Mat(dstSize[j].height, dstSize[j].width, CV_8UC3, temp_output);
            imwrite(temp, mat_op_image);
        }
        if(ip_channel == 1)
        {
            mat_op_image = Mat(dstSize[j].height, dstSize[j].width, CV_8UC1, temp_output);
            imwrite(temp, mat_op_image);
        }
        free(temp_output);
    }

    free(srcSize);
    free(dstSize);
    free(input);
    free(output);
	 return 0; 
}
