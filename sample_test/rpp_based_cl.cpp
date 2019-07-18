//==============================================================================
// Only compiled with hipcc compiler
//==============================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <CL/cl.hpp>
#include <rpp.h>
#include<rppi.h>
//#include<rppi_geometric_functions.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define CL_USE_DEPRECATED_OPENCL_1_2_APIS


int main( int argc, char* argv[] )
{
    typedef unsigned char TYPE_t;
    TYPE_t* h_a;
    TYPE_t* h_b;
    TYPE_t* h_c;
    int height;
    int width;
    int channel;
    Rpp32f alpha=0.5;

    RppiGaussParameter np;
    Rpp32f mean=20;
    Rpp32f sigma=30;
    np.mean=mean;
    np.sigma=sigma;
    RppiNoise nt=GAUSSIAN;

    Rpp32f noiseParameter=0.2;
    RppiNoise noiseType=SNP;

    Rpp32f temp=-40;
    Rpp32f exposureValue=0;

    h_a = stbi_load( "/home/mcw/Desktop/AMDRPP/sample_test/images/Image2.jpg",
                        &width, &height, &channel, 0);
    h_b = stbi_load( "/home/mcw/Desktop/AMDRPP/sample_test/images/Image2.jpg",
                        &width, &height, &channel, 0);
    size_t n = height * width * channel;
    size_t bytes = n*sizeof(TYPE_t);
    
    Rpp32u x1=100;Rpp32u y1=100;Rpp32u x2=700;Rpp32u y2=400;

    std::cout << "width:" << width << std::endl;
    std::cout << "height:" << height << std::endl;
    std::cout << "channel:" << channel << std::endl;



    RppiSize dstSize;
    // dstSize.height=y2-y1+6;
    // dstSize.width=x2-x1+6;
    dstSize.height=height;
    dstSize.width=width;

    size_t dest_bytes = dstSize.width * dstSize.height * channel* sizeof(TYPE_t);
    h_c = (TYPE_t*)malloc(dest_bytes);


//------ CL Alloc Stuffs
    cl_mem d_a;
    cl_mem d_b;
    cl_platform_id platform_id;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_mem d_c;
    cl_context theContext;               // theContext
    cl_command_queue theQueue;           // command theQueue
    cl_program theProgram;               // theProgram
    cl_kernel theKernel;                 // theKernel

    cl_int err;
    err = clGetPlatformIDs(1, &platform_id, NULL);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    theContext = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    theQueue = clCreateCommandQueue(theContext, device_id, 0, &err);


    d_a = clCreateBuffer(theContext, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(theContext, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, dest_bytes, NULL, NULL);

    err = clEnqueueWriteBuffer(theQueue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_b, CL_TRUE, 0,
                                   bytes, h_b, 0, NULL, NULL);

   

    RppiSize srcSize;
    srcSize.height=height;
    srcSize.width=width;

    // std::cout<<"\n\nINTO PROGRAM \n\n";
    
    rppi_fog_u8_pkd3_gpu(d_b, srcSize, d_c, exposureValue, theQueue);

    // std::cout<<"\n\nOUT OF PROGRAM SAVING IMAGE \n\n";

    clEnqueueReadBuffer(theQueue, d_c, CL_TRUE, 0,
                               dest_bytes, h_c, 0, NULL, NULL );

    stbi_write_png("/home/mcw/Desktop/AMDRPP/sample_test/images/FOG_GPU.png",
                           dstSize.width,dstSize.height, channel, h_c, dstSize.width *channel);

    //std::cout<<"\n\nSAVED IMAGE \n\n";

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);


    clReleaseCommandQueue(theQueue);
    clReleaseContext(theContext);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
    