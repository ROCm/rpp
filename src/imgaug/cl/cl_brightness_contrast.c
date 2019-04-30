#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rppdefs.h"
#include <CL/opencl.h>
#define C L_USE_DEPRECATED_OPENCL_1_2_APIS

const char *kernelSource =                                       "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void brightness(  __global double *a,                       \n" \
"                       __global double *c,                       \n"\
"                         int alpha,                      \n" \
"                         int beta,                       \n"\
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = alpha * a[id] + beta;                                  \n" \
"}                                                               \n" \
                                                                "\n" ;

RppStatus cl_brightness_contrast( Rpp8u* pSrc, unsigned int height, unsigned int width, Rpp8u* pDst)
{
    // Length of vectors
    unsigned int n = 10000000;

    // Host input vectors
    double *h_a;
    double *h_b;
    // Host output vector
    double *h_c;
    int alpha = 2;
    int beta = 1;

    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    //h_b = (doauble*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    // Initialize vectors on host
    int i;
    for( i = 0; i < n; i++ )
    {
        h_a[i] = i;
        //h_b[i] = cosf(i)*cosf(i);
    }

    size_t globalSize, localSize;
    cl_int err;

    // Number of work items in each local work group
    localSize = 64;

    // Number of total work items - localSize must be devisor
    globalSize = ceil(n/(float)localSize)*localSize;

    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);

    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);

    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "brightness", &err);

    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    //d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);
    //err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                  // bytes, h_b, 0, NULL, NULL);

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    //err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &alpha);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &beta);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &n);

    // Execute the kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                bytes, h_c, 0, NULL, NULL );

    //Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<10; i++)
    printf("final result: %f\n",h_c[i]);

   // printf("final result: %f\n", sum/n);

    // release OpenCL resources
    clReleaseMemObject(d_a);
   // clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    //release host memory
    free(h_a);
    //free(h_b);
    free(h_c);

    return RPP_SUCCESS;
}
