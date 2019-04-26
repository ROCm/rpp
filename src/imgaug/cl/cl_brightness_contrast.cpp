#define OCL
#ifdef OCL
#include "cl/rpp_cl_common.h"

//File Name : cl_brightness_contrast.cpp
//Has the opencl implementation of rpp brightness and contrast function
int cl_brightness_contrast( unsigned char *sPtr,unsigned int height, unsigned int width,
                        unsigned char *dPtr)
{
    unsigned int n = height * width;
    // Device input buffers
    cl_mem d_input;
    // Device output buffer
    cl_mem d_output;

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *kernel_source;
    size_t source_size;

    fp = fopen("brightness_contrast.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    kernel_source = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(unsigned char);

    size_t globalSize, localSize;
    cl_int err;

    // Number of work items in each local work group
    localSize = LOCAL_SIZE;

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
                            (const char **) & kernel_source, NULL, &err);

    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "brightness", &err);

    // Create the input and output arrays in device memory for our calculation
    d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    //d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);
    //err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                  // bytes, h_b, 0, NULL, NULL);

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    err  = clSetKernelArg(kernel, 2, sizeof(int), &alpha);
    err  = clSetKernelArg(kernel, 3, sizeof(int), &beta);
    err  = clSetKernelArg(kernel, 4, sizeof(unsigned int), &n);

    // Execute the kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Read the results from the device
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
                                bytes, dPtr, 0, NULL, NULL );

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
#endif