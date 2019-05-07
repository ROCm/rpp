#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include "rppdefs.h"

// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                      "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                   \n" \
"__kernel void vecAdd(  __global Rpp8u *a,                       \n" \
"                       __global Rpp8u *b,                       \n" \
"                       const unsigned int n)                    \n" \
"                       const Rpp32f alpha,                      \n" \
"                       const Rpp32f beta,                       \n" \

"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        b[id] = a[id] * alpha + beta;                           \n" \
"}                                                               \n" \
;


RppStatus cl_brightness_contrast( Rpp8u* srcPtr, unsigned int height, unsigned int width,
                                  Rpp8u* dstPtr, Rpp32f alpha, Rpp32f beta)
{
    unsigned int n = height * width;
    // Device input buffers
    cl::Buffer d_a;
    cl::Buffer d_b;
     // Size, in bytes, of each vector
    size_t bytes = n*sizeof(unsigned char);
    cl_int err = CL_SUCCESS;
     try {

        // Query platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "Platform size 0\n";
            return RPP_ERROR;
         }
    // Get list of devices on default platform and create context
        cl_context_properties properties[] =
           { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
         // Create command queue for first device
        cl::CommandQueue queue(context, devices[0], 0, &err);
     // Create device memory buffers
        d_a = cl::Buffer(context, CL_MEM_READ_ONLY, bytes);
        d_b = cl::Buffer(context, CL_MEM_WRITE_ONLY, bytes);

         // Bind memory buffers
        queue.enqueueWriteBuffer(d_a, CL_TRUE, 0, bytes, pSrc);

        //Build kernel from source string
        cl::Program::Sources source(1,
            std::make_pair(kernelSource,strlen(kernelSource)));
        cl::Program program_ = cl::Program(context, source);
        program_.build(devices);

         // Create kernel object
        cl::Kernel kernel(program_, "vecAdd", &err);
        // Bind kernel arguments to kernel
        kernel.setArg(0, d_a);
        kernel.setArg(1, d_b);
        kernel.setArg(2, n);
        kernel.setArg(3, alpha);
        kernel.setArg(3, beta);

        // Number of work items in each local work group
        cl::NDRange localSize(64);
        // Number of total work items - localSize must be devisor
        cl::NDRange globalSize((int)(ceil(n/(float)64)*64));

        // Enqueue kernel
        cl::Event event;
        queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            globalSize,
            localSize,
            NULL,
            &event);

        // Block until kernel completion
        event.wait();

        // Read back d_c
        queue.enqueueReadBuffer(d_b, CL_TRUE, 0, bytes, pDst);
        }
    catch (cl::Error err) {
         std::cerr
            << "ERROR: "<<err.what()<<"("<<err.err()<<")"<<std::endl;
    }

  return RPP_SUCCESS;

}