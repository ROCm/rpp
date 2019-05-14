#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"


float gauss_3x3[] = {
0.0625, 0.125, 0.0625,
0.125 , 0.25 , 0.125,
0.0625, 0.125, 0.0625,
};


cl_int
cl_gaussian_blur(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, size_t filterSize,
                RppiChnFormat chnFormat, size_t channel,
                cl_command_queue theQueue)
{
    cl_int err;

    float* filterBuffer;
    if (filterSize == 3) filterBuffer= gauss_3x3;
    else  std::cerr << "Unimplemeted kernel Size";
    cl_context theContext;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    cl_mem filtPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
                                    sizeof(float)*filterSize*filterSize, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, filtPtr, CL_TRUE, 0,
                                   sizeof(float)*filterSize*filterSize,
                                   filterBuffer, 0, NULL, NULL);

    cl_kernel theKernel;
    cl_program theProgram;
    cl_kernel_initializer(  theQueue,
                            "gaussian_blur.cl",
                            "gaussian_blur_planar",
                            theProgram, theKernel);


    err  = clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, 2, sizeof(cl_mem), &filtPtr);
    err |= clSetKernelArg(theKernel, 3, sizeof(size_t), &srcSize.height);
    err |= clSetKernelArg(theKernel, 4, sizeof(size_t), &srcSize.width);
    err |= clSetKernelArg(theKernel, 5, sizeof(size_t), &channel);
    err |= clSetKernelArg(theKernel, 6, sizeof(size_t), &filterSize);

    size_t dim3[3];
    dim3[0] = srcSize.width;
    dim3[1] = srcSize.height;
    dim3[2] = channel;
    cl_kernel_implementer (theQueue, dim3, theProgram, theKernel);

    clReleaseMemObject(filtPtr);

}