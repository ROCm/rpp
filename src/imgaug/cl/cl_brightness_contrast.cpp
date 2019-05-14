#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

RppStatus
cl_brightness_contrast (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32f alpha, Rpp32s beta,
                            RppiChnFormat chnFormat, size_t channel,
                            cl_command_queue theQueue)
{
    cl_kernel theKernel;
    cl_program theProgram;
    cl_kernel_initializer(theQueue,
                          "brightness_contrast.cl",
                          "brightness_contrast",
                          theProgram, theKernel);

    //---- Args Setter
    size_t n = srcSize.height * srcSize.width * channel ;
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 2, sizeof(float), &alpha);
    clSetKernelArg(theKernel, 3, sizeof(int), &beta);
    clSetKernelArg(theKernel, 4, sizeof(size_t), &srcSize.height);
    clSetKernelArg(theKernel, 5, sizeof(size_t), &srcSize.width);
    clSetKernelArg(theKernel, 6, sizeof(size_t), &channel);
    //----

    size_t dim3[3];
    dim3[0] = srcSize.width;
    dim3[1] = srcSize.height;
    dim3[2] = channel;
    cl_kernel_implementer (theQueue, dim3, theProgram, theKernel);

    return RPP_SUCCESS;

}
