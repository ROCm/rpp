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
    cl_kernel_initializer(theQueue, "brightness_contrast", theProgram, theKernel); // .cl will get be added internally

    //---- Args Setter
    size_t n = srcSize.height * srcSize.width * channel ;
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 2, sizeof(size_t), &n);
    clSetKernelArg(theKernel, 3, sizeof(float), &alpha);
    clSetKernelArg(theKernel, 4, sizeof(int), &beta);
    //----

    cl_kernel_implementer (theQueue, srcSize.height*srcSize.height*channel, theProgram, theKernel);

    return RPP_SUCCESS;

}