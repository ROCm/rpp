#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

RppStatus
cl_contrast_streach (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp8u newMin, Rpp8u newMax,
                            RppiChnFormat chnFormat, size_t channel,
                            cl_command_queue theQueue)
{
    Rpp8u min = 0; /* Kernel has to be called */
    Rpp8u max = 255; /* Kernel has to be called */
    cl_kernel theKernel;
    cl_program theProgram;
    cl_kernel_initializer(theQueue,
                          "contrast_streach.cl",
                          "contrast_streach",
                          theProgram, theKernel);

    
    //----- Args Setter
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(kernel, 2, sizeof(int), &min);
    clSetKernelArg(kernel, 3, sizeof(int), &max);
    clSetKernelArg(kernel, 4, sizeof(unsigned int), &new_min);
    clSetKernelArg(kernel, 5, sizeof(unsigned int), &new_max);
    clSetKernelArg(kernel, 6, sizeof(unsigned int), &height);
    clSetKernelArg(kernel, 7, sizeof(unsigned int), &width);
    //-----

    size_t dim3[3];
    dim3[0] = srcSize.height;
    dim3[1] = srcSize.width;
    dim3[2] = channel;
    cl_kernel_implementer (theQueue, dim3, theProgram, theKernel);

    return RPP_SUCCESS;

}
