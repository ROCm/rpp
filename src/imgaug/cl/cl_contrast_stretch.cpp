#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

RppStatus
contrast_stretch_cl (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32u newMin, Rpp32u newMax,
                            RppiChnFormat chnFormat, unsigned int channel,
                            cl_command_queue theQueue)
{
    Rpp32u min = 0; /* Kernel has to be called */
    Rpp32u max = 255; /* Kernel has to be called */
    cl_kernel theKernel;
    cl_program theProgram;
    cl_kernel_initializer(theQueue,
                          "contrast_stretch.cl",
                          "contrast_stretch",
                          theProgram, theKernel);


    //----- Args Setter
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 2, sizeof(int), &min);
    clSetKernelArg(theKernel, 3, sizeof(int), &max);
    clSetKernelArg(theKernel, 4, sizeof(unsigned int), &newMin);
    clSetKernelArg(theKernel, 5, sizeof(unsigned int), &newMax);
    clSetKernelArg(theKernel, 6, sizeof(unsigned int), &(srcSize.height));
    clSetKernelArg(theKernel, 7, sizeof(unsigned int), &(srcSize.width));
    clSetKernelArg(theKernel, 8, sizeof(unsigned int), &channel);
    //-----

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;

    cl_kernel_implementer(theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;

}
