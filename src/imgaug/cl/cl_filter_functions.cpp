#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

RppStatus
bilateral_filter_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, unsigned int filterSize,
                double sigmaI, double sigmaS,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue)
{
    cl_int err;
    cl_context theContext;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);

    cl_kernel theKernel;
    cl_program theProgram;


    if (chnFormat == RPPI_CHN_PLANAR)
    {
        cl_kernel_initializer(  theQueue, "bilateral_filter.cl",
                                "bilateral_filter_planar", theProgram, theKernel);

    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        cl_kernel_initializer(  theQueue, "bilateral_filter.cl",
                                "bilateral_filter_packed", theProgram, theKernel);
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}




    err  = clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, 2, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, 4, sizeof(unsigned int), &channel);
    err |= clSetKernelArg(theKernel, 5, sizeof(unsigned int), &filterSize);
    err |= clSetKernelArg(theKernel, 6, sizeof(double), &sigmaI);
    err |= clSetKernelArg(theKernel, 7, sizeof(double), &sigmaS);

//----
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
}