#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

cl_int
cl_flip(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiAxis flipAxis,
                RppiChnFormat chnFormat, size_t channel,
                cl_command_queue theQueue)

{
        cl_int err;

    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (flipAxis == RPPI_VERTICAL_AXIS)
        {   cl_kernel_initializer(  theQueue, "flip.cl", "flip_vertical_planar",
                                theProgram, theKernel);
        }
        else if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {   cl_kernel_initializer(  theQueue, "flip.cl", "flip_horizontal_planar",
                                theProgram, theKernel);
        }
        else if (flipAxis == RPPI_BOTH_AXIS)
        {   cl_kernel_initializer(  theQueue, "flip.cl", "flip_bothaxis_planar",
                                theProgram, theKernel);
        }
    }
    else
    {std::cerr << "Unimplemented Functionality";}

    err  = clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, 2, sizeof(size_t), &srcSize.height);
    err |= clSetKernelArg(theKernel, 3, sizeof(size_t), &srcSize.width);
    err |= clSetKernelArg(theKernel, 4, sizeof(size_t), &channel);

    size_t dim3[3];
    dim3[0] = srcSize.width;
    dim3[1] = srcSize.height;
    dim3[2] = channel;
    cl_kernel_implementer (theQueue, dim3, theProgram, theKernel);
}
