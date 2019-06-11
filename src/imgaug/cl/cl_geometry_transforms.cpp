#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"


cl_int
flip_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiAxis flipAxis,
                RppiChnFormat chnFormat, unsigned int channel,
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
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        if (flipAxis == RPPI_VERTICAL_AXIS)
        {   cl_kernel_initializer(  theQueue, "flip.cl", "flip_vertical_packed",
                                theProgram, theKernel);
        }
        else if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {   cl_kernel_initializer(  theQueue, "flip.cl", "flip_horizontal_packed",
                                theProgram, theKernel);
        }
        else if (flipAxis == RPPI_BOTH_AXIS)
        {   cl_kernel_initializer(  theQueue, "flip.cl", "flip_bothaxis_packed",
                                theProgram, theKernel);
        }
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    err  = clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, 2, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, 4, sizeof(unsigned int), &channel);

//-----

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
}
