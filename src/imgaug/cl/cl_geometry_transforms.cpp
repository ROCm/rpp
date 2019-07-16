#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"


cl_int
flip_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiAxis flipAxis,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue)

{
    unsigned short counter=0;
    cl_int err;

    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (flipAxis == RPPI_VERTICAL_AXIS)
        {   
            CreateProgramFromBinary(theQueue,"flip.cl","flip.cl.bin","flip_vertical_planar",theProgram,theKernel);
            clRetainKernel(theKernel);
            // cl_kernel_initializer(  theQueue, "flip.cl", "flip_vertical_planar",
            //                     theProgram, theKernel);
        }
        else if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {   
            CreateProgramFromBinary(theQueue,"flip.cl","flip.cl.bin","flip_horizontal_planar",theProgram,theKernel);
            clRetainKernel(theKernel);   
            // cl_kernel_initializer(  theQueue, "flip.cl", "flip_horizontal_planar",
            //                     theProgram, theKernel);
        }
        else if (flipAxis == RPPI_BOTH_AXIS)
        {   
            CreateProgramFromBinary(theQueue,"flip.cl","flip.cl.bin","flip_bothaxis_planar",theProgram,theKernel);
            clRetainKernel(theKernel);   
            // cl_kernel_initializer(  theQueue, "flip.cl", "flip_bothaxis_planar",
            //                     theProgram, theKernel);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        if (flipAxis == RPPI_VERTICAL_AXIS)
        {   
            CreateProgramFromBinary(theQueue,"flip.cl","flip.cl.bin","flip_vertical_packed",theProgram,theKernel);
            clRetainKernel(theKernel);   
            // cl_kernel_initializer(  theQueue, "flip.cl", "flip_vertical_packed",
            //                     theProgram, theKernel);
        }
        else if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {   
            CreateProgramFromBinary(theQueue,"flip.cl","flip.cl.bin","flip_horizontal_packed",theProgram,theKernel);
            clRetainKernel(theKernel);   
            // cl_kernel_initializer(  theQueue, "flip.cl", "flip_horizontal_packed",
            //                     theProgram, theKernel);
        }
        else if (flipAxis == RPPI_BOTH_AXIS)
        {   
            CreateProgramFromBinary(theQueue,"flip.cl","flip.cl.bin","flip_bothaxis_packed",theProgram,theKernel);
            clRetainKernel(theKernel);   
            // cl_kernel_initializer(  theQueue, "flip.cl", "flip_bothaxis_packed",
            //                     theProgram, theKernel);
        }
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);


    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
}

//Resize-------------
cl_int
resize_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiSize dstSize, 
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;
    cl_context theContext;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {   
        CreateProgramFromBinary(theQueue,"resize.cl","resize.cl.bin","resize_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
        // cl_kernel_initializer(  theQueue, "resize.cl", "resize_pln",
        //                         theProgram, theKernel); 
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {   
        CreateProgramFromBinary(theQueue,"resize.cl","resize.cl.bin","resize_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
        // cl_kernel_initializer(  theQueue, "resize.cl", "resize_pkd",
        //                         theProgram, theKernel); 
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dstSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dstSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);


    size_t gDim3[3];
    gDim3[0] = dstSize.width;
    gDim3[1] = dstSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
}

//------------Resize

//Resize Crop----------
cl_int
resize_crop_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiSize dstSize,
                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,  
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;
    cl_context theContext;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {   
        CreateProgramFromBinary(theQueue,"resize.cl","resize.cl.bin","resize_crop_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
        // cl_kernel_initializer(  theQueue, "resize.cl", "resize_crop_pln",
        //                         theProgram, theKernel); 
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {   
        CreateProgramFromBinary(theQueue,"resize.cl","resize.cl.bin","resize_crop_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
        // cl_kernel_initializer(  theQueue, "resize.cl", "resize_crop_pkd",
        //                         theProgram, theKernel); 
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dstSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dstSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &x1);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &y1);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &x2);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &y2);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);


    size_t gDim3[3];
    gDim3[0] = dstSize.width;
    gDim3[1] = dstSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
}

//-----------Resize Crop

//Rotate---------------------------
cl_int
rotate_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiSize dstSize, float angleDeg, 
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_int err;

    cl_kernel theKernel;
    cl_program theProgram;
    

    if (chnFormat == RPPI_CHN_PLANAR)
    {   
        CreateProgramFromBinary(theQueue,"rotate.cl","rotate.cl.bin","rotate_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
        // cl_kernel_initializer(  theQueue, "rotate.cl", "rotate_pln",
        //                         theProgram, theKernel); 
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {   
        CreateProgramFromBinary(theQueue,"rotate.cl","rotate.cl.bin","rotate_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
        // cl_kernel_initializer(  theQueue, "rotate.cl", "rotate_pkd",
        //                         theProgram, theKernel); 
    }

    else
    {std::cerr << "Internal error: Unknown Channel format";}

    err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(float), &angleDeg);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dstSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dstSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);

    size_t gDim3[3];
    gDim3[0] = dstSize.width;
    gDim3[1] = dstSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

}
//--------------------------------Rotate