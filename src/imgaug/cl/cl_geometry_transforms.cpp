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
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;
    cl_context theContext;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        cl_kernel_initializer(  theQueue, "resize.cl", "resize_pln",
                                theProgram, theKernel); 
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        cl_kernel_initializer(  theQueue, "resize.cl", "resize_pkd",
                                theProgram, theKernel); 
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    err  = clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, 2, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, 4, sizeof(unsigned int), &dstSize.height);
    err |= clSetKernelArg(theKernel, 5, sizeof(unsigned int), &dstSize.width);
    err |= clSetKernelArg(theKernel, 6, sizeof(unsigned int), &channel);


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
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;
    cl_context theContext;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        cl_kernel_initializer(  theQueue, "resize.cl", "resize_crop_pln",
                                theProgram, theKernel); 
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        cl_kernel_initializer(  theQueue, "resize.cl", "resize_crop_pkd",
                                theProgram, theKernel); 
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    err  = clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, 2, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, 4, sizeof(unsigned int), &dstSize.height);
    err |= clSetKernelArg(theKernel, 5, sizeof(unsigned int), &dstSize.width);
    err |= clSetKernelArg(theKernel, 6, sizeof(unsigned int), &x1);
    err |= clSetKernelArg(theKernel, 7, sizeof(unsigned int), &y1);
    err |= clSetKernelArg(theKernel, 8, sizeof(unsigned int), &x2);
    err |= clSetKernelArg(theKernel, 9, sizeof(unsigned int), &y2);
    err |= clSetKernelArg(theKernel, 10, sizeof(unsigned int), &channel);


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
    cl_int err;

    cl_kernel theKernel;
    cl_program theProgram;
    

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        cl_kernel_initializer(  theQueue, "rotate.cl", "rotate_pln",
                                theProgram, theKernel); 
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        cl_kernel_initializer(  theQueue, "rotate.cl", "rotate_pkd",
                                theProgram, theKernel); 
    }

    else
    {std::cerr << "Internal error: Unknown Channel format";}

    err  = clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, 2, sizeof(float), &angleDeg);
    err |= clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, 4, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, 5, sizeof(unsigned int), &dstSize.height);
    err |= clSetKernelArg(theKernel, 6, sizeof(unsigned int), &dstSize.width);
    err |= clSetKernelArg(theKernel, 7, sizeof(unsigned int), &channel);

    size_t gDim3[3];
    gDim3[0] = dstSize.width;
    gDim3[1] = dstSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

}
//--------------------------------Rotate

/*********** RandomCropLetterBox ***********/

RppStatus
Random_Crop_Letter_Box_cl(  cl_mem srcPtr, RppiSize srcSize, 
                            cl_mem dstPtr, RppiSize dstSize, 
                            Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                            RppiChnFormat chnFormat, unsigned int channel,
                            cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    cl_kernel_initializer(theQueue,
                          "randomcropletterbox.cl",
                          "trandomcropletterbox_planar",
                          theProgram, theKernel);
    else
    cl_kernel_initializer(theQueue,
                          "randomcropletterbox.cl",
                          "randomcropletterbox_packed",
                          theProgram, theKernel);
    
    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &dstSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &dstSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x1);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y1);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x2);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y2);
    
    size_t gDim3[3];
    gDim3[0] = dstSize.width;
    gDim3[1] = dstSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;
}

//Warp -Affine
cl_int
warp_affine_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiSize dstSize, float *affine, 
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue)
{
    cl_int err;
    float affine_inv[6];
    float det; //for Deteminent
    det = (affine[0] * affine [4])  - (affine[1] * affine[3]);
    affine_inv[0] = affine[4]/ det;
    affine_inv[1] = (- 1 * affine[1])/ det;
    affine_inv[2] = -1 * affine[2];
    affine_inv[3] = (-1 * affine[3]) /det ;
    affine_inv[4] = affine[0]/det;
    affine_inv[5] = -1 * affine[5];

    cl_kernel theKernel;
    cl_program theProgram;
    float *affine_matrix;
    cl_context theContext;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    cl_mem affine_array = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
                                    sizeof(float)*6, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, affine_array, CL_TRUE, 0,
                                   sizeof(float)*6,
                                   affine_inv, 0, NULL, NULL);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        cl_kernel_initializer(  theQueue, "warp_affine.cl", "warp_affine_pln",
                                theProgram, theKernel); 
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        cl_kernel_initializer(  theQueue, "warp_affine.cl", "warp_affine_pkd",
                                theProgram, theKernel); 
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}
    int ctr =0;
    err  = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &affine_matrix);
    err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &dstSize.height);
    err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &dstSize.width);
    err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);


    size_t gDim3[3];
    gDim3[0] = dstSize.width;
    gDim3[1] = dstSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
}