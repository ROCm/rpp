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
//Fish eye
cl_int
fisheye_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, 
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue)
{
    cl_int err;
    short counter;
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"fish_eye.cl","fish_eye.bin","fisheye_planar",theProgram,theKernel);
        clRetainKernel(theKernel);
        // cl_kernel_initializer(  theQueue, "fish_eye.cl", "fisheye_planar",
        //                         theProgram, theKernel); 
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"fish_eye.cl","fish_eye.bin","fisheye_packed",theProgram,theKernel);
        clRetainKernel(theKernel);
        // cl_kernel_initializer(  theQueue, "fish_eye.cl", "fisheye_packed",
        //                         theProgram, theKernel); 
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

cl_int
lens_correction_cl( cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr,
           float strength,float zoom,
           RppiChnFormat chnFormat, unsigned int channel,
           cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"lens_correction.cl","lens_correction.bin","lenscorrection_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
        // cl_kernel_initializer(theQueue,
        //                   "lens_correction.cl",
        //                   "lenscorrection_pln",
        //                   theProgram, theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"lens_correction.cl","lens_correction.bin","lenscorrection_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
        // cl_kernel_initializer(theQueue,
        //                   "lens_correction.cl",
        //                   "lenscorrection_pkd",
        //                   theProgram, theKernel);
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}
    if (strength == 0)
        strength = 0.000001;
    //---- Args Setter
    err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &strength);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &zoom);
    //----

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;
}

//--------------------------------Rotate

/*********** RandomCropLetterBox ***********/

RppStatus
random_crop_letterbox_cl(  cl_mem srcPtr, RppiSize srcSize, 
                            cl_mem dstPtr, RppiSize dstSize, 
                            Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                            RppiChnFormat chnFormat, unsigned int channel,
                            cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"randomcropletterbox.cl","randomcropletterbox.bin","randomcropletterbox_planar",theProgram,theKernel);
        clRetainKernel(theKernel);    
    }
    // cl_kernel_initializer(theQueue,
    //                       "randomcropletterbox.cl",
    //                       "trandomcropletterbox_planar",
    //                       theProgram, theKernel);
    else
    {
        CreateProgramFromBinary(theQueue,"randomcropletterbox.cl","randomcropletterbox.bin","randomcropletterbox_packed",theProgram,theKernel);
        clRetainKernel(theKernel);    
    }
    // cl_kernel_initializer(theQueue,
    //                       "randomcropletterbox.cl",
    //                       "randomcropletterbox_packed",
    //                       theProgram, theKernel);
    
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