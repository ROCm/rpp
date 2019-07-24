#include <cl/rpp_cl_common.hpp>
#include <cpu/rpp_cpu_common.hpp>
#include "cl_declarations.hpp"

RppStatus
rgb_to_hsv_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue){
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {   
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","rgb2hsv_pln",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(theQueue,
        //                 "hsv_kernels.cl",
        //                 "rgb2hsv_pln",
        //                 theProgram, theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","rgb2hsv_pkd",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(theQueue,
        //                 "hsv_kernels.cl",
        //                 "rgb2hsv_pkd",
        //                 theProgram, theKernel);
    }

    //---- Args Setter
    unsigned int n = srcSize.height * srcSize.width * channel ;
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);


    size_t gDim3[3];
    gDim3[0]= srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL, theProgram, theKernel);


    return RPP_SUCCESS;

}

RppStatus
hsv_to_rgb_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue){
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","hsv2rgb_pln",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(theQueue,
        //                       "hsv_kernels.cl",
        //                       "hsv2rgb_pln",
        //                       theProgram, theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","hsv2rgb_pkd",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(theQueue,
        //                       "hsv_kernels.cl",
        //                       "hsv2rgb_pkd",
        //                       theProgram, theKernel);
    }

    //---- Args Setter
    unsigned int n = srcSize.height * srcSize.width * channel ;
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);


    size_t gDim3[3];
    gDim3[0]= srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL, theProgram, theKernel);


    return RPP_SUCCESS;

}

//////////////////////HUE CODE////////////////////////////////
RppStatus
hueRGB_cl (cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f hue, Rpp32f saturation,
                RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue){
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","huergb_pln",theProgram,theKernel);
        clRetainKernel(theKernel); 
        //    cl_kernel_initializer(theQueue,
        //                       "hsv_kernels.cl",
        //                       "huergb_pln",
        //                       theProgram, theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","huergb_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);     
        // cl_kernel_initializer(theQueue,
        //                   "hsv_kernels.cl",
        //                   "huergb_pkd",
        //                   theProgram, theKernel);
    }

    //---- Args Setter
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(double), &hue);
    clSetKernelArg(theKernel, counter++, sizeof(double), &saturation);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);

    size_t gDim3[3];
    gDim3[0]= srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL, theProgram, theKernel);
}

RppStatus
hueHSV_cl (cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f hue, Rpp32f saturation,
                RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue){
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","hsvhsv_pln",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(theQueue,
        //                       "hsv_kernels.cl",
        //                       "hsvhsv_pln",
        //                       theProgram, theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","huehsv_pkd",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(theQueue,
        //                       "hsv_kernels.cl",
        //                       "huehsv_pkd",
        //                       theProgram, theKernel);
    }

    //---- Args Setter
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(double), &hue);
    clSetKernelArg(theKernel, counter++, sizeof(double), &saturation);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);


    size_t gDim3[3];
    gDim3[0]= srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL, theProgram, theKernel);

}

/****************  Gamma correction *******************/

RppStatus
gamma_correction_cl ( cl_mem srcPtr1,RppiSize srcSize,
                 cl_mem dstPtr, float gamma,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;

    CreateProgramFromBinary(theQueue,"gamma_correction.cl","gamma_correction.cl.bin","gamma_correction",theProgram,theKernel);
    clRetainKernel(theKernel); 

    // cl_kernel_initializer(theQueue,
    //                       "gamma_correction.cl",
    //                       "gamma_correction",
    //                       theProgram, theKernel);

    //---- Args Setter
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(float), &gamma);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    //----

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;

}

/****************  Temprature modification *******************/

RppStatus
color_temperature_cl( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, float adjustmentValue, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"temprature.cl","temprature.cl.bin","temprature_planar",theProgram,theKernel);
        clRetainKernel(theKernel);    
    }
    else
    {
        CreateProgramFromBinary(theQueue,"temprature.cl","temprature.cl.bin","temprature_packed",theProgram,theKernel);
        clRetainKernel(theKernel);    
    }
    //---- Args Setter
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 2, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, 4, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, 5, sizeof(float), &adjustmentValue);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;
}

/****************  vignette effect *******************/

// value should always be greater than 0
//0-> full vignette effect
//100-> no vignette effect

RppStatus
vignette_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, float stdDev, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    CreateProgramFromBinary(theQueue,"vignette.cl","vignette.cl.bin","vignette",theProgram,theKernel);
    clRetainKernel(theKernel);  

    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(float), &stdDev);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;
}