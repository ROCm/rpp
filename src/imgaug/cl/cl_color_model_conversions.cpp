#include <cl/rpp_cl_common.hpp>
#include <cpu/rpp_cpu_common.hpp>
#include "cl_declarations.hpp"

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
    int counter = 0;
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue, "temperature.cl", "temperature.cl.bin", "temperature_planar", theProgram, theKernel);
        clRetainKernel(theKernel);    
    }
    else
    {
        CreateProgramFromBinary(theQueue, "temperature.cl", "temperature.cl.bin", "temperature_packed", theProgram, theKernel);
        clRetainKernel(theKernel);    
    }
    //---- Args Setter
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, counter++, sizeof(float), &adjustmentValue);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = 1;
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
    if(chnFormat == RPPI_CHN_PLANAR)
    {    
        CreateProgramFromBinary(theQueue,"vignette.cl","vignette.cl.bin","vignette_pln",theProgram,theKernel);
        clRetainKernel(theKernel); 
    } 
    else
    {
        CreateProgramFromBinary(theQueue,"vignette.cl","vignette.cl.bin","vignette_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    
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

