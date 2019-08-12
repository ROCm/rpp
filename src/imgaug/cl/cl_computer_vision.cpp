#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

/********************** local binary pattern ************************/
RppStatus
local_binary_pattern_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"local_binary_pattern.cl","local_binary_pattern.cl.bin","local_binary_pattern_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"local_binary_pattern.cl","local_binary_pattern.cl.bin","local_binary_pattern_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    
    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}

RppStatus
data_object_copy_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    clEnqueueCopyBuffer(theQueue, srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
    
    return RPP_SUCCESS;    
}

RppStatus
gaussian_image_pyramid_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);
    
    cl_context theContext;
    clGetCommandQueueInfo(theQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(theQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_mem kernel = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, kernelSize * kernelSize * sizeof(Rpp32f), NULL, NULL);
    clEnqueueWriteBuffer(theQueue, kernel, CL_TRUE, 0, kernelSize * kernelSize * sizeof(Rpp32f), kernelMain, 0, NULL, NULL);




    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"gaussian_image_pyramid.cl","gaussian_image_pyramid.cl.bin","gaussian_image_pyramid_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"gaussian_image_pyramid.cl","gaussian_image_pyramid.cl.bin","gaussian_image_pyramid_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }

    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &kernel);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;  
}

RppStatus
laplacian_image_pyramid_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);    
    cl_context theContext;
    clGetCommandQueueInfo(theQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(theQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_mem kernel = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, kernelSize * kernelSize * sizeof(Rpp32f), NULL, NULL);
    clEnqueueWriteBuffer(theQueue, kernel, CL_TRUE, 0, kernelSize * kernelSize * sizeof(Rpp32f), kernelMain, 0, NULL, NULL);
    cl_mem srcPtr1 = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, srcSize.height * srcSize.width * channel * sizeof(Rpp8u), NULL, NULL);

    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;

    if(chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"gaussian_image_pyramid.cl","gaussian_image_pyramid.cl.bin","gaussian_image_pyramid_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"gaussian_image_pyramid.cl","gaussian_image_pyramid.cl.bin","gaussian_image_pyramid_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }

    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr1);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &kernel);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    if(chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"laplacian_image_pyramid.cl","laplacian_image_pyramid.cl.bin","laplacian_image_pyramid_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"laplacian_image_pyramid.cl","laplacian_image_pyramid.cl.bin","laplacian_image_pyramid_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    ctr=0;
    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr1);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &kernel);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
    
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);


    return RPP_SUCCESS;  
}