#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

RppStatus
bilateral_filter_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, unsigned int filterSize,
                double sigmaI, double sigmaS,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_int err;
    cl_context theContext;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);

    cl_kernel theKernel;
    cl_program theProgram;


    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"bilateral_filter.cl","bilateral_filter.cl.bin","bilateral_filter_planar",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(  theQueue, "bilateral_filter.cl",
        //                         "bilateral_filter_planar", theProgram, theKernel);

    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"bilateral_filter.cl","bilateral_filter.cl.bin","bilateral_filter_packed",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(  theQueue, "bilateral_filter.cl",
        //                         "bilateral_filter_packed", theProgram, theKernel);
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}




    err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &filterSize);
    err |= clSetKernelArg(theKernel, counter++, sizeof(double), &sigmaI);
    err |= clSetKernelArg(theKernel, counter++, sizeof(double), &sigmaS);

//----
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
}

/********************** median filter ************************/
RppStatus
median_filter_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"median_filter.cl","median_filter.cl.bin","median_filter_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"median_filter.cl","median_filter.cl.bin","median_filter_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    
    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
        
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}

RppStatus
non_max_suppression_cl( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"non_max_suppression.cl","non_max_suppression.cl.bin","non_max_suppression_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"non_max_suppression.cl","non_max_suppression.cl.bin","non_max_suppression_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }

     //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);

     size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;  

}

RppStatus
custom_convolution_cl( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32f* kernel, RppiSize kernelSize, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    cl_context theContext;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_mem clkernel = clCreateBuffer(theContext, CL_MEM_READ_WRITE,
                                    sizeof(Rpp32f)*kernelSize.height*kernelSize.width, NULL, NULL);
    clEnqueueWriteBuffer(theQueue, clkernel, CL_TRUE, 0, sizeof(Rpp32f)*kernelSize.height*kernelSize.width, kernel, 0, NULL, NULL);
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"custom_convolution.cl","custom_convolution.cl.bin","custom_convolution_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"custom_convolution.cl","custom_convolution.cl.bin","custom_convolution_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }

    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &clkernel);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize.width);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;  
}

RppStatus
box_filter_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    float box_3x3[] = {
    0.111, 0.111, 0.111,
    0.111, 0.111, 0.111,
    0.111, 0.111, 0.111,
    };

    int ctr=0;
    cl_context theContext;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    cl_mem filtPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
                                    sizeof(float)*3*3, NULL, NULL);
    clEnqueueWriteBuffer(theQueue, filtPtr, CL_TRUE, 0,
                                   sizeof(float)*3*3,
                                   box_3x3, 0, NULL, NULL);
    cl_kernel theKernel;
    cl_program theProgram;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"convolution.cl","convolution.cl.bin","naive_convolution_planar",theProgram,theKernel);
        clRetainKernel(theKernel); 

    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"convolution.cl","convolution.cl.bin","naive_convolution_packed",theProgram,theKernel);
        clRetainKernel(theKernel); 
    }
    kernelSize=3;
     //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &filtPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;  

}

RppStatus
gaussian_filter_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
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
        CreateProgramFromBinary(theQueue,"gaussian.cl","gaussian.cl.bin","gaussian_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"gaussian.cl","gaussian.cl.bin","gaussian_pln",theProgram,theKernel);
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