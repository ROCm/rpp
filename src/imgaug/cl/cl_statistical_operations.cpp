#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

RppStatus
integral_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"integral.cl","integral.cl.bin","integral_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"integral.cl","integral.cl.bin","integral_pkd",theProgram,theKernel);
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
min_cl( cl_mem srcPtr1,cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;
    CreateProgramFromBinary(theQueue,"min.cl","min.cl.bin","min",theProgram,theKernel);
    //---- Args Setter
    clRetainKernel(theKernel);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr2);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
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

RppStatus
max_cl( cl_mem srcPtr1,cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;
    CreateProgramFromBinary(theQueue,"max.cl","max.cl.bin","max",theProgram,theKernel);
    //---- Args Setter
    clRetainKernel(theKernel);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr2);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
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

RppStatus
histogram_cl(cl_mem srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_int err;

    cl_context theContext;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);

    
    cl_kernel theKernel;
    cl_program theProgram;
    unsigned int numGroups;


    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"histogram.cl","histogram.cl.bin","partial_histogram_pln",
                                    theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"histogram.cl","histogram.cl.bin","partial_histogram_pkd",
                                    theProgram,theKernel);
        clRetainKernel(theKernel);

    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    size_t lDim3[3];
    size_t gDim3[3];
    int num_pixels_per_work_item = 16;

    gDim3[0] = srcSize.width / num_pixels_per_work_item + 1;
    gDim3[1] = srcSize.height / num_pixels_per_work_item + 1;
    lDim3[0] = num_pixels_per_work_item;
    lDim3[1] = num_pixels_per_work_item;
    gDim3[2] = 1;
    lDim3[2] = 1;
    

    numGroups = gDim3[0] * gDim3[1];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    
    cl_mem partialHistogram = clCreateBuffer(theContext, CL_MEM_READ_WRITE,
                                    sizeof(unsigned int)*256*channel*numGroups, NULL, NULL);
    cl_mem histogram = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
                                    sizeof(unsigned int)*256*channel, NULL, NULL);
    
    

    counter = 0;
    err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &partialHistogram);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);


    cl_kernel_implementer (theQueue, gDim3, lDim3, theProgram, theKernel);

    // // For sum histogram kernel
    CreateProgramFromBinary(theQueue,"histogram.cl","histogram.cl.bin","histogram_sum_partial",
                                                                        theProgram,theKernel);
    clRetainKernel(theKernel);

    counter = 0;
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &partialHistogram);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &histogram);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &numGroups);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);

    gDim3[0] = 256 * channel;
    lDim3[0] = 256;
    gDim3[1] = 1; 
    gDim3[2] = 1;
    lDim3[1] = 1;
    lDim3[2] = 1;

    cl_kernel_implementer (theQueue, gDim3, lDim3, theProgram, theKernel);
    
    clEnqueueReadBuffer(theQueue, histogram, CL_TRUE, 0, sizeof(unsigned int)*256*channel, outputHistogram, 0, NULL, NULL );
}

RppStatus
min_max_loc_cl(cl_mem srcPtr, RppiSize srcSize, Rpp8u* min, Rpp8u* max, Rpp8u** minLoc, Rpp8u** maxLoc, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    unsigned short counter=0;
    
    cl_kernel theKernel;
    cl_program theProgram;
    CreateProgramFromBinary(theQueue,"min_max_loc.cl","min_max_loc.cl.bin","min_loc_cl",theProgram,theKernel);
    clRetainKernel(theKernel);   
    
    int i;
    
    const int LIST_SIZE = srcSize.height * srcSize.width * channel;
    int numGroups;
    for(i = LIST_SIZE / 256 ; i > 0 ; i--)
    {
        if(LIST_SIZE % i ==0)
        {
            numGroups = i;
            break;
        }
    }
    float sum = 0;
    unsigned char *partial_min_max;
    partial_min_max = (unsigned char *) calloc (numGroups, sizeof(unsigned char));
 
    cl_context theContext;
    clGetCommandQueueInfo(theQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(theQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);

    cl_mem b_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned char), NULL, NULL);
    cl_mem c_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned char), NULL, NULL);

    clEnqueueWriteBuffer(theQueue, b_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min_max, 0, NULL, NULL);
    clEnqueueWriteBuffer(theQueue, c_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min_max, 0, NULL, NULL);

    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &b_mem_obj);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), NULL);

    size_t gDim3[3];
    gDim3[0] = LIST_SIZE;
    gDim3[1] = 1;
    gDim3[2] = 1;
    size_t local_item_size[3];
    local_item_size[0] = LIST_SIZE / numGroups;
    local_item_size[1] = 1;
    local_item_size[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, local_item_size, theProgram, theKernel);
    clEnqueueReadBuffer(theQueue, b_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min_max, 0, NULL, NULL);   
    
    sum = partial_min_max[0];
    for(i = 0; i < numGroups; i++)
    {
        if(sum > partial_min_max[1])
            sum = partial_min_max[1];
    }
    *min = sum; 

    CreateProgramFromBinary(theQueue,"min_max_loc.cl","min_max_loc.cl.bin","max_loc_cl",theProgram,theKernel);
    clRetainKernel(theKernel); 
        
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &c_mem_obj);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), NULL);

    cl_kernel_implementer (theQueue, gDim3, local_item_size, theProgram, theKernel);
    clEnqueueReadBuffer(theQueue, c_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min_max, 0, NULL, NULL);   
    sum = partial_min_max[0];
    for(i = 0; i < numGroups; i++)
    {
        if(sum < partial_min_max[1])
            sum = partial_min_max[1];
    }
    *max = sum; 
    
    clReleaseMemObject(b_mem_obj); 
    free(partial_min_max);
  
}