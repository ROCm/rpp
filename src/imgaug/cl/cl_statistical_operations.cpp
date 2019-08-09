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
        if(channel > 1)
            CreateProgramFromBinary(theQueue,"histogram.cl","histogram.cl.bin","partial_histogram_pln",
                                    theProgram,theKernel);
        else
            CreateProgramFromBinary(theQueue,"histogram.cl","histogram.cl.bin","partial_histogram_pln1",
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

    // // For minElement histogram kernel
    CreateProgramFromBinary(theQueue,"histogram.cl","histogram.cl.bin","histogram_minElement_partial",
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
min_max_loc_cl(cl_mem srcPtr, RppiSize srcSize, Rpp8u* min, Rpp8u* max, Rpp32u* minLoc, Rpp32u* maxLoc, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    unsigned short counter=0;
    
    cl_kernel theKernel;
    cl_program theProgram;
    CreateProgramFromBinary(theQueue,"min_max_loc.cl","min_max_loc.cl.bin","min",theProgram,theKernel);
    clRetainKernel(theKernel);   
    
    int i;

    const int LIST_SIZE = srcSize.height * srcSize.width * channel;
    int numGroups = std::ceil(LIST_SIZE / 256);
    
    cl_context theContext;
    clGetCommandQueueInfo(theQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(theQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);

    unsigned char minElement = 255;
    unsigned int minLocation;
    unsigned char *partial_min;
    partial_min = (unsigned char *) calloc (numGroups, sizeof(unsigned char));
    unsigned int *partial_min_location;
    partial_min_location = (unsigned int *) calloc (numGroups, sizeof(unsigned int));
    cl_mem b_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned char), NULL, NULL);
    clEnqueueWriteBuffer(theQueue, b_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min, 0, NULL, NULL);
    cl_mem b_mem_obj1 = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned int), NULL, NULL);
    clEnqueueWriteBuffer(theQueue, b_mem_obj1, CL_TRUE, 0, numGroups * sizeof(unsigned int), partial_min_location, 0, NULL, NULL);

    unsigned char maxElement = 0;
    unsigned int maxLocation;
    unsigned char *partial_max;
    partial_max = (unsigned char *) calloc (numGroups, sizeof(unsigned char));
    unsigned int *partial_max_location;
    partial_max_location = (unsigned int *) calloc (numGroups, sizeof(unsigned int));
    cl_mem c_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned char), NULL, NULL);
    clEnqueueWriteBuffer(theQueue, c_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_max, 0, NULL, NULL);
    cl_mem c_mem_obj1 = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned int), NULL, NULL);
    clEnqueueWriteBuffer(theQueue, c_mem_obj1, CL_TRUE, 0, numGroups * sizeof(unsigned int), partial_max_location, 0, NULL, NULL);


    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &b_mem_obj);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &b_mem_obj1);

    size_t gDim3[3];
    gDim3[0] = LIST_SIZE - (LIST_SIZE % 256);
    gDim3[1] = 1;
    gDim3[2] = 1;
    size_t local_item_size[3];
    local_item_size[0] = 256;
    local_item_size[1] = 1;
    local_item_size[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, local_item_size, theProgram, theKernel);
    clEnqueueReadBuffer(theQueue, b_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min, 0, NULL, NULL);
    clEnqueueReadBuffer(theQueue, b_mem_obj1, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min_location, 0, NULL, NULL);   
    
    for(i = 0; i < numGroups; i++)
    {
        if(minElement > partial_min[i])
        {
            minElement = partial_min[i];
            minLocation = partial_min_location[i];

        }
    }
    *min = minElement;
    *minLoc=minLocation;


    CreateProgramFromBinary(theQueue,"min_max_loc.cl","min_max_loc.cl.bin","max",theProgram,theKernel);
    clRetainKernel(theKernel); 

    counter = 0;
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &c_mem_obj);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &c_mem_obj1);

    cl_kernel_implementer (theQueue, gDim3, local_item_size, theProgram, theKernel);
    clEnqueueReadBuffer(theQueue, c_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_max, 0, NULL, NULL); 
    clEnqueueReadBuffer(theQueue, b_mem_obj1, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_max_location, 0, NULL, NULL); 
    for(i = 0; i < numGroups; i++)
    {
        if(maxElement < partial_max[i])
        {
            maxElement = partial_max[i];
            maxLocation = partial_max_location[i];
        }
    }

    *max = maxElement;
    *maxLoc=maxLocation;

    clReleaseMemObject(b_mem_obj); 
    free(partial_min);
    clReleaseMemObject(c_mem_obj); 
    free(partial_max);
    clReleaseMemObject(b_mem_obj1); 
    free(partial_min_location);
    clReleaseMemObject(c_mem_obj1); 
    free(partial_max_location);
}