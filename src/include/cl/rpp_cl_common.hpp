#ifndef RPP_CL_COMMON_H
#define RPP_CL_COMMON_H

#include <stdio.h>
#include <CL/cl.hpp>
#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define __forceinline // __attribute__((always_inline))

__forceinline cl_int
cl_kernel_initializer (RppHandle_t rppHandle, const std::string kernelFile, cl_program& theProgram, cl_kernel& theKernel)
{

// File Handling
    char *sourceStr;
    size_t sourceSize;
    std::string kernelFile_cl = "cl/" + kernelFile + ".cl";
    FILE *filePtr = fopen( kernelFile_cl.c_str(), "r");
    if (!filePtr) {
        fprintf(stderr, "Failed to load kernel.\n");
        return RPP_ERROR;
    }
    //TODO: Add dynamic calculation of string length
    sourceStr = (char*)malloc(2048);
    sourceSize = fread( sourceStr, 1, 2048, filePtr);
    fclose( filePtr );

    cl_context clContext;
    clGetCommandQueueInfo(  static_cast<cl_command_queue>(rppHandle),
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &clContext, NULL);

    cl_device_id clDevice;
    clGetCommandQueueInfo(  static_cast<cl_command_queue>(rppHandle),
                            CL_QUEUE_DEVICE, sizeof(cl_device_id), &clDevice, NULL);
// kernel creation
    cl_int ret = 0;//TODO
    theProgram = clCreateProgramWithSource(clContext, 1,
                         (const char **)&sourceStr, (const size_t *)&sourceSize, &ret);
    ret = clBuildProgram(theProgram, 1, &clDevice, NULL, NULL, NULL);
    theKernel = clCreateKernel(theProgram, kernelFile.c_str(), &ret);

    return ret;
}


__forceinline cl_int
cl_kernel_implementer (RppHandle_t rppHandle, RppiSize srcSize,cl_program& theProgram, cl_kernel& theKernel  )
{
    cl_int ret = 0;//TODO
    cl_event event;
    size_t global_item_size = srcSize.height * srcSize.width;
    size_t local_item_size = 64;
    ret = clEnqueueNDRangeKernel( static_cast<cl_command_queue>(rppHandle),
                            theKernel,
                            1 /*work_dim*/, NULL /*global_work_offset*/,
                            &global_item_size,
                            &local_item_size,
                            0 /*num_events_in_wait_list*/, NULL /*event_wait_list*/,
                            &event );
    //TODO: event.wait
    clReleaseProgram(theProgram);
    clReleaseKernel(theKernel);
    return ret;
}

#endif RPP_CL_COMMON_H