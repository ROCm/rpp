#include <stdio.h>

#include <CL/cl.hpp>
#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS


inline cl_int
cl_kernel_initializer (const std::string kernelFile, cl_kernel& theKernel )
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
    sourceStr = (char*)malloc(MAX_SOURCE_SIZE);
    sourceSize = fread( sourceStr, 1, MAX_SOURCE_SIZE, fp);
    fclose( fileptr );


// kernel creation
    cl_int ret = 0;//TODO
    cl_program program = clCreateProgramWithSource(context/*TODO*/, 1,
           (const char **)&sourceStr, (const size_t *)&sourceSize, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    theKernel = clCreateKernel(program, kernelFile.c_str(), &ret);

    return ret;
}