#include <cl/rpp_cl_common.hpp>

cl_int
cl_kernel_initializer (cl_command_queue theHandle, std::string kernelFile,
                        cl_program& theProgram, cl_kernel& theKernel)
{
    cl_int err;
    // File Handling
    char *sourceStr;
    size_t sourceSize;
    std::string kernelFile_cl = kernelFile + ".cl" ;
    std::cout << kernelFile_cl;
    FILE *filePtr = fopen( kernelFile_cl.c_str(), "rb");
    if (!filePtr) {
        fprintf(stderr, "Failed to load kernel.\n");
        return 1;
    }
    fseek(filePtr, 0, SEEK_END);
    size_t fileSize = ftell(filePtr);
    rewind(filePtr);
    sourceStr = (char*)malloc(fileSize + 1);
    sourceStr[fileSize] = '\0';
    fread(sourceStr, sizeof(char), fileSize, filePtr);
    fclose(filePtr);

    cl_context theContext;
    clGetCommandQueueInfo(  theHandle,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(  theHandle,
                            CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);


    theProgram = clCreateProgramWithSource(theContext, 1,
                            (const char **)& sourceStr, NULL, &err);

    clBuildProgram(theProgram, 0, NULL, NULL, NULL, NULL);
    theKernel = clCreateKernel(theProgram, kernelFile.c_str(), &err);

}

cl_int
cl_kernel_implementer (cl_command_queue theHandle, size_t n, cl_program& theProgram,
                        cl_kernel& theKernel  )
{
    cl_int err;
    size_t localSize = 64;
    // Number of total work items - localSize must be devisor
    size_t globalSize = ceil(n/(float)localSize)*localSize;

    err = clEnqueueNDRangeKernel(theHandle, theKernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);
    clFinish(theHandle);
    clReleaseProgram(theProgram);
    clReleaseKernel(theKernel);
    return err;
}