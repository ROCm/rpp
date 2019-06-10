#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"


#ifndef MOD_CL_PATH
#   error Kernel files base path not defined; undefined `MOD_CL_PATH`
#endif //MOD_CL_PATH

#define TO_STRING(x) #x

cl_int
cl_kernel_initializer ( cl_command_queue theQueue,
                        std::string kernelFile, std::string kernelName,
                        cl_program& theProgram, cl_kernel& theKernel)
{
    cl_int err;
    // File Handling
    char *sourceStr;
    size_t sourceSize;
    std::string kernelFile_cl = TO_STRING(MOD_CL_PATH) + kernelFile;
    std::cout << kernelFile_cl << std::endl;
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
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);


    theProgram = clCreateProgramWithSource(theContext, 1,
                            (const char **)& sourceStr, NULL, &err);

    clBuildProgram(theProgram, 0, NULL, NULL, NULL, NULL);
    theKernel = clCreateKernel(theProgram, kernelName.c_str(), &err);

}

cl_int
cl_kernel_implementer (cl_command_queue theQueue, size_t* globalDim3, size_t* localDim3,
                        cl_program& theProgram, cl_kernel& theKernel  )
{
    cl_int err;

    err = clEnqueueNDRangeKernel(theQueue, theKernel, 3, NULL, globalDim3, localDim3,
                                                              0, NULL, NULL);
    clFinish(theQueue);
    clReleaseProgram(theProgram);
    clReleaseKernel(theKernel);
    return err;
}
