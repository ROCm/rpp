#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"
#include <map>
#include <vector>

#ifndef MOD_CL_PATH
# error Kernel files base path not defined; undefined `MOD_CL_PATH`
#endif //MOD_CL_PATH

// Note: gcc throws stray issue without this preprocessor derivative being used
//       for reading MOD_CL_PATH used for reading kernel file
//#define TO_STRING(x) #x

class CLKernelManager
{
    public:
    ~CLKernelManager()
    {
        // release resources
        
        for(auto& kernel : _kernel_map)
            clReleaseKernel(kernel.second);
    }
    static CLKernelManager* obj() 
    {
        if(!_obj)
            _obj = new CLKernelManager();
        
        return _obj;
    }
    cl_kernel find( const std::string& kernel_name)
    {
        auto it = _kernel_map.find(kernel_name);
        if( it != _kernel_map.end())
            return it->second;
        
        return nullptr;
    }
    
    bool set(cl_kernel theKernel, const std::string& kernel_name)
    {
        if(find(kernel_name)!= nullptr)
            return false;
        _kernel_map.insert(std::make_pair(kernel_name, theKernel));
        return true;
    }
//    std::vector< cl_program> _prog_map;
private:
    std::map<std::string, cl_kernel> _kernel_map;
    static CLKernelManager* _obj;
    CLKernelManager(){};
};

CLKernelManager* CLKernelManager::_obj = NULL;


cl_int
cl_kernel_initializer ( cl_command_queue theQueue,
                        std::string kernelFile, std::string kernelName,
                        cl_program& theProgram, cl_kernel& theKernel)
{
    cl_int err;
    // File Handling
    char *sourceStr;
    size_t sourceSize;
    std::string kernelFile_cl = MOD_CL_PATH + kernelFile;
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
    // theKernel = clCreateKernel(theProgram, kernelName.c_str(), &err);

    return err;
}

cl_int
cl_kernel_implementer (cl_command_queue theQueue, size_t* globalDim3, size_t* localDim3,
                        cl_program& theProgram, cl_kernel& theKernel  )
{
    cl_int err;

    err = clEnqueueNDRangeKernel(theQueue, theKernel, 3, NULL, globalDim3, localDim3,
                                                              0, NULL, NULL);
    //clFinish(theQueue);
    //clReleaseProgram(theProgram);
    //clReleaseKernel(theKernel);
    return err;
}

//
///
//  Retreive program binary for all of the devices attached to the
//  program an and store the one for the device passed in
//
bool SaveProgramBinary(cl_program program, cl_device_id device, const std::string fileName)
{
    cl_uint numDevices = 0;
    cl_int errNum;

    // 1 - Query for number of devices attached to program
    errNum = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint),
                              &numDevices, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for number of devices." << std::endl;
        return false;
    }

    // 2 - Get all of the Device IDs
    cl_device_id *devices = new cl_device_id[numDevices];
    errNum = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                              sizeof(cl_device_id) * numDevices,
                              devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for devices." << std::endl;
        delete [] devices;
        return false;
    }

    // 3 - Determine the size of each program binary
    size_t *programBinarySizes = new size_t [numDevices];
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                              sizeof(size_t) * numDevices,
                              programBinarySizes, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for program binary sizes." << std::endl;
        delete [] devices;
        delete [] programBinarySizes;
        return false;
    }

    unsigned char **programBinaries = new unsigned char*[numDevices];
    for (cl_uint i = 0; i < numDevices; i++)
    {
        programBinaries[i] = new unsigned char[programBinarySizes[i]];
    }

    // 4 - Get all of the program binaries
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * numDevices,
                              programBinaries, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for program binaries" << std::endl;

        delete [] devices;
        delete [] programBinarySizes;
        for (cl_uint i = 0; i < numDevices; i++)
        {
            delete [] programBinaries[i];
        }
        delete [] programBinaries;
        return false;
    }

    // 5 - Finally store the binaries for the device requested out to disk for future reading.
    for (cl_uint i = 0; i < numDevices; i++)
    {
        // Store the binary just for the device requested.  In a scenario where
        // multiple devices were being used you would save all of the binaries out here.
        if (devices[i] == device)
        {
            FILE *fp = fopen(fileName.c_str(), "wb");
            fwrite(programBinaries[i], 1, programBinarySizes[i], fp);
            fclose(fp);
            break;
        }
    }

    // Cleanup
    delete [] devices;
    delete [] programBinarySizes;
    for (cl_uint i = 0; i < numDevices; i++)
    {
        delete [] programBinaries[i];
    }
    delete [] programBinaries;
    return true;
}


///
//  Attempt to create the program object from a cached binary.  Note that
//  on first run this will fail because the binary has not yet been created.
//
cl_int CreateProgramFromBinary(cl_command_queue theQueue, const std::string kernelFile,
                                const std::string binaryFile, std::string kernelName,
                                cl_program& theProgram, cl_kernel& theKernel)
{
    cl_int status = CL_SUCCESS, err;
    cl_context theContext;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
       
    theKernel = CLKernelManager::obj()->find(kernelName);

    if( theKernel != nullptr)
        return status;
    
    FILE *fp = fopen(binaryFile.c_str(), "rb");
    
    if (fp == NULL)
    {
        theProgram = NULL;
    }
    else {
        // Determine the size of the binary
        size_t binarySize;
        fseek(fp, 0, SEEK_END);
        binarySize = ftell(fp);
        rewind(fp);

        unsigned char *programBinary = new unsigned char[binarySize];
        fread(programBinary, 1, binarySize, fp);
        fclose(fp);

        cl_int errNum = 0;
        cl_int binaryStatus;
        theProgram = clCreateProgramWithBinary(theContext,
                                            1,
                                            &theDevice,
                                            &binarySize,
                                            (const unsigned char**)&programBinary,
                                            &binaryStatus,
                                            &errNum);
        delete [] programBinary;
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error loading program binary." << std::endl;
            return NULL;
        }

        if (binaryStatus != CL_SUCCESS)
        {
            std::cerr << "Invalid binary for device" << std::endl;
            return NULL;
        }

        errNum = clBuildProgram(theProgram, 0, NULL, NULL, NULL, NULL);
        if (errNum != CL_SUCCESS)
        {
            // Determine the reason for the error
            char buildLog[16384];
            clGetProgramBuildInfo(theProgram, theDevice, CL_PROGRAM_BUILD_LOG,
                                sizeof(buildLog), buildLog, NULL);

            std::cerr << "Error in program: " << std::endl;
            std::cerr << buildLog << std::endl;
            clReleaseProgram(theProgram);
            return NULL;
        }

    }

    
    if (theProgram == NULL)
    {
        //std::cout << "Binary not loaded, create from source..." << kernelFile << " "  << kernelName << std::endl;
        err = cl_kernel_initializer(theQueue,
                           kernelFile,
                           kernelName,
                           theProgram, theKernel);

        //std::cout << "Save program binary for future run..." << std::endl;
        if (SaveProgramBinary(theProgram, theDevice, binaryFile) == false)
        {
            std::cerr << "Failed to write program binary" << std::endl;
            clFinish(theQueue);
            clReleaseProgram(theProgram);
            clReleaseKernel(theKernel);
            return 1;
        }
        
    }
    else
    {
        //std::cout << "Read program from binary." << std::endl;
    }

    theKernel = clCreateKernel(theProgram, kernelName.c_str(), &err);
    clReleaseProgram(theProgram);
    CLKernelManager::obj()->set( theKernel, kernelName);
    return err;
}
