#include <hip/rpp_hip_common.hpp>
#include "hipoc_declarations.hpp"

hipError_t
hipoc_brightness_contrast( void* srcPtr, RppiSize srcSize,
                                void* dstPtr,
                                Rpp32f alpha, Rpp32s beta,
                                RppiChnFormat chnFormat, unsigned int channel,
                                hipStream_t theQueue )
{

    unsigned int height = srcSize.height;
    unsigned int width = srcSize.width;

    void* argBuffer[7];
    argBuffer[0] = &srcPtr;
    argBuffer[1] = &dstPtr;
    argBuffer[2] = &alpha;
    argBuffer[3] = &beta;
    argBuffer[4] = &height;
    argBuffer[5] = &width;
    argBuffer[6] = &channel;

    size_t argSize = 7*sizeof(void*);

    // Note if not working try similar one to cl args setting

    hipModule_t module;
    hipModuleLoad(&module, "/home/neel/jgeob/rpp-workspace/AMD-RPP/build/src/imgaug/cl/libbrightness_contrast.cl.a");
    hipFunction_t function;
    hipModuleGetFunction(&function, module, "brightness_contrast");

    hipStream_t stream;
    hipStreamCreate(&stream);

    hipEvent_t  start = nullptr;
    hipEvent_t stop  = nullptr;

    void* config[] = {  HIP_LAUNCH_PARAM_BUFFER_POINTER, &argBuffer[0],
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                        HIP_LAUNCH_PARAM_END };

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    size_t lDim3[3];
    lDim3[0] = 32;
    lDim3[1] = 32;
    lDim3[2] = channel;

    hipHccModuleLaunchKernel(function, gDim3[0],
                                       gDim3[1],
                                       gDim3[2],
                                       lDim3[0],
                                       lDim3[1],
                                       lDim3[2],
                                       0, 0,
                                       nullptr,
                                      (void**)&config,
                                       start,
                                       stop);

    hipDeviceSynchronize();

    return hipSuccess;

}