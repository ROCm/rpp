hipError_t
hipoc_kernel_caller()
{
    hipModule_t module;
    hipModuleLoad(&module, "brightness_contrast.cl.o");
    hipFunction_t function;
    hipModuleGetFunction(&function, module, "brightness_contrast");

    hipStream_t stream;
    hipStreamCreate(&stream);

    hipEvent_t  start = nullptr;
    hipEvent_t stop  = nullptr;

    void* config[] = {  HIP_LAUNCH_PARAM_BUFFER_POINTER, &argBuffer[0],
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                        HIP_LAUNCH_PARAM_END};

    hipHccModuleLaunchKernel(function, gdims[0],
                                       gdims[1], gdims[2],
                                       ldims[0], ldims[1],
                                       ldims[2], 0,
                                       0,
                                       nullptr,
                                      (void**)&config,
                                       start,
                                       stop);

    hipDeviceSynchronize();

}