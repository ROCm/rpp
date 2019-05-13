#ifndef RPP_CL_IMGAUG_DECLATAIONS_H
#define RPP_CL_IMGAUG_DECLATAIONS_H

//===== Utils
cl_int
cl_kernel_initializer ( cl_command_queue theQueue,
                        std::string kernelFile, std::string kernelName,
                        cl_program& theProgram, cl_kernel& theKernel);

cl_int
cl_kernel_implementer (cl_command_queue theHandle, size_t* dim3, cl_program& theProgram,
                        cl_kernel& theKernel  );

//===== Internal CL functions

RppStatus
cl_brightness_contrast (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32f alpha, Rpp32s beta,
                            RppiChnFormat chnFormat, size_t channel,
                            cl_command_queue theQueue);

cl_int
cl_gaussian_blur(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, size_t filterSize,
                RppiChnFormat chnFormat, size_t channel,
                cl_command_queue theQueue);
cl_int
cl_flip(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiAxis flipAxis,
                RppiChnFormat chnFormat, size_t channel,
                cl_command_queue theQueue);


#endif //RPP_CL_IMGAUG_DECLATAIONS_H