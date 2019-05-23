#ifndef RPP_CL_IMGAUG_DECLATAIONS_H
#define RPP_CL_IMGAUG_DECLATAIONS_H

//===== Utils
cl_int
cl_kernel_initializer ( cl_command_queue theQueue,
                        std::string kernelFile, std::string kernelName,
                        cl_program& theProgram, cl_kernel& theKernel);

cl_int
cl_kernel_implementer (cl_command_queue theHandle, unsigned int* gDim3, unsigned int* lDim3, cl_program& theProgram,
                        cl_kernel& theKernel  );

//===== Internal CL functions

RppStatus
cl_brightness_contrast (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32f alpha, Rpp32s beta,
                            RppiChnFormat chnFormat, unsigned int channel,
                            cl_command_queue theQueue);

RppStatus
cl_contrast_stretch (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32u newMin, Rpp32u newMax,
                            RppiChnFormat chnFormat, unsigned int channel,
                            cl_command_queue theQueue);
cl_int
cl_gaussian_blur(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, unsigned int filterSize,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);
cl_int
cl_flip(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiAxis flipAxis,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

RppStatus 
cl_convert_rgb2hsv(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr,RppiChnFormat chnFormat, unsigned int chanel,
                cl_command_queue theQueue);

RppStatus 
cl_convert_hsv2rgb(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr,RppiChnFormat chnFormat, unsigned int chanel,
                cl_command_queue theQueue);

RppStatus
cl_hue_saturation_rgb (cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f hue, Rpp32f Saturation,
                RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue);

RppStatus
cl_hue_saturation_hsv( cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f hue, Rpp32f Saturation,
                RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue);
                
#endif //RPP_CL_IMGAUG_DECLATAIONS_H


//Hi Hello How are you?