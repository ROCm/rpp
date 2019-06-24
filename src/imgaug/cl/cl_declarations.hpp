#ifndef RPP_CL_IMGAUG_DECLATAIONS_H
#define RPP_CL_IMGAUG_DECLATAIONS_H

// ===== Utils

cl_int
cl_kernel_initializer ( cl_command_queue theQueue,
                        std::string kernelFile, std::string kernelName,
                        cl_program& theProgram, cl_kernel& theKernel);

cl_int
cl_kernel_implementer (cl_command_queue theHandle, size_t* gDim3, size_t* lDim3, cl_program& theProgram,
                        cl_kernel& theKernel  );


//===== Internal CL functions

RppStatus
brightness_contrast_cl (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32f alpha, Rpp32s beta,
                            RppiChnFormat chnFormat, unsigned int channel,
                            cl_command_queue theQueue);

RppStatus
contrast_stretch_cl (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32u newMin, Rpp32u newMax,
                            RppiChnFormat chnFormat, unsigned int channel,
                            cl_command_queue theQueue);
cl_int
gaussian_blur_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, unsigned int filterSize,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);
cl_int
flip_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiAxis flipAxis,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

RppStatus
convert_rgb2hsv_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr,RppiChnFormat chnFormat, unsigned int chanel,
                cl_command_queue theQueue);

RppStatus
convert_hsv2rgb_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr,RppiChnFormat chnFormat, unsigned int chanel,
                cl_command_queue theQueue);

RppStatus
hue_saturation_rgb_cl (cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f hue, Rpp32f Saturation,
                RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue);

RppStatus
hue_saturation_hsv_cl( cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f hue, Rpp32f Saturation,
                RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue);

RppStatus
bitwise_AND_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

RppStatus
bitwise_NOT_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

RppStatus
exclusive_OR_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

RppStatus
inclusive_OR_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

RppStatus
add_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
         RppiSize srcSize, cl_mem dstPtr,
         RppiChnFormat chnFormat, unsigned int channel,
         cl_command_queue theQueue);

RppStatus
subtract_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

RppStatus
absolute_difference_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                         RppiSize srcSize, cl_mem dstPtr,
                         RppiChnFormat chnFormat, unsigned int channel,
                         cl_command_queue theQueue);

RppStatus
bilateral_filter_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                      unsigned int filterSize, double sigmaI, double sigmaS,
                      RppiChnFormat chnFormat, unsigned int channel,
                      cl_command_queue theQueue);

RppStatus
gamma_correction_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, float gamma,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

RppStatus
accumulate_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

RppStatus
accumulate_weighted_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, double alpha,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

cl_int
box_filter_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, unsigned int filterSize,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);


cl_int
resize_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiSize dstSize, 
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

#endif //RPP_CL_IMGAUG_DECLATAIONS_H
