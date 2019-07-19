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

bool SaveProgramBinary(cl_program program, cl_device_id device, const std::string fileName);

cl_int CreateProgramFromBinary(cl_command_queue theQueue, const std::string kernelFile, 
                                const std::string binaryFile, std::string kernelName,
                                cl_program& theProgram, cl_kernel& theKernel);

//===== Internal CL functions

RppStatus
brightness_cl (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32f alpha, Rpp32s beta,
                            RppiChnFormat chnFormat, unsigned int channel,
                            cl_command_queue theQueue);

RppStatus
contrast_cl (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32u newMin, Rpp32u newMax,
                            RppiChnFormat chnFormat, unsigned int channel,
                            cl_command_queue theQueue);
cl_int
blur_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, unsigned int filterSize,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);
cl_int
flip_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiAxis flipAxis,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

RppStatus
rgb_to_hsv_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr,RppiChnFormat chnFormat, unsigned int chanel,
                cl_command_queue theQueue);

RppStatus
hsv_to_rgb_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr,RppiChnFormat chnFormat, unsigned int chanel,
                cl_command_queue theQueue);

RppStatus
hueRGB_cl (cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f hue, Rpp32f Saturation,
                RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue);

RppStatus
hueHSV_cl( cl_mem srcPtr, RppiSize srcSize,
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
gamma_correction_cl ( cl_mem srcPtr1, RppiSize srcSize, 
                 cl_mem dstPtr,float gamma,
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

cl_int
resize_crop_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiSize dstSize,
                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,  
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

cl_int
rotate_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiSize dstSize, float angleDeg, 
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

RppStatus
blend_cl( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr, float alpha,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

cl_int
pixelate_cl(cl_mem srcPtr, RppiSize srcSize,cl_mem dstPtr, 
            unsigned int filterSize, unsigned int x1, unsigned int y1,
            unsigned int x2, unsigned int y2,RppiChnFormat chnFormat,
            unsigned int channel,cl_command_queue theQueue);

cl_int
jitter_cl( cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr,
           unsigned int minJitter,unsigned int maxJitter,
           RppiChnFormat chnFormat, unsigned int channel,
           cl_command_queue theQueue);

cl_int
fisheye_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, 
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

cl_int
lens_correction_cl( cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr,
           float strength,float zoom,
           RppiChnFormat chnFormat, unsigned int channel,
           cl_command_queue theQueue);

cl_int
snow_cl( cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr,
           float snowCoefficient,
           RppiChnFormat chnFormat, unsigned int channel,
           cl_command_queue theQueue);
RppStatus  
noise_add_gaussian_cl(cl_mem srcPtr,
                RppiSize srcSize,
                cl_mem dstPtr, 
                RppiNoise noiseType,RppiGaussParameter *noiseParameter,
                RppiChnFormat chnFormat, unsigned int channel, 
                cl_command_queue theQueue);

RppStatus
noise_add_snp_cl(cl_mem srcPtr, 
                RppiSize srcSize,
                cl_mem dstPtr, 
                RppiNoise noiseType,Rpp32f *noiseParameter,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

RppStatus
color_temperature_cl( cl_mem srcPtr1,
                 RppiSize srcSize, cl_mem dstPtr, float adjustmentValue,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

RppStatus
random_crop_letterbox_cl(  cl_mem srcPtr, RppiSize srcSize, 
                            cl_mem dstPtr, RppiSize dstSize, 
                            Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                            RppiChnFormat chnFormat, unsigned int channel,
                            cl_command_queue theQueue);

RppStatus
exposure_cl(    cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f exposureValue,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

RppStatus
rain_cl(    cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f rainValue, Rpp32u rainWidth, Rpp32u rainHeight,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

RppStatus
vignette_cl( cl_mem srcPtr1, RppiSize srcSize, 
                cl_mem dstPtr, float stdDev,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

RppStatus
fog_cl( cl_mem srcPtr, RppiSize srcSize, 
                Rpp32f fogValue,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

RppStatus
random_shadow_cl(    cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

cl_int
warp_affine_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiSize dstSize, float *affine, 
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue);

RppStatus
occlusion_cl(   cl_mem srcPtr1,RppiSize srcSize1, 
                cl_mem srcPtr2,RppiSize srcSize2, cl_mem dstPtr,
                const unsigned int x11,
                const unsigned int y11,
                const unsigned int x12,
                const unsigned int y12,
                const unsigned int x21,
                const unsigned int y21,
                const unsigned int x22,
                const unsigned int y22, 
                RppiChnFormat chnFormat,unsigned int channel,
                cl_command_queue theQueue);

#endif //RPP_CL_IMGAUG_DECLATAIONS_H
