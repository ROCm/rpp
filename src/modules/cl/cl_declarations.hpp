#ifndef CL_DECLATAIONS_H
#define CL_DECLATAIONS_H

#include "hip/rpp/handle.hpp"

// ===== Utils

cl_int
cl_kernel_initializer ( rpp::Handle& handle,
                        std::string kernelFile, std::string kernelName,
                        cl_program& theProgram, cl_kernel& theKernel);

cl_int
cl_kernel_implementer (rpp::Handle& handle, size_t* gDim3, size_t* lDim3, cl_program& theProgram,
                        cl_kernel& theKernel  );

bool SaveProgramBinary(cl_program program, cl_device_id device, const std::string fileName);

cl_int CreateProgramFromBinary(rpp::Handle& handle, const std::string kernelFile,
                                const std::string binaryFile, std::string kernelName,
                                cl_program& theProgram, cl_kernel& theKernel);
void max_size(Rpp32u *height, Rpp32u *width, unsigned int batch_size, unsigned int *max_height, unsigned int *max_width);

void get_kernel_name(std::string &kernel_name, RPPTensorFunctionMetaData &tensor_info);
//===== Internal CL functions

RppStatus
data_object_copy_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                    RppiChnFormat chnFormat, unsigned int channel,
                    rpp::Handle& handle);
RppStatus
data_object_copy_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                            RppiChnFormat chnFormat, unsigned int channel
                            );
RppStatus
local_binary_pattern_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                        RppiChnFormat chnFormat, unsigned int channel,
                        rpp::Handle& handle);
RppStatus
local_binary_pattern_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
gaussian_image_pyramid_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                         Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
gaussian_image_pyramid_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
control_flow_cl(cl_mem srcPtr1, cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr,
 Rpp32u type, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
control_flow_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, Rpp32u type, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
convert_bit_depth_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u type,
             RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
convert_bit_depth_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, 
                            Rpp32u type,rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
laplacian_image_pyramid_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                            Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat,
                            unsigned int channel, rpp::Handle& handle);
RppStatus
laplacian_image_pyramid_cl_batch (   cl_mem srcPtr, cl_mem dstPtr,rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
canny_edge_detector_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp8u minThreshold,
                        Rpp8u maxThreshold, RppiChnFormat chnFormat, unsigned int channel,
                        rpp::Handle& handle);
RppStatus
canny_edge_detector_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                            RppiChnFormat chnFormat, unsigned int channel);
RppStatus
sobel_filter_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u sobelType,
            RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);

RppStatus
sobel_filter_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);

RppStatus
box_filter_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize,
                RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
box_filter_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
median_filter_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                 Rpp32u kernelSize, RppiChnFormat chnFormat,
                 unsigned int channel, rpp::Handle& handle);

RppStatus
median_filter_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);

RppStatus
non_max_suppression_cl( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                        Rpp32u kernelSize, RppiChnFormat chnFormat,
                        unsigned int channel, rpp::Handle& handle);
RppStatus
non_max_suppression_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);

RppStatus
bilateral_filter_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                      unsigned int filterSize, double sigmaI, double sigmaS,
                      RppiChnFormat chnFormat, unsigned int channel,
                      rpp::Handle& handle);
RppStatus
bilateral_filter_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
gaussian_filter_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                    Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat,
                    unsigned int channel, rpp::Handle& handle);
RppStatus
gaussian_filter_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
custom_convolution_cl( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                     Rpp32f* kernel, RppiSize kernelSize, RppiChnFormat chnFormat,
                     unsigned int channel, rpp::Handle& handle);
RppStatus
custom_convolution_cl_batch (cl_mem srcPtr, cl_mem dstPtr, Rpp32f *kernel, RppiSize KernelSize,
                        rpp::Handle& handle,RppiChnFormat chnFormat, unsigned int channel);
RppStatus
fisheye_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle);
RppStatus
fisheye_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
lens_correction_cl( cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr,
           float strength,float zoom,
           RppiChnFormat chnFormat, unsigned int channel,
           rpp::Handle& handle);

RppStatus
lens_correction_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);

RppStatus
brightness_cl (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32f alpha, Rpp32s beta,
                            RppiChnFormat chnFormat, unsigned int channel,
                            rpp::Handle& handle);

RppStatus brightness_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                                RppiChnFormat chnFormat, unsigned int channel);
RppStatus
gamma_correction_cl ( cl_mem srcPtr,RppiSize srcSize,
                 cl_mem dstPtr, float gamma,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);

RppStatus gamma_correction_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                                RppiChnFormat chnFormat, unsigned int channel);
RppStatus
contrast_cl (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32u newMin, Rpp32u newMax,
                            RppiChnFormat chnFormat, unsigned int channel,
                            rpp::Handle& handle);

RppStatus
contrast_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);

RppStatus
pixelate_cl(cl_mem srcPtr, RppiSize srcSize,cl_mem dstPtr,
            RppiChnFormat chnFormat,
            unsigned int channel,rpp::Handle& handle);
RppStatus
pixelate_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
exposure_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32f exposureValue,
            RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);

RppStatus
exposure_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
jitter_cl( cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr,
           unsigned int kernelSize,
           RppiChnFormat chnFormat, unsigned int channel,
           rpp::Handle& handle);
RppStatus
jitter_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
noise_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                Rpp32f noiseProbability,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle);
RppStatus
noise_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
blend_cl( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr, float alpha,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
blend_cl_batch (cl_mem srcPtr1, cl_mem srcPtr2,cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
rain_cl(cl_mem srcPtr, RppiSize srcSize,cl_mem dstPtr, Rpp32f rainPercentage,
        Rpp32u rainWidth, Rpp32u rainHeight, Rpp32f transparency,
        RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
rain_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);

RppStatus
snow_cl( cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr,
           float snowCoefficient,
           RppiChnFormat chnFormat, unsigned int channel,
           rpp::Handle& handle);

RppStatus
snow_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);

RppStatus
fog_cl( cl_mem srcPtr, RppiSize srcSize,cl_mem dstPtr, Rpp32f fogValue, RppiChnFormat chnFormat,
        unsigned int channel, rpp::Handle& handle);

RppStatus
fog_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
random_shadow_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u x1, Rpp32u y1,
                 Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, Rpp32u maxSizeX,
                 Rpp32u maxSizeY, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
random_shadow_cl_batch(   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
histogram_balance_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle);
RppStatus
histogram_balance_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
absolute_difference_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
absolute_difference_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr,
                 rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
accumulate_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
accumulate_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, rpp::Handle& handle,
                     RppiChnFormat chnFormat, unsigned int channel);
RppStatus
accumulate_weighted_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, Rpp32f alpha,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
accumulate_weighted_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2,rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
add_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
add_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
subtract_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
subtract_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
magnitude_cl( cl_mem srcPtr1,cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr,
              RppiChnFormat chnFormat, unsigned int channel,
              rpp::Handle& handle);
RppStatus
magnitude_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
phase_cl( cl_mem srcPtr1,cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr,
         RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
phase_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
multiply_cl( cl_mem srcPtr1,cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr,
            RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
multiply_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
accumulate_squared_cl(cl_mem srcPtr, RppiSize srcSize, RppiChnFormat chnFormat,
                    unsigned int channel, rpp::Handle& handle);
RppStatus
accumulate_squared_cl_batch ( cl_mem srcPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
color_temperature_cl( cl_mem srcPtr1,
                 RppiSize srcSize, cl_mem dstPtr, int adjustmentValue,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
color_temperature_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                                RppiChnFormat chnFormat, unsigned int channel);
RppStatus
vignette_cl( cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, float stdDev,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle);
RppStatus
vignette_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                                RppiChnFormat chnFormat, unsigned int channel);
RppStatus
hueRGB_cl ( cl_mem srcPtr,RppiSize srcSize,
                 cl_mem dstPtr, float hue_factor,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);

RppStatus
hueRGB_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);

RppStatus
saturationRGB_cl ( cl_mem srcPtr,RppiSize srcSize,
                 cl_mem dstPtr, float saturation_factor,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);

RppStatus
saturationRGB_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
channel_extract_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                    Rpp32u extractChannelNumber, RppiChnFormat chnFormat,
                    unsigned int channel, rpp::Handle& handle);
RppStatus
channel_extract_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                                RppiChnFormat chnFormat, unsigned int channel);
RppStatus
channel_combine_cl(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem srcPtr3,
                    RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat,
                    unsigned int channel, rpp::Handle& handle);
RppStatus
channel_combine_cl_batch ( cl_mem srcPtr1, cl_mem srcPtr2, cl_mem srcPtr3,
                            cl_mem dstPtr,rpp::Handle& handle,
                            RppiChnFormat chnFormat, unsigned int channel);
RppStatus
look_up_table_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,Rpp8u* lutPtr,
 RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
look_up_table_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, Rpp8u* lutPtr,rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
dilate_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize,
            RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);

RppStatus
dilate_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
erode_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize,
            RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
erode_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
flip_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, uint flipAxis,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle);
RppStatus
flip_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
bitwise_AND_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
bitwise_AND_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr,
                 rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

RppStatus
bitwise_NOT_cl ( cl_mem srcPtr1,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
bitwise_NOT_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);

RppStatus
exclusive_OR_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
exclusive_OR_cl_batch  ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr,
                 rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

RppStatus
inclusive_OR_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
inclusive_OR_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr,
                 rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
thresholding_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp8u min,
 Rpp8u max, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
thresholding_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
min_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
min_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
max_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle);
RppStatus
max_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
min_max_loc_cl(cl_mem srcPtr, RppiSize srcSize, Rpp8u* min, Rpp8u* max, Rpp32u* minLoc,
                Rpp32u* maxLoc, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
min_max_loc_cl_batch(cl_mem srcPtr, cl_mem dstPtr, Rpp8u *min, Rpp8u *max,
                    Rpp32u *minLoc, Rpp32u *maxLoc, rpp::Handle& handle,
                    RppiChnFormat chnFormat, unsigned int channel);
RppStatus
integral_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat,
             unsigned int channel, rpp::Handle& handle);
RppStatus
integral_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                    RppiChnFormat chnFormat, unsigned int channel);
RppStatus
mean_stddev_cl(cl_mem srcPtr, RppiSize srcSize, Rpp32f *mean, Rpp32f *stddev, RppiChnFormat chnFormat,
                                                         unsigned int channel, rpp::Handle& handle);
RppStatus
mean_stddev_cl_batch(cl_mem srcPtr, Rpp32f *mean, Rpp32f *stddev, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
resize_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiSize dstSize,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle);
RppStatus
resize_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel = 3,
                        RPPTensorDataType dataType = RPPTensorDataType::U8);
RppStatus
resize_crop_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiSize dstSize,
                Rpp32u x1, Rpp32u x2, Rpp32u y1, Rpp32u y2,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle);
RppStatus
resize_crop_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle,
                     RppiChnFormat chnFormat, unsigned int channel = 3,
                     RPPTensorDataType dataType = RPPTensorDataType::U8);

RppStatus
warp_affine_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, float *affine,
                RppiChnFormat chnFormat, unsigned int channel);

RppStatus
warp_affine_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiSize dstSize, float *affine,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle);
RppStatus
warp_perspective_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                    RppiSize dstSize, float *perspective, RppiChnFormat chnFormat,
                    unsigned int channel, rpp::Handle& handle);
RppStatus
warp_perspective_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, float *perspective,
                RppiChnFormat chnFormat, unsigned int channel);

RppStatus
scale_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiSize dstSize,
 Rpp32f percentage, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
scale_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiSize dstSize,
 Rpp32f percentage, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
scale_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
harris_corner_detector_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
 Rpp32u gaussianKernelSize, Rpp32f stdDev, Rpp32u kernelSize, Rpp32f kValue,
 Rpp32f threshold, Rpp32u nonmaxKernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
harris_corner_detector_cl_batch(cl_mem srcPtr, cl_mem dstPtr,rpp::Handle& handle,
                                RppiChnFormat chnFormat, unsigned int channel);
RppStatus
match_template_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                 cl_mem templateImage, RppiSize templateImageSize,
                 RppiChnFormat chnFormat,unsigned int channel, rpp::Handle& handle);
RppStatus
match_template_cl_batch(cl_mem srcPtr, RppiSize *srcSize, cl_mem dstPtr,cl_mem templateImage, 
 RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
fast_corner_detector_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                        Rpp32u numOfPixels, Rpp8u threshold, Rpp32u nonmaxKernelSize,
                        RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
fast_corner_detector_cl_batch( cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);
RppStatus
reconstruction_laplacian_image_pyramid_cl(cl_mem srcPtr1, RppiSize srcSize1,
 cl_mem srcPtr2, RppiSize srcSize2, cl_mem dstPtr, Rpp32f stdDev, Rpp32u kernelSize,
  RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
reconstruction_laplacian_image_pyramid_cl_batch( cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);

RppStatus
tensor_add_cl( Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle);

RppStatus
tensor_subtract_cl( Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle);

RppStatus
tensor_multiply_cl( Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle);
RppStatus
tensor_matrix_multiply_cl(cl_mem srcPtr1, cl_mem srcPtr2, Rpp32u* tensorDimensionValues1, Rpp32u* tensorDimensionValues2, cl_mem dstPtr, rpp::Handle& handle);
RppStatus
tensor_look_up_table_cl(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, 
                        cl_mem srcPtr, cl_mem dstPtr, Rpp8u* lutPtr, rpp::Handle& handle);
RppStatus
tensor_convert_bit_depth_cl( Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, cl_mem srcPtr,
                             cl_mem dstPtr, Rpp32u type, rpp::Handle& handle);
RppStatus
occlusion_cl(   cl_mem srcPtr1,RppiSize srcSize1,
                cl_mem srcPtr2,RppiSize srcSize2, cl_mem dstPtr,//Destiation Size is Same as the Second Image's Dimensions
                const unsigned int x11,
                const unsigned int y11,
                const unsigned int x12,
                const unsigned int y12,
                const unsigned int x21,
                const unsigned int y21,
                const unsigned int x22,
                const unsigned int y22,
                RppiChnFormat chnFormat,unsigned int channel,
                rpp::Handle& handle);
RppStatus
occlusion_cl_batch (cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel);

RppStatus
rotate_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiSize dstSize, float angleDeg,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle);
RppStatus
rotate_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat = RPPI_CHN_PACKED, 
                        unsigned int channel = 3, RPPTensorDataType dataType = RPPTensorDataType::U8);

RppStatus
histogram_cl(cl_mem srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);            

RppStatus
remap_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u* rowRemapTable, Rpp32u* colRemapTable,
RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);

RppStatus
remap_cl_batch(cl_mem srcPtr, cl_mem dstPtr, Rpp32u* rowRemapTable, Rpp32u* colRemapTable,
         rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

/* Fused Functions */
RppStatus
color_twist_cl( cl_mem srcPtr,RppiSize srcSize,
                cl_mem dstPtr, const float alpha/*contrast*/, const float beta /*brightness*/, const float hue_factor /*hue factor*/, const float sat /* saturation_factor*/,
                RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
color_twist_cl_batch ( cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,  RppiChnFormat chnFormat = RPPI_CHN_PACKED, 
                unsigned int channel = 3, RPPTensorDataType dataType = RPPTensorDataType::U8);

RppStatus
crop_mirror_normalize_cl( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiSize dstSize, Rpp32u crop_pox_x,
                                        Rpp32u crop_pos_y, Rpp32f mean, Rpp32f std_dev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
crop_mirror_normalize_cl_batch( cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, 
                                RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_cl_batch( cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat = RPPI_CHN_PACKED, unsigned int channel = 3, RPPTensorDataType dataType = RPPTensorDataType::U8 );
RppStatus
resize_crop_mirror_cl_batch( cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat = RPPI_CHN_PACKED,
                                         unsigned int channel = 3, RPPTensorDataType dataType = RPPTensorDataType::U8);
/* Fused Functions End*/

#endif //CL_DECLATAIONS_H
