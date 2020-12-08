#include "hip/hip_runtime_api.h"
#include "hip_declarations.hpp"
#include <hip/rpp_hip_common.hpp>

RppStatus
color_twist_hip( Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, 
            float alpha, float beta, float hue_shift, float sat,
            RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
color_twist_hip_batch (Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "color_twist.cpp", "color_twist_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                                        plnpkdind, plnpkdind
                                                                                        );
    
    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip( Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiSize dstSize, Rpp32u crop_pox_x,
                                        Rpp32u crop_pos_y, Rpp32f mean, Rpp32f std_dev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}
RppStatus
crop_mirror_normalize_hip_batch( Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "crop_mirror_normalize.cpp", "crop_mirror_normalize_batch", vld, vgd, "")(srcPtr, dstPtr,                                                                                                                                                                                                                                                                               
                                                                    handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                    handle.GetInitHandle()->mem.mgpu.dstSize.width,       
                                                                    handle.GetInitHandle()->mem.mgpu.srcSize.width,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                    handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                    handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                    handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                                                                    handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                                                                    handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                                                                    handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                    handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                    handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                    handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                    channel,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                    //handle.GetBatchSize(),
                                                                    handle.GetInitHandle()->mem.mgpu.inc,
                                                                    handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                    plnpkdind,
                                                                    plnpkdind);
    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch( Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "","crop.cpp", "crop_batch", vld, vgd, "")(srcPtr, dstPtr,                                                                                                                                                                                                                                                                               
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.width,       
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        channel,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                                                                       // handle.GetBatchSize(),
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        plnpkdind, plnpkdind);

    return RPP_SUCCESS;
}

RppStatus
resize_crop_mirror_hip_batch( Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{                                                                                                                                                                                   
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    
    
    // int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    // int batch_size = handle.GetBatchSize();
    // InitHandle *handle_obj = handle.GetInitHandle();
    // Rpp32u max_height, max_width;
    // max_size(handle_obj->mem.mgpu.cdstSize.height, handle_obj->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    
    // std::vector<size_t> vld{16, 16, 1};
    // std::vector<size_t> vgd{max_width ,max_height , handle.GetBatchSize()};
    // std::string kernel_file  = "resize.cl";
    // std::string kernel_name = "resize_crop_mirror_batch";
    // get_kernel_name(kernel_name, tensor_info);
    
    handle.AddKernel("", "", "resize.cpp" , "resize_crop_mirror_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height, 
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,                                                                                                                                                                                                                                                                              
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.width,       
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        // tensor_info._in_channels,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                        channel,
                                                                        // handle.GetBatchSize(),
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        plnpkdind, plnpkdind);
    return RPP_SUCCESS;
} 