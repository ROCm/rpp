#include "hip/hip_runtime_api.h"
#include "hip_declarations.hpp"
#include <hip/rpp_hip_common.hpp>

/******************** color_twist ********************/

RppStatus
color_twist_hip( Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, float alpha, float beta, float hue_shift, float sat, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
color_twist_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "color_twist.cpp", "color_twist_batch", vld, vgd, "")(srcPtr,
                                                                                   dstPtr,
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
                                                                                   plnpkdind,
                                                                                   plnpkdind);

    return RPP_SUCCESS;
}

/******************** crop_mirror_normalize ********************/

RppStatus
crop_mirror_normalize_hip(
    Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiSize dstSize, Rpp32u crop_pox_x, Rpp32u crop_pos_y,
    Rpp32f mean, Rpp32f std_dev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}