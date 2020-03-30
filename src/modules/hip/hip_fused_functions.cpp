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
    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch( Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    return RPP_SUCCESS;
}