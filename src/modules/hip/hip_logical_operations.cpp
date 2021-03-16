#include "hip_declarations.hpp"

/******************** bitwise_AND ********************/

RppStatus
bitwise_AND_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "bitwise_AND.cpp", "bitwise_AND", vld, vgd, "")(srcPtr1,
                                                                             srcPtr2,
                                                                             dstPtr,
                                                                             srcSize.height,
                                                                             srcSize.width,
                                                                             channel);

    return RPP_SUCCESS;
}

RppStatus
bitwise_AND_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "bitwise_AND.cpp", "bitwise_AND_batch", vld, vgd, "")(srcPtr1,
                                                                                   srcPtr2,
                                                                                   dstPtr,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                   handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                   channel,
                                                                                   handle.GetInitHandle()->mem.mgpu.inc,
                                                                                   plnpkdind);

    return RPP_SUCCESS;
}

/******************** bitwise_NOT ********************/

RppStatus
bitwise_NOT_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31,channel};

    handle.AddKernel("", "", "bitwise_NOT.cpp", "bitwise_NOT", vld, vgd, "")(srcPtr,
                                                                             dstPtr,
                                                                             srcSize.height,
                                                                             srcSize.width,
                                                                             channel);

    return RPP_SUCCESS;
}

RppStatus
bitwise_NOT_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "bitwise_NOT.cpp", "bitwise_NOT_batch", vld, vgd, "")(srcPtr,
                                                                                   dstPtr,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                   handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                   channel,
                                                                                   handle.GetInitHandle()->mem.mgpu.inc,
                                                                                   plnpkdind);

    return RPP_SUCCESS;
}

/******************** exclusive_OR ********************/

RppStatus
exclusive_OR_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "exclusive_OR.cpp", "exclusive_OR", vld, vgd, "")(srcPtr1,
                                                                               srcPtr2,
                                                                               dstPtr,
                                                                               srcSize.height,
                                                                               srcSize.width,
                                                                               channel);

    return RPP_SUCCESS;
}

RppStatus
exclusive_OR_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "exclusive_OR.cpp", "exclusive_OR_batch", vld, vgd, "")(srcPtr1,
                                                                                     srcPtr2,
                                                                                     dstPtr,
                                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                     handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                     handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                     channel,
                                                                                     handle.GetInitHandle()->mem.mgpu.inc,
                                                                                     plnpkdind);

    return RPP_SUCCESS;
}

/******************** inclusive_OR ********************/

RppStatus
inclusive_OR_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "inclusive_OR.cpp", "inclusive_OR", vld, vgd, "")(srcPtr1,
                                                                               srcPtr2,
                                                                               dstPtr,
                                                                               srcSize.height,
                                                                               srcSize.width,
                                                                               channel);

    return RPP_SUCCESS;
}

RppStatus
inclusive_OR_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "inclusive_OR.cpp", "inclusive_OR_batch", vld, vgd, "")(srcPtr1,
                                                                                     srcPtr2,
                                                                                     dstPtr,
                                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                     handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                     handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                     channel,
                                                                                     handle.GetInitHandle()->mem.mgpu.inc,
                                                                                     plnpkdind);

    return RPP_SUCCESS;
}