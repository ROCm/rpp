#include "hip_declarations.hpp"
// /********************** Dilate ************************/
RppStatus
dilate_hip ( Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31,channel};
        handle.AddKernel("", "", "dilate.cpp", "dilate_pkd", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                kernelSize
                                                                );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31,channel};
        handle.AddKernel("", "", "dilate.cpp", "dilate_pln", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                kernelSize
                                                                );
    }
    
    // //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
        
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}


RppStatus
dilate_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "dilate.cpp", "dilate_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}



/********************** Erode ************************/
RppStatus
erode_hip ( Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31,channel};
        handle.AddKernel("", "", "erode.cpp", "erode_pkd", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                kernelSize
                                                                );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31,channel};
        handle.AddKernel("", "", "erode.cpp", "erode_pln", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                kernelSize
                                                                );
    }
    
    // //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
        
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}


RppStatus
erode_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "erode.cpp", "erode_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}
