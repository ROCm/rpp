#include "hip/rpp_hip_common.hpp"
#include "hip_declarations.hpp"

/************* Fish eye ******************/
RppStatus
fisheye_hip(Rpp8u* srcPtr, RppiSize srcSize,
                Rpp8u* dstPtr,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "fish_eye.cpp", "fisheye_planar", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "fish_eye.cpp", "fisheye_packed", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
    }
    return RPP_SUCCESS;
}

RppStatus
fisheye_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "fish_eye.cpp", "fisheye_batch", vld, vgd, "")(srcPtr,
                                                                            dstPtr,
                                                                            handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                            handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                            handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                            handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                            handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                            handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                            handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                            handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                            channel,
                                                                            handle.GetInitHandle()->mem.mgpu.inc,
                                                                            plnpkdind);
    return RPP_SUCCESS;
}

/************* LensCorrection ******************/
RppStatus
lens_correction_hip( Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr,
           float strength,float zoom,
           RppiChnFormat chnFormat, unsigned int channel,
           rpp::Handle& handle)
{
    unsigned short counter=0;
    if (strength == 0)
        strength = 0.000001;
    float halfWidth = (float)srcSize.width / 2.0;
    float halfHeight = (float)srcSize.height / 2.0;
    float correctionRadius = (float)sqrt((float)srcSize.width * srcSize.width + srcSize.height * srcSize.height) / (float)strength;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "lens_correction.cpp", "lenscorrection_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        strength,
                                                                        zoom,
                                                                        halfWidth,
                                                                        halfHeight,
                                                                        correctionRadius,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "lens_correction.cpp", "lenscorrection_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        strength,
                                                                        zoom,
                                                                        halfWidth,
                                                                        halfHeight,
                                                                        correctionRadius,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }
    return RPP_SUCCESS;
}

RppStatus
lens_correction_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "lens_correction.cpp", "lens_correction_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
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

/************* Flip ******************/

RppStatus
flip_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr, uint flipAxis,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)

{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (flipAxis == 1)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_vertical_planar", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
        }
        else if (flipAxis == 0)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_horizontal_planar", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
        }
        else if (flipAxis == 2)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_bothaxis_planar", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        if (flipAxis == 1)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_vertical_packed", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
        }
        else if (flipAxis == 0)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_horizontal_packed", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
        }
        else if (flipAxis == 2)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_bothaxis_packed", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
        }
    }
     return RPP_SUCCESS;
}

RppStatus
flip_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;

    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "flip.cpp", "flip_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,   
                                                                        channel,
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        plnpkdind);


    return RPP_SUCCESS;
}

/************* Resize ******************/
RppStatus
resize_hip(Rpp8u * srcPtr, RppiSize srcSize,
                Rpp8u * dstPtr, RppiSize dstSize,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "resize.cpp", "resize_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "resize.cpp", "resize_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
    }
    return RPP_SUCCESS;
}

RppStatus
resize_hip_batch (   Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = channel;
    unsigned int padding = 0;
    unsigned int type = 0;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "resize.cpp", "resize_crop_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight, 
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        channel,
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        padding,
                                                                        type,
                                                                        plnpkdind, plnpkdind);
    return RPP_SUCCESS;
}

/************* Resize Crop ******************/
RppStatus
resize_crop_hip(Rpp8u * srcPtr, RppiSize srcSize,
                Rpp8u * dstPtr, RppiSize dstSize,
                Rpp32u x1, Rpp32u x2, Rpp32u y1, Rpp32u y2, 
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    unsigned int type = 0,  padding = 0;
    unsigned int width,height;
    if(type == 1)
    {
        width = dstSize.width - padding * 2;
        height = dstSize.height - padding * 2;
    }
    else
    {
        width = dstSize.width;
        height = dstSize.height;
    }
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{width, height, channel};
        handle.AddKernel("", "", "resize.cpp", "resize_crop_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        height,
                                                                        width,
                                                                        x1,
                                                                        x2,
                                                                        y1,
                                                                        y2,
                                                                        padding,
                                                                        type,
                                                                        channel
                                                                        );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{width, height, channel};
        handle.AddKernel("", "", "resize.cpp", "resize_crop_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        height,
                                                                        width,
                                                                        x1,
                                                                        x2,
                                                                        y1,
                                                                        y2,
                                                                        padding,
                                                                        type,
                                                                        channel
                                                                        );
    }
    return RPP_SUCCESS;
}
RppStatus
resize_crop_hip_batch (   Rpp8u * srcPtr, Rpp8u * dstPtr,  rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;

    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    unsigned int padding = 10;
    unsigned int type = 1;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);


    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "resize.cpp", "resize_crop_batch", vld, vgd, "")(srcPtr, dstPtr,
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
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        channel,
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        padding,
                                                                        type,
                                                                        plnpkdind, plnpkdind);
    return RPP_SUCCESS;
}
/*************Rotate ******************/
RppStatus
rotate_hip(Rpp8u * srcPtr, RppiSize srcSize,
                Rpp8u * dstPtr, RppiSize dstSize, float angleDeg,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "rotate.cpp", "rotate_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        angleDeg,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "rotate.cpp", "rotate_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        angleDeg,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
    }
    return RPP_SUCCESS;
}
RppStatus
rotate_hip_batch (   Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = channel;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "rotate.cpp", "rotate_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                plnpkdind, plnpkdind
                                                                );
    return RPP_SUCCESS;
}

/************* Warp affine ******************/
RppStatus
warp_affine_hip(Rpp8u * srcPtr, RppiSize srcSize,
                Rpp8u * dstPtr, RppiSize dstSize, float *affine,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{

    float affine_inv[6];
    float det; 
    det = (affine[0] * affine [4])  - (affine[1] * affine[3]);
    affine_inv[0] = affine[4]/ det;
    affine_inv[1] = (- 1 * affine[1])/ det;
    affine_inv[2] = -1 * affine[2];
    affine_inv[3] = (-1 * affine[3]) /det ;
    affine_inv[4] = affine[0]/det;
    affine_inv[5] = -1 * affine[5];

    float *affine_matrix;
    Rpp32u* affine_array;
    hipMalloc(&affine_array, sizeof(float)*6);
    hipMemcpy(affine_array,affine_inv,sizeof(float)*6,hipMemcpyHostToDevice);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "warp_affine.cpp", "warp_affine_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        affine_array,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "warp_affine.cpp", "warp_affine_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        affine_array,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
    }

    else
    {std::cerr << "Internal error: Unknown Channel format";}

    return RPP_SUCCESS;
}

RppStatus
warp_affine_hip_batch(Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle,Rpp32f *affine,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;

    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    
    Rpp32u nBatchSize = handle.GetBatchSize();
    
    Rpp32f *hip_affine;
    hipMalloc(&hip_affine, nBatchSize * 6 * sizeof(Rpp32f));
    hipMemcpy(hip_affine, affine, nBatchSize * 6 * sizeof(Rpp32f), hipMemcpyHostToDevice);
    
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "warp_affine.cpp", "warp_affine_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                hip_affine,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                plnpkdind
                                                                );
    hipFree(&hip_affine);
    return RPP_SUCCESS;
}



/************* Warp Perspective ******************/
RppStatus
warp_perspective_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr,
                    RppiSize dstSize, float *perspective, RppiChnFormat chnFormat,
                    unsigned int channel, rpp::Handle& handle)
{

    float perspective_inv[9];
    float det; //for Deteminent
    det = (perspective[0] * ((perspective[4] * perspective[8]) - (perspective[5] * perspective[7]))) - (perspective[1] * ((perspective[3] * perspective[8]) - (perspective[5] * perspective[6]))) + (perspective[2] * ((perspective[3] * perspective[7]) - (perspective[4] * perspective[6])));
    perspective_inv[0] = (1 * ((perspective[4] * perspective[8]) - (perspective[5] * perspective[7]))) / det;
    perspective_inv[1] = (-1 * ((perspective[1] * perspective[8]) - (perspective[7] * perspective[2]))) / det;
    perspective_inv[2] = (1 * ((perspective[1] * perspective[5]) - (perspective[4] * perspective[2]))) / det;
    perspective_inv[3] = (-1 * ((perspective[3] * perspective[8]) - (perspective[6] * perspective[5]))) / det;
    perspective_inv[4] = (1 * ((perspective[0] * perspective[8]) - (perspective[6] * perspective[2]))) / det;
    perspective_inv[5] = (-1 * ((perspective[0] * perspective[5]) - (perspective[3] * perspective[2]))) / det;
    perspective_inv[6] = (1 * ((perspective[3] * perspective[7]) - (perspective[6] * perspective[4]))) / det;
    perspective_inv[7] = (-1 * ((perspective[0] * perspective[7]) - (perspective[6] * perspective[1]))) / det;
    perspective_inv[8] = (1 * ((perspective[0] * perspective[4]) - (perspective[3] * perspective[1]))) / det;

    float *perspective_matrix;
    Rpp32f* perspective_array;
    hipMalloc(&perspective_array, sizeof(float) * 9);
    hipMemcpy(perspective_array, perspective_inv,sizeof(float) * 9, hipMemcpyHostToDevice);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "warp_perspective.cpp", "warp_perspective_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        perspective_array,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
        
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "warp_perspective.cpp", "warp_perspective_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        perspective_array,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
       
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }

    
    return RPP_SUCCESS;
}

RppStatus
warp_perspective_hip_batch(Rpp8u * srcPtr, Rpp8u * dstPtr,  rpp::Handle& handle,Rpp32f *perspective, 
                        RppiChnFormat chnFormat, unsigned int channel)
    
{
    int plnpkdind;

    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    
    Rpp32f* perspective_array;
    hipMalloc(&perspective_array, sizeof(float) * 9 * handle.GetBatchSize());
    hipMemcpy(perspective_array, perspective, sizeof(float) * 9 * handle.GetBatchSize(), hipMemcpyHostToDevice);

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "warp_perspective.cpp", "warp_perspective_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                        perspective_array,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,   
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        channel,
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        plnpkdind);
    
    hipFree(perspective_array);

    return RPP_SUCCESS;
}
    
    
    
//     Rpp32u nBatchSize = handle.GetBatchSize(); 
//     float perspective_inv[9];
//     float det; //for Deteminent
//     short counter;
//     size_t gDim3[3];
//     Rpp32f* perspective_array;
//     hipMalloc(&perspective_array,sizeof(float)*9);

//     unsigned int maxsrcHeight, maxsrcWidth, maxdstHeight, maxdstWidth;
//     maxsrcHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
//     maxsrcWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
//     maxdstHeight = handle.GetInitHandle()->mem.mgpu.cdstSize.height[0];
//     maxdstWidth = handle.GetInitHandle()->mem.mgpu.cdstSize.width[0];
//     for(int i = 0 ; i < nBatchSize ; i++)
//     {
//         if(maxsrcHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
//             maxsrcHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
//         if(maxsrcWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
//             maxsrcWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        
//         if(maxdstHeight < handle.GetInitHandle()->mem.mgpu.cdstSize.height[i])
//             maxdstHeight = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
//         if(maxdstWidth < handle.GetInitHandle()->mem.mgpu.cdstSize.width[i])
//             maxdstWidth = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
//     }

//     Rpp8u * srcPtr1;
//     hipMalloc(&srcPtr1, sizeof(unsigned char) * maxsrcHeight * maxsrcWidth * channel);
//     Rpp8u * dstPtr1;
//     hipMalloc(&dstPtr1, sizeof(unsigned char) * maxdstHeight * maxdstWidth * channel);

//     int ctr;
//     size_t srcbatchIndex = 0, dstbatchIndex = 0;
//     size_t index =0;

//     for(int i =0; i<nBatchSize; i++)
//     {
//         hipMemcpy(srcPtr1, srcPtr+srcbatchIndex , sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * 
//         handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
//         det = (perspective[index] * ((perspective[index+4] * perspective[index+8]) - (perspective[index+5] * perspective[index+7]))) - (perspective[index+1] * ((perspective[index+3] * perspective[index+8]) - (perspective[index+5] * perspective[index+6]))) + (perspective[index+2] * ((perspective[index+3] * perspective[index+7]) - (perspective[index+4] * perspective[index+6])));
//         perspective_inv[0] = (1 * ((perspective[index+4] * perspective[index+8]) - (perspective[index+5] * perspective[index+7]))) / det;
//         perspective_inv[1] = (-1 * ((perspective[index+1] * perspective[index+8]) - (perspective[index+7] * perspective[index+2]))) / det;
//         perspective_inv[2] = (1 * ((perspective[index+1] * perspective[index+5]) - (perspective[index+4] * perspective[index+2]))) / det;
//         perspective_inv[3] = (-1 * ((perspective[index+3] * perspective[index+8]) - (perspective[index+6] * perspective[index+5]))) / det;
//         perspective_inv[4] = (1 * ((perspective[index] * perspective[index+8]) - (perspective[index+6] * perspective[index+2]))) / det;
//         perspective_inv[5] = (-1 * ((perspective[index] * perspective[index+5]) - (perspective[index+3] * perspective[index+2]))) / det;
//         perspective_inv[6] = (1 * ((perspective[index+3] * perspective[index+7]) - (perspective[index+6] * perspective[index+4]))) / det;
//         perspective_inv[7] = (-1 * ((perspective[index] * perspective[index+7]) - (perspective[index+6] * perspective[index+1]))) / det;
//         perspective_inv[8] = (1 * ((perspective[index] * perspective[index+4]) - (perspective[index+3] * perspective[index+1]))) / det;

//         hipMemcpy(perspective_array,perspective,sizeof(float)*9,hipMemcpyHostToDevice);

//         if (chnFormat == RPPI_CHN_PLANAR)
//         {
//             std::vector<size_t> vld{32, 32, 1};
//             std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.cdstSize.width[i], handle.GetInitHandle()->mem.mgpu.cdstSize.height[i], channel};
//             handle.AddKernel("", "", "warp_perspective.cpp", "warp_perspective_pln", vld, vgd, "")(srcPtr1,
//                                                                         dstPtr1,
//                                                                         perspective_array,
//                                                                         handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
//                                                                         handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
//                                                                         handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
//                                                                         handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
//                                                                         channel
//                                                                         );
            
//         }
//         else if (chnFormat == RPPI_CHN_PACKED)
//         {   
//             std::vector<size_t> vld{32, 32, 1};
//             std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.cdstSize.width[i], handle.GetInitHandle()->mem.mgpu.cdstSize.height[i], channel};
//             handle.AddKernel("", "", "warp_perspective.cpp", "warp_perspective_pkd", vld, vgd, "")(srcPtr1,
//                                                                         dstPtr1,
//                                                                         perspective_array,
//                                                                         handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
//                                                                         handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
//                                                                         handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
//                                                                         handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
//                                                                         channel
//                                                                         );
           
//         }

//         else
//         {std::cerr << "Internal error: Unknown Channel format";}

        
//         hipMemcpy(dstPtr+dstbatchIndex, dstPtr1, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);        srcbatchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
//         dstbatchIndex += handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * channel * sizeof(unsigned char);
//         index = index + 9;
//     }
//     return RPP_SUCCESS;
//     /* CLrelease should come here */
// }
// /************* Scale ******************/
RppStatus
scale_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr, RppiSize dstSize,
 Rpp32f percentage, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    percentage /= 100;
    unsigned int dstheight = (Rpp32s) (percentage * (Rpp32f) srcSize.height);
    unsigned int dstwidth = (Rpp32s) (percentage * (Rpp32f) srcSize.width);    
    unsigned short counter=0;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "scale.cpp", "scale_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel,
                                                                        dstheight,
                                                                        dstwidth
                                                                        );
        
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {   
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "scale.cpp", "scale_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel,
                                                                        dstheight,
                                                                        dstwidth
                                                                        );
        
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }

    return RPP_SUCCESS;
}

RppStatus
scale_hip_batch (Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
     int plnpkdind;

    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    unsigned int padding = 0;
    unsigned int type = 0;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "scale.cpp", "scale_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,   
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        channel,
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        plnpkdind);
    return RPP_SUCCESS;
}