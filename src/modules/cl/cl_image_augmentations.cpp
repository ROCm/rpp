#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"


/****************** Brightness ******************/
RppStatus
brightness_cl ( cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32f alpha, Rpp32s beta,
                            RppiChnFormat chnFormat, unsigned int channel,
                            rpp::Handle& handle) {

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "brightness_contrast.cl", "brightness_contrast", vld, vgd, "")(srcPtr, dstPtr, alpha, beta, srcSize.height, srcSize.width, channel);
    return RPP_SUCCESS;

}

RppStatus
brightness_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "brightness_contrast.cl", "brightness_batch", vld, vgd, "")(srcPtr, dstPtr,
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

/****************  Gamma correction *******************/

RppStatus
gamma_correction_cl ( cl_mem srcPtr,RppiSize srcSize,
                 cl_mem dstPtr, float gamma,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "gamma_correction.cl", "gamma_correction", vld, vgd, "")(srcPtr, dstPtr, gamma, srcSize.height, srcSize.width, channel);
    return RPP_SUCCESS;

}


RppStatus
gamma_correction_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "gamma_correction.cl", "gamma_correction_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
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

/***************** Contrast *********************/

RppStatus
contrast_cl (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32u newMin, Rpp32u newMax,
                            RppiChnFormat chnFormat, unsigned int channel,
                            rpp::Handle& handle)
{
    unsigned short counter=0;
    Rpp32u min = 0; /* Kernel has to be called */
    Rpp32u max = 255; /* Kernel has to be called */
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "contrast.cl", "contrast_stretch", vld, vgd, "")(srcPtr, dstPtr, min, max, newMin, newMax, srcSize.height, srcSize.width, channel);
    return RPP_SUCCESS;

}



RppStatus
contrast_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;
    Rpp32u min = 0, max = 255;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "contrast.cl", "contrast_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                            min,
                                                                            max,
                                                                            handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                            handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
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
    
/********************** Blend ************************/

RppStatus
blend_cl( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr, float alpha,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "blend.cl", "blend", vld, vgd, "")(srcPtr1,
                                                                srcPtr2,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                alpha,
                                                                channel);
    return RPP_SUCCESS;
}


RppStatus
blend_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "blend.cl", "blend_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
                                                                    handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
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

/********************** pixelate ************************/

RppStatus
pixelate_cl(cl_mem srcPtr, RppiSize srcSize,cl_mem dstPtr,
            RppiChnFormat chnFormat,
            unsigned int channel,rpp::Handle& handle)
{
    unsigned short counter=0;
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width / 7) + 1,(srcSize.height / 7) + 1, channel};
        handle.AddKernel("", "", "pixelate.cl", "pixelate_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
    }
    else if(chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "pixelate.cl", "pixelate_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
    }
    return RPP_SUCCESS;
}

RppStatus
pixelate_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "pixelate.cl", "pixelate_batch", vld, vgd, "")(srcPtr, dstPtr,
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

/********************** Jitter ************************/
RppStatus
jitter_cl( cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr,
           unsigned int kernelSize,
           RppiChnFormat chnFormat, unsigned int channel,
           rpp::Handle& handle)
{
    unsigned short counter=0;
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    if(chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "jitter.cl", "jitter_pkd", vld, vgd, "")(srcPtr,
                                                                     dstPtr,
                                                                     srcSize.height,
                                                                     srcSize.width,
                                                                     channel,
                                                                     kernelSize);
       
    }
    else if(chnFormat == RPPI_CHN_PLANAR)
    {
        handle.AddKernel("", "", "jitter.cl", "jitter_pln", vld, vgd, "")(srcPtr,
                                                                     dstPtr,
                                                                     srcSize.height,
                                                                     srcSize.width,
                                                                     channel,
                                                                     kernelSize);
        
    }

    return RPP_SUCCESS;
}
RppStatus
jitter_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "jitter.cl", "jitter_batch", vld, vgd, "")(srcPtr, dstPtr,
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


//----
/********************** ADDING NOISE ************************/


RppStatus
noise_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr,
                Rpp32f noiseProbability,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    srand(time(0));
    int ctr = 0;
    clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
    if(noiseProbability != 0)
    {
        Rpp32u noisePixel = (Rpp32u)(noiseProbability * srcSize.width * srcSize.height );
        const Rpp32u pixelDistance = (srcSize.width * srcSize.height) / noisePixel;
        cl_kernel theKernel;
        cl_program theProgram;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, 1};
            handle.AddKernel("", "", "noise.cl", "snp_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        pixelDistance
                                                                        );
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, 1};
            handle.AddKernel("", "", "noise.cl", "snp_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        pixelDistance
                                                                        );
        }
    }

    return RPP_SUCCESS;
}

RppStatus
noise_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{
    Rpp32u nbatchSize = handle.GetBatchSize();
    clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, sizeof(unsigned char) *
     (handle.GetInitHandle()->mem.mcpu.srcBatchIndex[nbatchSize-1] +
     (handle.GetInitHandle()->mem.mcpu.srcSize[nbatchSize-1].width *
     handle.GetInitHandle()->mem.mcpu.srcSize[nbatchSize-1].height) * channel), 0, NULL, NULL);
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "noise.cl", "noise_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
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

/********************** Snow ************************/
RppStatus
snow_cl( cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr,
           float snowCoefficient,
           RppiChnFormat chnFormat, unsigned int channel,
           rpp::Handle& handle)
{
       
    if(snowCoefficient == 0)
    {
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
    }
    else
    {
        int ctr=0; 
        Rpp32u snowDrops= (Rpp32u)((snowCoefficient * srcSize.width * srcSize.height )/100);
        Rpp32u pixelDistance= (Rpp32u)((srcSize.width * srcSize.height) / snowDrops);
        size_t gDim3[3];
        gDim3[0] = srcSize.width;
        gDim3[1] = srcSize.height;
        gDim3[2] = 1;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
            handle.AddKernel("", "", "snow.cl", "snow_pkd", vld, vgd, "")(dstPtr,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    channel,
                                                                    pixelDistance
                                                                    );
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
            handle.AddKernel("", "", "snow.cl", "snow_pln", vld, vgd, "")(dstPtr,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    channel,
                                                                    pixelDistance
                                                                    );
           
        }

        gDim3[2] = channel;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "snow.cl", "snow", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel);
    }

    return RPP_SUCCESS;
}

RppStatus
snow_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{
    Rpp32u nbatchSize = handle.GetBatchSize();
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, max_height * max_width * channel * nbatchSize, 0, NULL, NULL);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "snow.cl", "snow_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
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


/********************** Exposure modification ************************/

RppStatus
exposure_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32f exposureValue, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "exposure.cl", "exposure", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                exposureValue
                                                                );
    return RPP_SUCCESS;
}

RppStatus
exposure_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "exposure.cl", "exposure_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
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


/********************** Rain ************************/

RppStatus
rain_cl(cl_mem srcPtr, RppiSize srcSize,cl_mem dstPtr, Rpp32f rainPercentage, Rpp32u rainWidth, Rpp32u rainHeight, Rpp32f transparency, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(rainPercentage == 0)
    {
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
    }
    else
    {
        int ctr=0;
        Rpp32u rainDrops= (Rpp32u)((rainPercentage * srcSize.width * srcSize.height )/100);
        Rpp32u pixelDistance= (Rpp32u)((srcSize.width * srcSize.height) / rainDrops);
        transparency /= 5;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width,srcSize.height,1};
            handle.AddKernel("", "", "rain.cl", "rain_pkd", vld, vgd, "")(dstPtr,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    channel,
                                                                    pixelDistance,
                                                                    rainWidth,
                                                                    rainHeight,
                                                                    transparency
                                                                    );
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width,srcSize.height,1};
            handle.AddKernel("", "", "rain.cl", "rain_pln", vld, vgd, "")(dstPtr,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    channel,
                                                                    pixelDistance,
                                                                    rainWidth,
                                                                    rainHeight,
                                                                    transparency
                                                                    );
        }

        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "rain.cl", "rain", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel
                                                                );
        
    }

    return RPP_SUCCESS;
}


RppStatus
rain_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{
    Rpp32u nbatchSize = handle.GetBatchSize();
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, max_height * max_width * channel * nbatchSize, 0, NULL, NULL);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "rain.cl", "rain_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
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

/********************** Fog ************************/

RppStatus
fog_cl( cl_mem srcPtr, RppiSize srcSize, cl_mem temp, Rpp32f fogValue, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(fogValue == 0)
    {
        clEnqueueCopyBuffer(handle.GetStream(), temp, srcPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
    }
    else
    {
       if (chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
            handle.AddKernel("", "", "fog.cl", "fog_planar", vld, vgd, "")(srcPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                fogValue
                                                                );
            
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
            handle.AddKernel("", "", "fog.cl", "fog_pkd", vld, vgd, "")(srcPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                fogValue
                                                                );
        }
        else
        {std::cerr << "Internal error: Unknown Channel format";}
    }

    return RPP_SUCCESS;
}

RppStatus
fog_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "fog.cl", "fog_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
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

/********************** Random Shadow ************************/
RppStatus
random_shadow_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u x1, Rpp32u y1,
                 Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, Rpp32u maxSizeX,
                 Rpp32u maxSizeY, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp32u row1, row2, column2, column1;
    int i, j, ctr = 0;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "random_shadow.cl", "random_shadow", vld, vgd, "")(srcPtr, dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                channel);


    for(i = 0 ; i < numberOfShadows ; i++)
    {
        ctr = 0;
        do
        {
            row1 = rand() % srcSize.height;
            column1 = rand() % srcSize.width;
        } while (column1 <= x1 || column1 >= x2 || row1 <= y1 || row1 >= y2);
        do
        {
            row2 = rand() % srcSize.height;
            column2 = rand() % srcSize.width;
        } while ((row2 < row1 || column2 < column1) || (column2 <= x1 || column2 >= x2 || row2 <= y1 || row2 >= y2) || (row2 - row1 >= maxSizeY || column2 - column1 >= maxSizeX));

        if(RPPI_CHN_PACKED == chnFormat)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{column2 - column1,row2 - row1,channel};
            handle.AddKernel("", "", "random_shadow.cl", "random_shadow_packed", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                column1,
                                                                row1,
                                                                column2,
                                                                row2
                                                                );
            
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{column2 - column1,row2 - row1,channel};
            handle.AddKernel("", "", "random_shadow.cl", "random_shadow_planar", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                column1,
                                                                row1,
                                                                column2,
                                                                row2
                                                                );
           
        }
    }
    return RPP_SUCCESS;
}

RppStatus
random_shadow_cl_batch(   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;            
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    unsigned int maxHeight, maxWidth, maxKernelSize;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    }
    cl_mem srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth * channel, NULL, NULL);
    cl_mem dstPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth * channel, NULL, NULL);


    size_t batchIndex = 0;

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        Rpp32u row1, row2, column2, column1;
        int x, y;

        clEnqueueCopyBuffer(handle.GetStream(), srcPtr, srcPtr1, batchIndex, 0, sizeof(unsigned char) * maxWidth * maxHeight * channel, 0, NULL, NULL);
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr1, dstPtr1, 0, 0, sizeof(unsigned char) * maxWidth * maxHeight * channel, 0, NULL, NULL);

        for(x = 0 ; x < handle.GetInitHandle()->mem.mcpu.uintArr[4].uintmem[i]; x++)
        {
            do
            {
                row1 = rand() % handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
                column1 = rand() % handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
            } while (column1 <= handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i] || column1 >= handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i] || row1 <= handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] || row1 >= handle.GetInitHandle()->mem.mcpu.uintArr[3].uintmem[i]);
            do
            {
                row2 = rand() % handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
                column2 = rand() % handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
            } while ((row2 < row1 || column2 < column1) || (column2 <= handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i] || column2 >= handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i]
            || row2 <= handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] || row2 >= handle.GetInitHandle()->mem.mcpu.uintArr[3].uintmem[i]) || (row2 - row1 >= handle.GetInitHandle()->mem.mcpu.uintArr[6].uintmem[i]
            || column2 - column1 >= handle.GetInitHandle()->mem.mcpu.uintArr[5].uintmem[i]));

            if(RPPI_CHN_PACKED == chnFormat)
            {
                std::vector<size_t> vld{32, 32, 1};
                std::vector<size_t> vgd{column2 - column1,row2 - row1,channel};
                handle.AddKernel("", "", "random_shadow.cl", "random_shadow_packed", vld, vgd, "")(srcPtr1,
                                                                                                   dstPtr1,
                                                                                                   handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                   handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                   channel,
                                                                                                   column1,row1,
                                                                                                   column2,row2);
               
            }
            else
            {
                std::vector<size_t> vld{32, 32, 1};
                std::vector<size_t> vgd{column2 - column1,row2 - row1,channel};
                handle.AddKernel("", "", "random_shadow.cl", "random_shadow_planar", vld, vgd, "")(srcPtr1,
                                                                                                   dstPtr1,
                                                                                                   handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                   handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                   channel,
                                                                                                   column1,row1,
                                                                                                   column2,row2);
                    
            }
           
        }
        clEnqueueCopyBuffer(handle.GetStream(), dstPtr1, dstPtr, 0, batchIndex, sizeof(unsigned char) * maxWidth * maxHeight * channel, 0, NULL, NULL);
        batchIndex += maxHeight * maxWidth * channel * sizeof(unsigned char);
    }
    return RPP_SUCCESS;
}

/********************** Histogram balance ************************/
RppStatus
histogram_balance_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    cl_context theContext;
    clGetCommandQueueInfo(  handle.GetStream(),
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(  handle.GetStream(),
                            CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    unsigned int numGroups;

    size_t lDim3[3];
    size_t gDim3[3];
    int num_pixels_per_work_item = 16;

    gDim3[0] = srcSize.width / num_pixels_per_work_item ;
    gDim3[1] = srcSize.height / num_pixels_per_work_item ;
    lDim3[0] = num_pixels_per_work_item;
    lDim3[1] = num_pixels_per_work_item;
    gDim3[2] = 1;
    lDim3[2] = 1;
    

    numGroups = gDim3[0] * gDim3[1];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    
    cl_mem partialHistogram = clCreateBuffer(theContext, CL_MEM_READ_WRITE,
                                    sizeof(unsigned int)*256*numGroups, NULL, NULL);
    cl_mem histogram = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
                                    sizeof(unsigned int)*256, NULL, NULL);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "histogram.cl", "partial_histogram_pln", vld, vgd, "")(srcPtr,
                                                                                        partialHistogram,
                                                                                        srcSize.width,
                                                                                        srcSize.height,
                                                                                        channel);
        
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};

        handle.AddKernel("", "", "histogram.cl", "partial_histogram_pkd", vld, vgd, "")(srcPtr,
                                                                                        partialHistogram,
                                                                                        srcSize.width,
                                                                                        srcSize.height,
                                                                                        channel);
       

    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

   
    //  For sum histogram kernel
    gDim3[0] = 256;
    lDim3[0] = 256;
    gDim3[1] = 1; 
    gDim3[2] = 1;
    lDim3[1] = 1;
    lDim3[2] = 1;
    std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
    std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
    handle.AddKernel("", "", "histogram.cl", "histogram_sum_partial", vld, vgd, "")(partialHistogram,
                                                                                    histogram,
                                                                                    numGroups);
   
   
    cl_mem cum_histogram = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
                                    sizeof(unsigned int)*256, NULL, NULL); 
    // For scan kernel
    gDim3[0] = 256;
    gDim3[1] = 1; 
    gDim3[2] = 1;
    lDim3[0] = 32;
    lDim3[1] = 1; 
    lDim3[2] = 1;
    std::vector<size_t> vld1{lDim3[0], lDim3[1], lDim3[2]};
    std::vector<size_t> vgd1{gDim3[0],gDim3[1],gDim3[2]};
    handle.AddKernel("", "", "scan.cl", "scan", vld1, vgd1, "")(histogram,
                                                             cum_histogram);


    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "histogram.cl", "histogram_equalize_pln", vld, vgd, "")(srcPtr,
                                                                                        dstPtr,
                                                                                        cum_histogram,
                                                                                        srcSize.width,
                                                                                        srcSize.height,
                                                                                        channel);
        
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "histogram.cl", "histogram_equalize_pkd", vld, vgd, "")(srcPtr,
                                                                                        dstPtr,
                                                                                        cum_histogram,
                                                                                        srcSize.width,
                                                                                        srcSize.height,
                                                                                        channel);
        

    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    clReleaseMemObject(cum_histogram);
    clReleaseMemObject(partialHistogram);
    clReleaseMemObject(histogram);
    //Freeing the Memory is yet to be done!
    return RPP_SUCCESS;
}

RppStatus
histogram_balance_cl_batch (cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
   
{
    Rpp32u nBatchSize = handle.GetBatchSize();
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);

    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    int numGroups = 0;
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        int size = 0;
        // size = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel;
        size = maxHeight * maxWidth * channel;
        int group = std::ceil(size / 256);
        if(numGroups < group)
            numGroups = group;
    }

    cl_mem partialHistogram = clCreateBuffer(theContext, CL_MEM_READ_WRITE,
                                    sizeof(unsigned int)*256*numGroups, NULL, NULL);
    cl_mem histogram = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
                                  sizeof(unsigned int)*256, NULL, NULL);
    cl_mem cum_histogram = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
                                    sizeof(unsigned int)*256, NULL, NULL);

    cl_mem srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char)
                     * maxHeight * maxWidth * channel, NULL, NULL);
    cl_mem dstPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char)
                     * maxHeight * maxWidth * channel, NULL, NULL);

    int ctr;
           
    size_t gDim3[3];

    size_t batchIndex = 0;
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        size_t lDim3[3];
        size_t gDim3[3];
        int num_pixels_per_work_item = 16;

        // gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] / num_pixels_per_work_item ;// Plus 1 needs to be there
        gDim3[0] = maxWidth / num_pixels_per_work_item ;// Plus 1 needs to be there
        // gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] / num_pixels_per_work_item ;
        gDim3[1] = maxHeight / num_pixels_per_work_item ;
        lDim3[0] = num_pixels_per_work_item;
        lDim3[1] = num_pixels_per_work_item;
        gDim3[2] = 1;
        lDim3[2] = 1;
        

        numGroups = gDim3[0] * gDim3[1];
        gDim3[0] = maxWidth;
        gDim3[1] = maxHeight;
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr, srcPtr1, batchIndex, 0, sizeof(unsigned char) * maxWidth * maxHeight * channel, 0, NULL, NULL);
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
            std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
            handle.AddKernel("", "", "histogram.cl", "partial_histogram_pln", vld, vgd, "")(srcPtr1,
                                                                                            partialHistogram,
                                                                                            // handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                            // handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                            maxWidth,
                                                                                            maxHeight,
                                                                                            channel);
            
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
            std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
            handle.AddKernel("", "", "histogram.cl", "partial_histogram_pkd", vld, vgd, "")(srcPtr1,
                                                                                            partialHistogram,
                                                                                            // handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                            // handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                            maxWidth,
                                                                                            maxHeight,
                                                                                            channel);
           
        }
        else
        {std::cerr << "Internal error: Unknown Channel format";}


        // For sum histogram kernel
        gDim3[0] = 256;
        lDim3[0] = 256;
        gDim3[1] = 1; 
        gDim3[2] = 1;
        lDim3[1] = 1;
        lDim3[2] = 1;
        std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "histogram.cl", "histogram_sum_partial", vld, vgd, "")(partialHistogram,
                                                                                        histogram,
                                                                                        numGroups);
        gDim3[0] = 256;
        gDim3[1] = 1; 
        gDim3[2] = 1;
        lDim3[0] = 32;
        lDim3[1] = 1; 
        lDim3[2] = 1;
        std::vector<size_t> vld1{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd1{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "scan.cl", "scan", vld1, vgd1, "")(histogram,
                                                             cum_histogram);
       

        // For histogram equalize

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            // std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],channel};
            std::vector<size_t> vgd{maxWidth,maxHeight,channel};
            handle.AddKernel("", "", "histogram.cl", "histogram_equalize_pln", vld, vgd, "")(srcPtr1,
                                                                                        dstPtr1,
                                                                                        cum_histogram,
                                                                                        // handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                        // handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                        maxWidth,
                                                                                        maxHeight,
                                                                                        channel);
           
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            // std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],channel};
            std::vector<size_t> vgd{maxWidth,maxHeight,channel};
            handle.AddKernel("", "", "histogram.cl", "histogram_equalize_pkd", vld, vgd, "")(srcPtr1,
                                                                                        dstPtr1,
                                                                                        cum_histogram,
                                                                                        // handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                        // handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                        maxWidth,
                                                                                        maxHeight,
                                                                                        channel);
            
        }
        else
        {std::cerr << "Internal error: Unknown Channel format";}
        
        // clEnqueueCopyBuffer(handle.GetStream(), dstPtr1, dstPtr, 0, batchIndex, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);
        // batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        clEnqueueCopyBuffer(handle.GetStream(), dstPtr1, dstPtr, 0, batchIndex, sizeof(unsigned char) * maxWidth * maxHeight * channel, 0, NULL, NULL);
        batchIndex += maxHeight * maxWidth * channel * sizeof(unsigned char);
    }

    clReleaseMemObject(cum_histogram);
    clReleaseMemObject(partialHistogram);
    clReleaseMemObject(histogram);
    return RPP_SUCCESS;
               
}

/********************** Occlusion ************************/
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
                rpp::Handle& handle)
{
    size_t gDim3[3];
    gDim3[0] = srcSize2.width;
    gDim3[1] = srcSize2.height;
    gDim3[2] = channel;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
     if (chnFormat == RPPI_CHN_PLANAR)
    {
        handle.AddKernel("", "", "occlusion.cl", "occlusion_pln", vld, vgd, "")(srcPtr1,
                                                                                srcPtr2,
                                                                                dstPtr,
                                                                                srcSize1.height,
                                                                                srcSize1.width,
                                                                                srcSize2.height,
                                                                                srcSize2.width,
                                                                                x11,
                                                                                y11,
                                                                                x12,
                                                                                y12,
                                                                                x21,
                                                                                y21,
                                                                                x22,
                                                                                y22,
                                                                                channel);
       
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
            handle.AddKernel("", "", "occlusion.cl", "occlusion_pkd", vld, vgd, "")(srcPtr1,
                                                                                srcPtr2,
                                                                                dstPtr,
                                                                                srcSize1.height,
                                                                                srcSize1.width,
                                                                                srcSize2.height,
                                                                                srcSize2.width,
                                                                                x11,
                                                                                y11,
                                                                                x12,
                                                                                y12,
                                                                                x21,
                                                                                y21,
                                                                                x22,
                                                                                y22,
                                                                                channel);
        
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    return RPP_SUCCESS;
}

RppStatus
occlusion_cl_batch (cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u nBatchSize = handle.GetBatchSize();
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    unsigned int src1MaxHeight, src1MaxWidth,src2MaxHeight, src2MaxWidth, dstMaxHeight, dstMaxWidth;
    src1MaxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    src1MaxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        if(src1MaxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            src1MaxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(src1MaxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            src1MaxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        if(src2MaxHeight < handle.GetInitHandle()->mem.mgpu.cdstSize.height[i])
            src2MaxHeight = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        if(src2MaxWidth < handle.GetInitHandle()->mem.mgpu.cdstSize.width[i])
            src2MaxWidth = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
    }

    cl_mem srcPtr11 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * src1MaxHeight * src2MaxWidth * channel, NULL, NULL);
    cl_mem srcPtr21 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * src2MaxHeight * src2MaxWidth * channel, NULL, NULL);
    cl_mem dstPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * src2MaxHeight * src2MaxWidth * channel, NULL, NULL);

    int ctr;
    size_t gDim3[3];
    size_t src1BatchIndex = 0, src2BatchIndex =0;
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        size_t gDim3[3];
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        gDim3[2] = channel;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr1, srcPtr11, src1BatchIndex, 0, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr2, srcPtr21, src2BatchIndex, 0, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * channel, 0, NULL, NULL);
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            handle.AddKernel("", "", "occlusion.cl", "occlusion_pln", vld, vgd, "")(srcPtr11,
                                                                                srcPtr21,
                                                                                dstPtr1,
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[3].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[4].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[5].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[6].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[7].uintmem[i],
                                                                                channel);
            
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "occlusion.cl", "occlusion_pkd", vld, vgd, "")(srcPtr11,
                                                                                srcPtr21,
                                                                                dstPtr1,
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[3].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[4].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[5].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[6].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[7].uintmem[i],
                                                                                channel);
            
        }
        else
        {std::cerr << "Internal error: Unknown Channel format";}

       

        clEnqueueCopyBuffer(handle.GetStream(), dstPtr1, dstPtr, 0, src2BatchIndex, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * channel, 0, NULL, NULL);
        src1BatchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        src2BatchIndex += handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * channel * sizeof(unsigned char);
    }
   /* Releasing of the stuff needs to be done */
    clReleaseMemObject(srcPtr11);
    clReleaseMemObject(srcPtr21);
    clReleaseMemObject(dstPtr1);
    return RPP_SUCCESS;   
}
