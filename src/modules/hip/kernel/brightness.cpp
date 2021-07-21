#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

__device__ unsigned char brighten(unsigned char input_pixel, float alpha, float beta)
{
    return saturate_8u(alpha * input_pixel + beta);
}

__device__ unsigned int get_pln_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, unsigned int height, unsigned channel)
{
    return (id_x + id_y * width + id_z * width * height);
}

extern "C" __global__ void brightness(unsigned char *input,
                                      unsigned char *output,
                                      const float alpha,
                                      const int beta,
                                      const unsigned int height,
                                      const unsigned int width,
                                      const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = get_pln_index(id_x, id_y, id_z, width, height, channel);
    int res = input[pixIdx] * alpha + beta;
    output[pixIdx] = saturate_8u(res);
}

extern "C" __global__ void brightness_batch(unsigned char *input,
                                            unsigned char *output,
                                            float *alpha,
                                            float *beta,
                                            unsigned int *xroi_begin,
                                            unsigned int *xroi_end,
                                            unsigned int *yroi_begin,
                                            unsigned int *yroi_end,
                                            unsigned int *height,
                                            unsigned int *width,
                                            unsigned int *max_width,
                                            unsigned long long *batch_index,
                                            const unsigned int channel,
                                            unsigned int *inc, // use width * height for pln and 1 for pkd
                                            const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float alphatmp = alpha[id_z], betatmp = beta[id_z];
    long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

    if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            unsigned char valuergb = input[pixIdx];
            output[pixIdx] = brighten(valuergb, alphatmp, betatmp);
            pixIdx += inc[id_z];
        }
    }
    else if((id_x < width[id_z]) && (id_y < height[id_z]))
    {
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[pixIdx] = input[pixIdx];
            pixIdx += inc[id_z];
        }
    }
}





















extern "C" __global__ void brightness_pkd_tensor(uchar *srcPtr,
                                                 int nStrideSrc,
                                                 int hStrideSrc,
                                                 uchar *dstPtr,
                                                 int nStrideDst,
                                                 int hStrideDst,
                                                 float *alpha,
                                                 float *beta,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3))
    {
        return;
    }

    uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

    uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
    uint2 dst;

    float4 alpha4 = (float4)alpha[id_z];
    float4 beta4 = (float4)beta[id_z];

    dst.x = rpp_hip_pack(rpp_hip_unpack(src.x) * alpha4 + beta4);
    dst.y = rpp_hip_pack(rpp_hip_unpack(src.y) * alpha4 + beta4);

    *((uint2 *)(&dstPtr[dstIdx])) = dst;



    // Code for non even multiple of 8 stride

    // if (id_x + 7 >= srcElementsInRow)
    // {
    //     int diff = srcElementsInRow - id_x;
    //     for (int x = 0; x < diff; x++)
    //     {
    //         dstPtr[dstIdx] = brighten_fmaf((float)srcPtr[srcIdx], alpha[id_z], beta[id_z]);
    //         srcIdx++;
    //         dstIdx++;
    //     }

    //     return;
    // }
}













extern "C" __global__ void brightness_pln_tensor(uchar *srcPtr,
                                                 int cStrideSrc,
                                                 int hStrideSrc,
                                                 uchar *dstPtr,
                                                 int cStrideDst,
                                                 int hStrideDst,
                                                 int channelsDst,
                                                 float *alpha,
                                                 float *beta,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;





    // Method 1

    // int id_z_corrected = id_z / channelsDst;

    // if ((id_y >= roiTensorPtrSrc[id_z_corrected].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z_corrected].xywhROI.roiWidth))
    // {
    //     return;
    // }

    // // printf("\n\nid_x, id_y, id_z | srcIdx, dstIdx - %d, %d, %d | %d, %d", id_x, id_y, id_z, srcIdx, dstIdx);

    // uint srcIdx = (id_z * cStrideSrc) + ((id_y + roiTensorPtrSrc[id_z_corrected].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z_corrected].xywhROI.xy.x);
    // uint dstIdx = (id_z * cStrideDst) + (id_y * hStrideDst) + id_x;

    // uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
    // uint2 dst;

    // float4 alpha4 = (float4)(alpha[id_z]);
    // float4 beta4 = (float4)(beta[id_z]);

    // dst.x = rpp_hip_pack(rpp_hip_unpack(src.x) * alpha4 + beta4);
    // dst.y = rpp_hip_pack(rpp_hip_unpack(src.y) * alpha4 + beta4);

    // *((uint2 *)(&dstPtr[dstIdx])) = dst;






    // Method 2 - trials with fixed pixel value

    int id_z_corrected = id_z / channelsDst;

    if ((id_y >= roiTensorPtrSrc[id_z_corrected].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z_corrected].xywhROI.roiWidth))
    {
        return;
    }

    // printf("\n\nid_x, id_y, id_z | srcIdx, dstIdx - %d, %d, %d | %d, %d", id_x, id_y, id_z, srcIdx, dstIdx);

    // NO SRC - uint srcIdx = (id_z * cStrideSrc) + ((id_y + roiTensorPtrSrc[id_z_corrected].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z_corrected].xywhROI.xy.x);
    uint dstIdx = (id_z * cStrideDst) + (id_y * hStrideDst) + id_x;

    // NO SRC - uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
    uint2 dst;
    float pix = (float)((id_z+1) * 18); // id_z is 0 to 6

    // type A - with typecast

    float4 alpha4 = (float4)(alpha[id_z]);
    float4 beta4 = (float4)(beta[id_z]);
    float4 pix4 = (float4)(pix);

    dst.x = rpp_hip_pack(pix4 * alpha4 + beta4);
    dst.y = rpp_hip_pack(pix4 * alpha4 + beta4);

    // type B - with make_float4

    // float4 alpha4 = make_float4(alpha[id_z], alpha[id_z], alpha[id_z], alpha[id_z]);
    // float4 beta4 = make_float4(beta[id_z], beta[id_z], beta[id_z], beta[id_z]);
    // float4 pix4 = make_float4((id_z+1) * 18, (id_z+1) * 18, (id_z+1) * 18, (id_z+1) * 18);

    // dst.x = rpp_hip_pack(pix4 * alpha4 + beta4);
    // dst.y = rpp_hip_pack(pix4 * alpha4 + beta4);

    *((uint2 *)(&dstPtr[dstIdx])) = dst;








    // Method 3 - trials with fmaf

    // int id_z_corrected = id_z / channelsDst;

    // if ((id_y >= roiTensorPtrSrc[id_z_corrected].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z_corrected].xywhROI.roiWidth))
    // {
    //     return;
    // }

    // // printf("\n\nid_x, id_y, id_z | srcIdx, dstIdx - %d, %d, %d | %d, %d", id_x, id_y, id_z, srcIdx, dstIdx);

    // uint srcIdx = (id_z * cStrideSrc) + ((id_y + roiTensorPtrSrc[id_z_corrected].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z_corrected].xywhROI.xy.x);
    // uint dstIdx = (id_z * cStrideDst) + (id_y * hStrideDst) + id_x;

    // uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
    // uint2 dst;

    // float pix = (float)((id_z+1) * 18); // id_z is 0 to 6

    // float4 dstF4x, dstF4y;
    // dstF4x.x = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4x.y = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4x.z = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4x.w = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4y.x = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4y.y = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4y.z = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4y.w = fmaf(alpha[id_z], pix, beta[id_z]);

    // dst.x = rpp_hip_pack(dstF4x);
    // dst.y = rpp_hip_pack(dstF4y);

    // *((uint2 *)(&dstPtr[dstIdx])) = dst;


















    // OTHER TRIALS

    // if ((id_y >= roiTensorPtrSrc[id_z_corrected].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z_corrected].xywhROI.roiWidth))
    // {
    //     return;
    // }

    // uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z_corrected].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z_corrected].xywhROI.xy.x);
    // uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

    // uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
    // uint2 dst;

    // float pix = (float)((id_z+1) * 18);

    // float4 alpha4 = (float4)(alpha[id_z]);
    // float4 beta4 = (float4)(beta[id_z]);
    // float4 pix4 = (float4)(pix);

    // float4 alpha4 = make_float4(alpha[id_z], alpha[id_z], alpha[id_z], alpha[id_z]);
    // float4 beta4 = make_float4(beta[id_z], beta[id_z], beta[id_z], beta[id_z]);
    // float4 pix4 = make_float4((id_z+1) * 18, (id_z+1) * 18, (id_z+1) * 18, (id_z+1) * 18);

    // dst.x = rpp_hip_pack(rpp_hip_unpack(src.x) * alpha4 + beta4);
    // dst.y = rpp_hip_pack(rpp_hip_unpack(src.y) * alpha4 + beta4);

    // float4 dstF4x, dstF4y;
    // dstF4x.x = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4x.y = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4x.z = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4x.w = fmaf(alpha[id_z], pix, beta[id_z]);

    // dstF4y.x = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4y.y = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4y.z = fmaf(alpha[id_z], pix, beta[id_z]);
    // dstF4y.w = fmaf(alpha[id_z], pix, beta[id_z]);

    // dst.x = rpp_hip_pack(pix4 * alpha4 + beta4);
    // dst.y = rpp_hip_pack(pix4 * alpha4 + beta4);

    // dst.x = rpp_hip_pack(dstF4x);
    // dst.y = rpp_hip_pack(dstF4y);

    // *((uint2 *)(&dstPtr[dstIdx])) = dst;
    // }










    // Code for non even multiple of 8 stride

    // if (id_x + 7 >= srcElementsInRow)
    // {
    //     int diff = srcElementsInRow - id_x;
    //     for (int x = 0; x < diff; x++)
    //     {
    //         dstPtr[dstIdx] = brighten_fmaf((float)srcPtr[srcIdx], alpha[id_z], beta[id_z]);
    //         srcIdx++;
    //         dstIdx++;
    //     }

    //     return;
    // }
}

RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(brightness_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
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
                       plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_brightness_tensor(Rpp8u *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8u *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle)
{
    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        int localThreads_x = 16;
        int localThreads_y = 16;
        int localThreads_z = 1;
        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();

        hipLaunchKernelGGL(brightness_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcDescPtr->strides.nStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           dstDescPtr->strides.hStride,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           roiTensorPtrSrc,
                           roiType);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        int localThreads_x = 16;
        int localThreads_y = 16;
        int localThreads_z = 1;
        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize() * dstDescPtr->c;

        std::cerr << "\n\n globalThreads_x - " << dstDescPtr->strides.hStride << " " << globalThreads_x;
        std::cerr << "\n\n globalThreads_y - " << dstDescPtr->h << " " << globalThreads_y;
        std::cerr << "\n\n globalThreads_z - " << dstDescPtr->c << " " << globalThreads_z;
        std::cerr << "\n\n";
        std::cerr << "\n\n srcDescPtr->strides.cStride - " << srcDescPtr->strides.cStride;
        std::cerr << "\n\n srcDescPtr->strides.hStride - " << srcDescPtr->strides.hStride;
        std::cerr << "\n\n dstDescPtr->strides.cStride - " << dstDescPtr->strides.cStride;
        std::cerr << "\n\n dstDescPtr->strides.hStride - " << dstDescPtr->strides.hStride;

        hipLaunchKernelGGL(brightness_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcDescPtr->strides.cStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.cStride,
                           dstDescPtr->strides.hStride,
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           roiTensorPtrSrc,
                           roiType);

        // Checking outputs

        Rpp8u *output = (Rpp8u *)calloc(dstDescPtr->strides.nStride * handle.GetBatchSize(), sizeof(Rpp8u));
        RpptROI *roi = (RpptROI *)calloc(handle.GetBatchSize(), sizeof(RpptROI));
        hipMemcpy(output, dstPtr, dstDescPtr->strides.nStride * handle.GetBatchSize() * sizeof(Rpp8u), hipMemcpyDeviceToHost);
        hipMemcpy(roi, roiTensorPtrSrc, handle.GetBatchSize() * sizeof(RpptROI), hipMemcpyDeviceToHost);

        Rpp8u *outputTemp;
        outputTemp = output;

        printf("\n\n\nPrinting images:\n");
        for (int n = 0; n < dstDescPtr->n; n++)
        {
            printf("\n\n\nPrinting ROI for image%d:\n", n + 1);
            printf("%d, %d, %d, %d", roi[n].xywhROI.xy.x, roi[n].xywhROI.xy.y, roi[n].xywhROI.roiWidth, roi[n].xywhROI.roiHeight);

            printf("\n\n\nPrinting image %d:\n", n + 1);
            for (int c = 0; c < dstDescPtr->c; c++)
            {
                for (int h = 0; h < dstDescPtr->h; h++)
                {
                    for (int w = 0; w < dstDescPtr->w; w++)
                    {
                        printf("%d ", *outputTemp);
                        outputTemp++;
                    }
                    printf("\n");
                }
                printf("\n\n");
            }
        }
    }

    return RPP_SUCCESS;
}