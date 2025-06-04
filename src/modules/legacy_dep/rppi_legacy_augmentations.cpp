/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "rppi_validate.hpp"
#include "rppi_legacy_augmentations.h"
#include "host_legacy_executors.hpp"

#ifdef HIP_COMPILE
#include "hip_legacy_executors.hpp"
#endif


/******************** fisheye ********************/

RppStatus rppi_fisheye_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            Rpp32u nbatchSize,
                                            rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    fisheye_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                              srcSize,
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                              static_cast<Rpp8u *>(dstPtr),
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                              rpp::deref(rppHandle).GetBatchSize(),
                              RPPI_CHN_PLANAR,
                              1,
                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus rppi_fisheye_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            Rpp32u nbatchSize,
                                            rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    fisheye_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                              srcSize,
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                              static_cast<Rpp8u *>(dstPtr),
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                              rpp::deref(rppHandle).GetBatchSize(),
                              RPPI_CHN_PACKED,
                              3,
                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** snow ********************/

RppStatus rppi_snow_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *snowValue,
                                         Rpp32u nbatchSize,
                                         rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    snow_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           snowValue,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PLANAR,
                           1,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus rppi_snow_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *snowValue,
                                         Rpp32u nbatchSize,
                                         rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    snow_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           snowValue,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PACKED,
                           3,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus rppi_hueRGB_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32f *hueShift,
                                           Rpp32u nbatchSize,
                                           rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    hueRGB_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                             srcSize,
                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                             static_cast<Rpp8u*>(dstPtr),
                             hueShift,
                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                             rpp::deref(rppHandle).GetBatchSize(),
                             RPPI_CHN_PACKED,
                             3,
                             rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus rppi_saturationRGB_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                                  RppiSize *srcSize,
                                                  RppiSize maxSrcSize,
                                                  RppPtr_t dstPtr,
                                                  Rpp32f *saturationFactor,
                                                  Rpp32u nbatchSize,
                                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    saturationRGB_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    saturationFactor,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PACKED,
                                    3,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** fisheye ********************/
RppStatus rppi_fisheye_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32u nbatchSize,
                                           rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

    fisheye_hip_batch(static_cast<Rpp8u *>(srcPtr),
                        static_cast<Rpp8u *>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        1);

    return RPP_SUCCESS;
}


RppStatus rppi_fisheye_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32u nbatchSize,
                                           rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

    fisheye_hip_batch(static_cast<Rpp8u *>(srcPtr),
                        static_cast<Rpp8u *>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PACKED,
                        3);


    return RPP_SUCCESS;
}

RppStatus rppi_snow_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp32f *snowValue,
                                        Rpp32u nbatchSize,
                                        rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_float(snowValue, rpp::deref(rppHandle), paramIndex++);

    snow_hip_batch(static_cast<Rpp8u*>(srcPtr),
                    static_cast<Rpp8u*>(dstPtr),
                    rpp::deref(rppHandle),
                    RPPI_CHN_PLANAR,
                    1);

    return RPP_SUCCESS;
}

RppStatus rppi_snow_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp32f *snowValue,
                                        Rpp32u nbatchSize,
                                        rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(snowValue, rpp::deref(rppHandle), paramIndex++);

    snow_hip_batch(static_cast<Rpp8u*>(srcPtr),
                    static_cast<Rpp8u*>(dstPtr),
                    rpp::deref(rppHandle),
                    RPPI_CHN_PACKED,
                    3);

    return RPP_SUCCESS;
}

RppStatus rppi_hueRGB_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *hueShift,
                                          Rpp32u nbatchSize,
                                          rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);

    hueRGB_hip_batch(static_cast<Rpp8u*>(srcPtr),
                        static_cast<Rpp8u*>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PACKED,
                        3);

    return RPP_SUCCESS;
}

RppStatus rppi_saturationRGB_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                                 RppiSize *srcSize,
                                                 RppiSize maxSrcSize,
                                                 RppPtr_t dstPtr,
                                                 Rpp32f *saturationFactor,
                                                 Rpp32u nbatchSize,
                                                 rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

    saturationRGB_hip_batch(static_cast<Rpp8u*>(srcPtr),
                            static_cast<Rpp8u*>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PACKED,
                            3);

    return RPP_SUCCESS;
}
#endif