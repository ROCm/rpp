#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_brightness_contrast.hpp"
#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>
#include "hip/hip_brightness_contrast.hpp"
#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#endif //backend

RppStatus
rppi_brighten_1C8U_pln( RppHandle_t rppHandle, RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta)
{

#ifdef HIP_COMPILE
    hip_brightness_contrast<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                   static_cast<Rpp8u*>(dstPtr),
                                   alpha, beta, RPPI_CHN_PLANAR );

#elif defined (OCL_COMPILE)
    cl_kernel theKernel;
    cl_program theProgram;
    cl_kernel_initializer(rppHandle, "brightness_contrast", theProgram, theKernel); // .cl will get be added internally

    //---- Args Setter
    unsigned int n = srcSize.height * srcSize.width * 1 /*channel*/ ;
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 2, sizeof(unsigned int), &n);
    clSetKernelArg(theKernel, 3, sizeof(Rpp32f), &alpha);
    clSetKernelArg(theKernel, 3, sizeof(Rpp32s), &beta);
    //----

    cl_kernel_implementer (rppHandle, srcSize, theProgram, theKernel);

#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_brighten_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha = 1, Rpp32f beta = 0)
{

    host_brightness_contrast<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), alpha, beta, RPPI_CHN_PLANAR );

    return RPP_SUCCESS;

}
