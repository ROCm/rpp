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
rppi_brighten_1C8U_pln( RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f alpha = 1, Rpp32f beta = 0)
{

#ifdef HIP_COMPILE
    hip_brightness_contrast<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                   static_cast<Rpp8u*>(dstPtr),
                                   alpha, beta, RPPI_CHN_PLANAR );

#elif defined (OCL_COMPILE)
    cl_kernel theKernel;
    cl_kernel_initializer("brightness_contrast",theKernel); // .cl will get be added internally

    //---- Args Setter

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

    //---- kernel caller
    cl::Event event;
    size_t global_item_size = srcSize.height * srcSize.width;
    size_t local_item_size = 64;
    clEnqueueNDRangeKernel( command_queue, theKernel, 1, NULL, &global_item_size,
                            &local_item_size, 0, NULL, event );
    event.wait();

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
