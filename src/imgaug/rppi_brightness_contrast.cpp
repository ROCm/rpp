#include <rppdefs.h>
#include <rppi_brightness_illumination_functions.h>
#include "cpu/host_brightness_contrast.hpp"

#ifdef HIP_COMPILE
#include "hip/hip_brightness_contrast.hpp"
#endif //backend

//--------------------------- CL declaration -------------------------------------

RppStatus cl_brightness_contrast( Rpp8u* pSrc, unsigned int height, unsigned int width,
                                  Rpp8u* pDst, Rpp32f alpha, Rpp32f beta);

//------------------------------------------------------------------------------

RppStatus
rppi_brighten_1C8U_pln(Rpp8u *pSrc, RppiSize size, Rpp8u *pDst, Rpp32f alpha = 1, Rpp32f beta = 0)
{

#ifdef HIP_COMPILE
    // temporay part
    void* devPtrS; void* devPtrD;
    hipMalloc((void**)&devPtrS, sizeof(Rpp8u)*size.height*size.width );
    hipMalloc((void**)&devPtrD, sizeof(Rpp8u)*size.height*size.width );
    hipMemcpy(devPtrS, pSrc, sizeof(Rpp8u)*size.height*size.width , hipMemcpyHostToDevice);
    hip_brightness_contrast<Rpp8u>((Rpp8u*)devPtrS, size, (Rpp8u*)devPtrD, alpha, beta, RPPI_CHN_PLANAR );
    hipMemcpy(devPtrD, pDst, sizeof(Rpp8u)*size.height*size.width , hipMemcpyHostToDevice);
    hipFree(devPtrS);
    hipFree(devPtrD);

#elif defined (OCL_COMPILE)
    cl_brightness_contrast( pSrc, size.height, size.width, pDst, alpha, beta);
#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_brighten_1C8U_pln_host(Rpp8u *pSrc, RppiSize size, Rpp8u *pDst, Rpp32f alpha = 1, Rpp32f beta = 0)
{

    host_brightness_contrast<Rpp8u>(pSrc, size, pDst, alpha, beta, RPPI_CHN_PLANAR );

#ifdef CV_COMPILE
    printf("OpenCV base not implemented");
    return RPP_ERROR;
#endif

    return RPP_SUCCESS;

}
