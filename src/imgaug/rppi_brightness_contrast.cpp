#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>
#include "cpu/host_brightness_contrast.hpp"

#ifdef HIP_COMPILE
#include "hip/hip_brightness_contrast.hpp"
#endif //backend

//--------------------------- CL declaration -----------------------------------

RppStatus cl_brightness_contrast( Rpp8u* pSrc, unsigned int height, unsigned int width,
                                  Rpp8u* pDst, Rpp32f alpha, Rpp32f beta);

//------------------------------------------------------------------------------

RppStatus
rppi_brighten_1C8U_pln(Rpp8u *pSrc, RppiSize size, Rpp8u *pDst, Rpp32f alpha = 1, Rpp32f beta = 0)
{

#ifdef HIP_COMPILE
    hip_brightness_contrast<Rpp8u>(pSrc, size, pDst, alpha, beta, RPPI_CHN_PLANAR );
#elif defined (OCL_COMPILE)
     cl_brightness_contrast( pSrc, size.height, size.width, pDst, alpha, beta);
#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_brighten_1C8U_pln_host(Rpp8u *pSrc, RppiSize size, Rpp8u *pDst, Rpp32f alpha = 1, Rpp32f beta = 0)
{

    host_brightness_contrast<Rpp8u>(pSrc, size, pDst, alpha, beta, RPPI_CHN_PLANAR );

    return RPP_SUCCESS;

}
