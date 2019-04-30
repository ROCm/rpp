#include "hip/rpp_hip_comman.hpp"

RppStatus
rppi_brighten_1C8U_pln(Rpp8u *pSrc, RppiSize size, Rpp8u *pDst, Rpp32f alpha, Rpp32f beta) {


#ifdef HIP_COMPILE
    brightness_contrast_caller<Rpp8u>(  pSrc, size, pDst,
                                        alpha, beta, RPPI_CHN_PLANAR );
#elif defined (OCL_COMPILE)
    cl_brightness_contrast( pSrc, size.height, size.width, pDst);
#endif

}
