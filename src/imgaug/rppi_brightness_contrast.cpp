#include <rppdefs.h>

RppStatus
rppi_brighten_1C8U_pln(Rpp8u *pSrc, RppiSize size, Rpp8u *pDst, Rpp32f alpha, Rpp32f beta) {


#ifdef BACKEND_HIP
    brightness_contrast_caller<Rpp8u>(  pSrc, size, pDst,
                                        alpha, beta, RPPI_CHN_PLANAR );
#elif defined (BACKEND_OCL)
    cl_brightness_contrast( pSrc, size.height, size.width, pDst);
#endif // BACKEND

}
