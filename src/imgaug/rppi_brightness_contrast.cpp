#include "hip/rpp_hip_comman.hpp"

RppStatus
rppi_brighten_3C8U_pln(Rpp8u *pSrc, RppiSize size, Rpp8u *pDst, Rpp32f alpha, Rpp32f beta) {


#if defined HIP_COMPILE
    brightness_contrast_caller<Rpp8u>(  Rpp8u *pSrc, RppiSize size, Rpp8u *pDst,
                                        Rpp32f alpha, Rpp32f beta,
                                        RppiStorageFormat RPPI_CHN_PLANAR )
#endif

}