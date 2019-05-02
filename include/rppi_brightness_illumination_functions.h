#ifndef RPPI_BRIGHTNESS_ILLUMINATION_FUNCTIONS_H
#define RPPI_BRIGHTNESS_ILLUMINATION_FUNCTIONS_H
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

RppStatus
rppi_brighten_1C8U_pln(Rpp8u *pSrc, RppiSize size, Rpp8u *pDst, Rpp32f alpha, Rpp32f beta);

RppStatus
rppi_brighten_1C8U_pln_host(Rpp8u *pSrc, RppiSize size, Rpp8u *pDst, Rpp32f alpha, Rpp32f beta);

#ifdef __cplusplus
}
#endif

#endif /* RPPI_BRIGHTNESS_ILLUMINATION_FUNCTIONS_H */