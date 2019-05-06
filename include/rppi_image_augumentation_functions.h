#ifndef RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H
#define RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

RppStatus
rppi_brighten_1C8U_pln(Rpp8u *srcPtr, RppiSize size, Rpp8u *dstPtr, Rpp32f alpha, Rpp32f beta);

RppStatus
rppi_brighten_1C8U_pln_host(Rpp8u *srcPtr, RppiSize size, Rpp8u *dstPtr, Rpp32f alpha, Rpp32f beta);

#ifdef __cplusplus
}
#endif

#endif /* RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H */