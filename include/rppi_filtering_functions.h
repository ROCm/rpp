#ifndef RPPI_FILTERING_FUNCTIONS_H
#define RPPI_FILTERING_FUNCTIONS_H
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif
RppStatus
Rppi_median_8u_pkd1_host(const Rpp8u * pSrc, int rSrcStep,Rpp8u * pDst, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_median_8u_pkd1(const Rpp8u * pSrc, int rSrcStep,Rpp8u * pDst, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_gaussian_8u_pkd1_host(const Rpp8u * pSrc, Rpp8u * pDst, RppiSize oROI, RppiAxis flip);
#ifdef __cplusplus
}
#endif
#endif /* RPPI_FILTERING_FUNCTIONS_H */