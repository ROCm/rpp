#ifndef RPPI_FILTERING_FUNCTIONS_H
#define RPPI_FILTERING_FUNCTIONS_H
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif
RppStatus
Rppi_median_8u_pkd1_host(const RppPtr_t  srcPtr, int rSrcStep,RppPtr_t  dstPtr, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_median_8u_pkd1(const RppPtr_t  srcPtr, int rSrcStep,RppPtr_t  dstPtr, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_gaussian_8u_pkd1_host(const RppPtr_t  srcPtr, RppPtr_t  dstPtr, RppiSize oROI, RppiAxis flip);
#ifdef __cplusplus
}
#endif
#endif /* RPPI_FILTERING_FUNCTIONS_H */