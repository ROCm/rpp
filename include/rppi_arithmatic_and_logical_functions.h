#ifndef RPPI_ARITHMATIC_AND_LOGICAL_FUNCTIONS_H
#define RPPI_ARITHMATIC_AND_LOGICAL_FUNCTIONS_H
#include "rppdefs.h"

#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


RppStatus
Rppi_Add_Constant_8u_pkd1(const Rpp8u * pSrc, int rSrcStep,
                        Rpp8u * pDst, int rDstStep,
                  const Rpp8u rConstant, RppiSize oSizeROI, int rScaleFactor);

RppStatus
Rppi_Add_Constant_8u_pln1(const Rpp8u * pSrc, int rSrcStep,
                        Rpp8u * pDst, int rDstStep,
                  const Rpp8u rConstant, RppiSize oSizeROI, int rScaleFactor);

RppStatus
Rppi_Add_Constant_8u_pkd3(const Rpp8u * pSrc, int rSrcStep,
                        Rpp8u * pDst, int rDstStep,
                  const Rpp8u rConstant, RppiSize oSizeROI, int rScaleFactor);

RppStatus
Rppi_Add_Constant_8u_pln3(const Rpp8u * pSrc, int rSrcStep,
                        Rpp8u * pDst, int rDstStep,
                  const Rpp8u rConstant, RppiSize oSizeROI, int rScaleFactor);
#ifdef __cplusplus
}
#endif
#endif /* RPPI_ARITHMATIC_AND_LOGICAL_FUNCTIONS_H */