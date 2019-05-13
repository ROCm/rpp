#ifndef RPPI_ADAS_FUNCTIONS_H
#define RPPI_ADAS_FUNCTIONS_H
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

RppStatus
Rppi_Foggy_8u_pkd1(const RppPtr_t  srcPtr, int rSrcStep,
                        RppPtr_t  dstPtr, int rDstStep,
                  RppiFuzzyLevel rlevel);

RppStatus
Rppi_Foggy_8u_pkd3(const RppPtr_t  srcPtr, int rSrcStep,
                        RppPtr_t  dstPtr, int rDstStep,
                  RppiFuzzyLevel rlevel);

RppStatus
Rppi_Foggy_8u_pln1(const RppPtr_t  srcPtr, int rSrcStep,
                        RppPtr_t  dstPtr, int rDstStep,
                  RppiFuzzyLevel rlevel);

RppStatus
Rppi_Foggy_8u_pln3(const RppPtr_t  srcPtr, int rSrcStep,
                        RppPtr_t  dstPtr, int rDstStep,
                  RppiFuzzyLevel rlevel);
#ifdef __cplusplus
}
#endif
#endif /* RPPI_ADAS_FUNCTIONS_H */