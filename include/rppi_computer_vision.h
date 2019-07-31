#ifndef RPPI_COMPUTER_VISION
#define RPPI_COMPUTER_VISION
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host local_binary_pattern functions declaration 
// ----------------------------------------

RppStatus
rppi_local_binary_pattern_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_local_binary_pattern_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_local_binary_pattern_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);
 
#ifdef __cplusplus
}
#endif
#endif
