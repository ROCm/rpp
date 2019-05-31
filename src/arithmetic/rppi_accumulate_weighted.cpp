#include <rppdefs.h>
#include <rppi_arithmetic_and_logical_functions.h>
#include "cpu/host_accumulate_weighted.hpp"

RppStatus
rppi_accumulate_weighted_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, Rpp32f alpha)
{
    host_accumulate_weighted<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
                                    alpha,
                                    1);
    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_weighted_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, Rpp32f alpha)
{
    host_accumulate_weighted<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
                                    alpha,
                                    3);
    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_weighted_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, Rpp32f alpha)
{
    host_accumulate_weighted<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
                                    alpha,
                                    3);
    return RPP_SUCCESS;
}
