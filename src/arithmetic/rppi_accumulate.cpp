#include <rppdefs.h>
#include <rppi_arithmetic_and_logical_functions.h>
#include "cpu/host_accumulate.hpp"

RppStatus
rppi_accumulate_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize)
{
    host_accumulate<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
                                    1);

    return RPP_SUCCESS;

}

RppStatus
rppi_accumulate_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize)
{
    host_accumulate<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
                                    3);

    return RPP_SUCCESS;

}

RppStatus
rppi_accumulate_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize)
{
    host_accumulate<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
                                    3);

    return RPP_SUCCESS;

}