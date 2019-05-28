#include <rppdefs.h>
#include <rppi_arithmatic_and_logical_functions.h>

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>
#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend

RppStatus
rppi_subtract_u8_pln1_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{

#ifdef HIP_COMPILE

// Yet  to be implemented

#elif defined (OCL_COMPILE)

    cl_subtract ( static_cast<cl_mem>(srcPtr1),
             static_cast<cl_mem>(srcPtr2),
             srcSize,
             static_cast<cl_mem>(dstPtr),
             RPPI_CHN_PLANAR, 1 /*Channel*/,
             static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}


RppStatus
rppi_subtract_u8_pln3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{


#ifdef HIP_COMPILE

// Yet to be implemented

#elif defined (OCL_COMPILE)
    cl_subtract ( static_cast<cl_mem>(srcPtr1),
             static_cast<cl_mem>(srcPtr2),
             srcSize,
             static_cast<cl_mem>(dstPtr),
             RPPI_CHN_PLANAR, 3 /*Channel*/,
             static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}



RppStatus
rppi_subtract_u8_pkd3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{


#ifdef OCL_COMPILE

    cl_subtract ( static_cast<cl_mem>(srcPtr1),
             static_cast<cl_mem>(srcPtr2),
             srcSize,
             static_cast<cl_mem>(dstPtr),
             RPPI_CHN_PACKED, 3 /*Channel*/,
             static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}
