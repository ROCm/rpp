#include <rppdefs.h>
#include <rppi_arithmetic_and_logical_functions.h>

#include "cpu/host_logical_operations.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>
#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend




/******* Bitwise AND ********/

// GPU calls for Bitwise AND function

RppStatus
rppi_bitwise_AND_u8_pln1_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{

#ifdef HIP_COMPILE

// Yet  to be implemented

#elif defined (OCL_COMPILE)

    bitwise_AND_cl ( static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     srcSize,
                     static_cast<cl_mem>(dstPtr),
                     RPPI_CHN_PLANAR, 1 /*Channel*/,
                     static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_AND_u8_pln3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{


#ifdef HIP_COMPILE

// Yet to be implemented

#elif defined (OCL_COMPILE)
    bitwise_AND_cl ( static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     srcSize,
                     static_cast<cl_mem>(dstPtr),
                     RPPI_CHN_PLANAR, 3 /*Channel*/,
                     static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_AND_u8_pkd3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{


#ifdef OCL_COMPILE

    bitwise_AND_cl ( static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     srcSize,
                     static_cast<cl_mem>(dstPtr),
                     RPPI_CHN_PACKED, 3 /*Channel*/,
                     static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

// Host calls for Bitwise AND function

RppStatus
rppi_and_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    host_and<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    1);

    return RPP_SUCCESS;

}

RppStatus
rppi_and_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    host_and<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    3);

    return RPP_SUCCESS;

}

RppStatus
rppi_and_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    host_and<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    3);

    return RPP_SUCCESS;

}




/******* Bitwise NOT ********/

// GPU calls for Bitwise NOT function

RppStatus
rppi_bitwise_NOT_u8_pln1_gpu( RppPtr_t srcPtr1,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{

#ifdef HIP_COMPILE

// Yet  to be implemented

#elif defined (OCL_COMPILE)

    bitwise_NOT_cl ( static_cast<cl_mem>(srcPtr1),
                     srcSize,
                     static_cast<cl_mem>(dstPtr),
                     RPPI_CHN_PLANAR, 1 /*Channel*/,
                     static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_NOT_u8_pln3_gpu( RppPtr_t srcPtr1,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{


#ifdef HIP_COMPILE

// Yet to be implemented

#elif defined (OCL_COMPILE)
    bitwise_NOT_cl ( static_cast<cl_mem>(srcPtr1),
                     srcSize,
                     static_cast<cl_mem>(dstPtr),
                     RPPI_CHN_PLANAR, 3 /*Channel*/,
                     static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_NOT_u8_pkd3_gpu( RppPtr_t srcPtr1,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{


#ifdef OCL_COMPILE

    bitwise_NOT_cl ( static_cast<cl_mem>(srcPtr1),
                     srcSize,
                     static_cast<cl_mem>(dstPtr),
                     RPPI_CHN_PACKED, 3 /*Channel*/,
                     static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

// Host calls for Bitwise NOT function

RppStatus
rppi_not_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    host_not<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    1);

    return RPP_SUCCESS;

}

RppStatus
rppi_not_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    host_not<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    3);

    return RPP_SUCCESS;

}

RppStatus
rppi_not_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    host_not<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    3);

    return RPP_SUCCESS;

}





/******* EXCLUSIVE OR ********/

// GPU calls for EXCLUSIVE OR function

RppStatus
rppi_exclusive_OR_u8_pln1_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{

#ifdef HIP_COMPILE

// Yet  to be implemented

#elif defined (OCL_COMPILE)

    exclusive_OR_cl ( static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     srcSize,
                     static_cast<cl_mem>(dstPtr),
                     RPPI_CHN_PLANAR, 1 /*Channel*/,
                     static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_exclusive_OR_u8_pln3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{


#ifdef HIP_COMPILE

// Yet to be implemented

#elif defined (OCL_COMPILE)
    exclusive_OR_cl ( static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     srcSize,
                     static_cast<cl_mem>(dstPtr),
                     RPPI_CHN_PLANAR, 3 /*Channel*/,
                     static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_exclusive_OR_u8_pkd3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{


#ifdef OCL_COMPILE

    exclusive_OR_cl ( static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     srcSize,
                     static_cast<cl_mem>(dstPtr),
                     RPPI_CHN_PACKED, 3 /*Channel*/,
                     static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

// Host calls for EXCLUSIVE OR function

RppStatus
rppi_xor_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    host_xor<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    1);

    return RPP_SUCCESS;

}

RppStatus
rppi_xor_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    host_xor<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    3);

    return RPP_SUCCESS;

}

RppStatus
rppi_xor_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    host_xor<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    3);

    return RPP_SUCCESS;

}




/******* INCLUSIVE OR ********/

// GPU calls for INCLUSIVE OR function

RppStatus
rppi_inclusive_OR_u8_pln1_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{

#ifdef HIP_COMPILE

// Yet  to be implemented

#elif defined (OCL_COMPILE)

    inclusive_OR_cl ( static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     srcSize,
                     static_cast<cl_mem>(dstPtr),
                     RPPI_CHN_PLANAR, 1 /*Channel*/,
                     static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_inclusive_OR_u8_pln3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{


#ifdef HIP_COMPILE

// Yet to be implemented

#elif defined (OCL_COMPILE)
    inclusive_OR_cl ( static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     srcSize,
                     static_cast<cl_mem>(dstPtr),
                     RPPI_CHN_PLANAR, 3 /*Channel*/,
                     static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_inclusive_OR_u8_pkd3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle )
{


#ifdef OCL_COMPILE

    inclusive_OR_cl ( static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     srcSize,
                     static_cast<cl_mem>(dstPtr),
                     RPPI_CHN_PACKED, 3 /*Channel*/,
                     static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

// Host calls for INCLUSIVE OR function

RppStatus
rppi_or_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    host_or<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    1);

    return RPP_SUCCESS;

}

RppStatus
rppi_or_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    host_or<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    3);

    return RPP_SUCCESS;

}

RppStatus
rppi_or_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    host_or<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    3);

    return RPP_SUCCESS;

}