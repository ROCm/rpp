#ifndef RPPI_GEOMETRY_TRANSFORMS.H
#define RPPI_GEOMETRY_TRANSFORMS.H
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host flip functions  declaration 
// ----------------------------------------


RppStatus
rppi_flip_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis);

RppStatus
rppi_flip_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis);

RppStatus
rppi_flip_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis);

// ----------------------------------------
// Host resize functions  declaration 
// ----------------------------------------


RppStatus
rppi_resize_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize);

RppStatus
rppi_resize_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize);

RppStatus
rppi_resize_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize);

// ----------------------------------------
// Host resize_crop functions  declaration 
// ----------------------------------------


RppStatus
rppi_resize_crop_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2);

RppStatus
rppi_resize_crop_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2);

RppStatus
rppi_resize_crop_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2);

// ----------------------------------------
// Host rotate functions  declaration 
// ----------------------------------------


RppStatus
rppi_rotate_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg);

RppStatus
rppi_rotate_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg);

RppStatus
rppi_rotate_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg);

// ----------------------------------------
// GPU flip functions  declaration 
// ----------------------------------------


RppStatus
rppi_flip_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis, RppHandle_t rppHandle) ;

RppStatus
rppi_flip_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis, RppHandle_t rppHandle) ;

RppStatus
rppi_flip_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU resize functions  declaration 
// ----------------------------------------


RppStatus
rppi_resize_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize, RppHandle_t rppHandle) ;

RppStatus
rppi_resize_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize, RppHandle_t rppHandle) ;

RppStatus
rppi_resize_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU resize_crop functions  declaration 
// ----------------------------------------


RppStatus
rppi_resize_crop_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;

RppStatus
rppi_resize_crop_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;

RppStatus
rppi_resize_crop_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU rotate functions  declaration 
// ----------------------------------------


RppStatus
rppi_rotate_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg, RppHandle_t rppHandle) ;

RppStatus
rppi_rotate_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg, RppHandle_t rppHandle) ;

RppStatus
rppi_rotate_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg, RppHandle_t rppHandle) ;
 
#ifdef __cplusplus
}
#endif
#endif
