#include <rppdefs.h>
#include <math.h>
#include <iostream>
#include "rppi_support_functions.h"
#define RAD(deg) (deg * PI / 180)
#define PI 3.14159265

/* All the rpp utilities functions comes here 
   Like Getting output sizes of Rotate and Warp Affine  etc 
   The coordinate offset for the same functions*/

/*Get Offset Function for Rotate*/
/*Get Offset Function for Warp Affine */
/*Get Output Size Function for Rotate */
/*Get Output Size Funtion for Warp Affine */

RppStatus warp_affine_output_size(RppiSize srcSize, RppiSize *dstSizePtr,
                                  float *affine)
{

    float minX = 0, minY = 0, maxX = 0, maxY = 0;
    
    RppiPoint corner[4];

    corner[0].x = (affine[0] * 0) + (affine[1] * 0  + affine[2]);
    corner[0].y = (affine[3] * 0) + (affine[4] * 0  + affine[5]);
    corner[1].x = (affine[0] * 0) + affine[1] * (srcSize.width-1) + affine[2];
    corner[1].y = (affine[3] * 0) + affine[4] * (srcSize.width-1) + affine[5];
    corner[2].x = (affine[0] * (srcSize.height-1)) + (affine[1] * 0) + affine[2];
    corner[2].y = (affine[3] * (srcSize.height-1)) + (affine[4] * 0) + affine[5];
    corner[3].x = (affine[0] * (srcSize.height-1)) + (affine[1] * (srcSize.width-1)) + affine[2];
    corner[3].y = (affine[3] * (srcSize.height-1)) + (affine[4] * (srcSize.width-1)) + affine[5];


    for (int i = 0; i< 4; i++){
        if(corner[i].x < minX)  minX = corner[i].x;
        if(corner[i].x > maxX)  maxX = corner[i].x;
        if(corner[i].y < minY)  minY = corner[i].y;
        if(corner[i].y > maxY)  maxY = corner[i].y;
    }

    dstSizePtr->height = ((Rpp32s)maxX - (Rpp32s)minX) + 1;
    dstSizePtr->width = ((Rpp32s)maxY - (Rpp32s)minY) + 1;

    return RPP_SUCCESS;
}

RppStatus warp_affine_output_offset(RppiSize srcSize, RppiPoint *offset,
                                    float *affine)
{
    float minX = 0, minY = 0, maxX = 0, maxY = 0;
    
    RppiPoint corner[4];

    corner[0].x = (affine[0] * 0) + (affine[1] * 0  + affine[2]);
    corner[0].y = (affine[3] * 0) + (affine[4] * 0  + affine[5]);
    corner[1].x = (affine[0] * 0) + affine[1] * (srcSize.width-1) + affine[2];
    corner[1].y = (affine[3] * 0) + affine[4] * (srcSize.width-1) + affine[5];
    corner[2].x = (affine[0] * (srcSize.height-1)) + (affine[1] * 0) + affine[2];
    corner[2].y = (affine[3] * (srcSize.height-1)) + (affine[4] * 0) + affine[5];
    corner[3].x = (affine[0] * (srcSize.height-1)) + (affine[1] * (srcSize.width-1)) + affine[2];
    corner[3].y = (affine[3] * (srcSize.height-1)) + (affine[4] * (srcSize.width-1)) + affine[5];


    for (int i = 0; i< 4; i++){
        if(corner[i].x < minX)  minX = corner[i].x;
        if(corner[i].x > maxX)  maxX = corner[i].x;
        if(corner[i].y < minY)  minY = corner[i].y;
        if(corner[i].y > maxY)  maxY = corner[i].y;
    }


    offset->x = ((unsigned int)minX);
    offset->y = ((unsigned int)minY);
    return RPP_SUCCESS;
}


RppStatus rotate_output_size(RppiSize srcSize, RppiSize *dstSizePtr,
                             Rpp32f angleDeg)
{
    Rpp32f angleRad = RAD(angleDeg);

    Rpp32f rotate[4] = {0};
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = -sin(angleRad);
    rotate[3] = cos(angleRad);
   
    float minX = 0, minY = 0, maxX = 0, maxY = 0;
    
    RppiPoint corner[4];

    corner[0].x = 0;
    corner[0].y = 0;
    corner[1].x =  rotate[1] * (srcSize.width-1);
    corner[1].y =  rotate[3] * (srcSize.width-1);
    corner[2].x = (rotate[0] * (srcSize.height-1)); 
    corner[2].y = (rotate[2] * (srcSize.height-1)) ;
    corner[3].x = (rotate[0] * (srcSize.height-1)) + (rotate[1] * (srcSize.width-1));
    corner[3].y = (rotate[2] * (srcSize.height-1)) + (rotate[3] * (srcSize.width-1));


    for (int i = 0; i< 4; i++){
        if(corner[i].x < minX)  minX = corner[i].x;
        if(corner[i].x > maxX)  maxX = corner[i].x;
        if(corner[i].y < minY)  minY = corner[i].y;
        if(corner[i].y > maxY)  maxY = corner[i].y;
    }

    dstSizePtr->width = ((Rpp32s)maxX - (Rpp32s)minX) + 20;
    dstSizePtr->height = ((Rpp32s)maxY - (Rpp32s)minY) + 20;
    return RPP_SUCCESS;
}

RppStatus rotate_output_offset(RppiSize srcSize, RppiPoint *offset,
                               Rpp32f angleDeg)
{
    Rpp32f angleRad = angleDeg;
    Rpp32f rotate[4] = {0};
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = -sin(angleRad);
    rotate[3] = cos(angleRad);

    float minX = 0, minY = 0, maxX = 0, maxY = 0;
    
    RppiPoint corner[4];
    int i;
    corner[0].x = (rotate[0] * 0) + (rotate[1] * 0);
    corner[0].y = (rotate[2] * 0) + (rotate[3] * 0);
    corner[1].x = (rotate[0] * 0) + rotate[1] * (srcSize.width-1);
    corner[1].y = (rotate[2] * 0) + rotate[3] * (srcSize.width-1);
    corner[2].x = (rotate[0] * (srcSize.height-1)) + (rotate[1] * 0);
    corner[2].y = (rotate[2] * (srcSize.height-1)) + (rotate[3] * 0);
    corner[3].x = (rotate[0] * (srcSize.height-1)) + (rotate[1] * (srcSize.width-1));
    corner[3].y = (rotate[2] * (srcSize.height-1)) + (rotate[3] * (srcSize.width-1));


    for ( i = 0; i< 4; i++){
        if(corner[i].x < minX)  minX = corner[i].x;
        if(corner[i].x > maxX)  maxX = corner[i].x;
        if(corner[i].y < minY)  minY = corner[i].y;
        if(corner[i].y > maxY)  maxY = corner[i].y;
    }
    
    offset->x = ((unsigned int)minX);
    offset->y = ((unsigned int)minY);
    return RPP_SUCCESS;
}