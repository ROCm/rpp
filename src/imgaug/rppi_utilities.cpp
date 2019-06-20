#include <rppdefs.h>
#include <math.h>
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
    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            Rpp32f newi = 0, newj = 0;
            newi = (affine[0] * i) + (affine[1] * j) + (affine[2] * 1);
            newj = (affine[3] * i) + (affine[4] * j) + (affine[5] * 1);
            if (newi < minX)
            {
                minX = newi;
            }
            if (newj < minY)
            {
                minY = newj;
            }
            if (newi > maxX)
            {
                maxX = newi;
            }
            if (newj > maxY)
            {
                maxY = newj;
            }
        }
    }
    dstSizePtr->height = ((Rpp32s)maxX - (Rpp32s)minX) + 1;
    dstSizePtr->width = ((Rpp32s)maxY - (Rpp32s)minY) + 1;
    return RPP_SUCCESS;
}

RppStatus warp_affine_output_offset(RppiSize srcSize, RppiPoint *offset,
                                    float *affine)
{
    float minX = 0, minY = 0, maxX = 0, maxY = 0;
    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            Rpp32f newi = 0, newj = 0;
            newi = (affine[0] * i) + (affine[1] * j) + (affine[2] * 1);
            newj = (affine[3] * i) + (affine[4] * j) + (affine[5] * 1);
            if (newi < minX)
            {
                minX = newi;
            }
            if (newj < minY)
            {
                minY = newj;
            }
            if (newi > maxX)
            {
                maxX = newi;
            }
            if (newj > maxY)
            {
                maxY = newj;
            }
        }
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
        if(corner[i].y > maxX)  maxY = corner[i].y;
    }

    dstSizePtr->height = ((Rpp32s)maxX - (Rpp32s)minX) + 1;
    dstSizePtr->width = ((Rpp32s)maxY - (Rpp32s)minY) + 1;

    return RPP_SUCCESS;
}

RppStatus rotate_output_offset(RppiSize srcSize, RppiPoint *offset,
                               Rpp32f angleDeg)
{
    Rpp32f angleRad = RAD(angleDeg);
    Rpp32f rotate[6] = {0};
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = 0;
    rotate[3] = -sin(angleRad);
    rotate[4] = cos(angleRad);
    rotate[5] = 0;

    float minX = 0, minY = 0, maxX = 0, maxY = 0;
    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            Rpp32f newi = 0, newj = 0;
            newi = (rotate[0] * i) + (rotate[1] * j) + (rotate[2] * 1);
            newj = (rotate[3] * i) + (rotate[4] * j) + (rotate[5] * 1);
            if (newi < minX)
            {
                minX = newi;
            }
            if (newj < minY)
            {
                minY = newj;
            }
            if (newi > maxX)
            {
                maxX = newi;
            }
            if (newj > maxY)
            {
                maxY = newj;
            }
        }
    }
    offset->x = ((unsigned int)minX);
    offset->y = ((unsigned int)minY);
    return RPP_SUCCESS;
}