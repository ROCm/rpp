#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value)   ))
__kernel void pixelate_planar (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            __global  float* filter,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int x1,
                            const unsigned int y1,
                            const unsigned int x2,
                            const unsigned int y2,
                            const unsigned int channel,
                            const unsigned int filterSize
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;
    int hfFiltSz = filterSize/2;
    if ((x1 <= id_x) && (id_x <= x2) && (y1 <= id_y) &&( id_y <= y2))
    {
        // Handle shd be padded area here or in validation
        if ( (id_x < hfFiltSz) || (id_y < hfFiltSz) ||
            (id_y >= (height-hfFiltSz)) || (id_x >= (width-hfFiltSz)) )
        {
            dstPtr[pixIdx] = srcPtr[pixIdx];
            return;
        }

        float sum = 0.0;
        for (int ri = (-1 * hfFiltSz) , rf = 0;
                (ri <= hfFiltSz) && (rf < filterSize);
                    ri++, rf++)
        {
            for (int ci = (-1 * hfFiltSz) , cf = 0;
                    (ci <= hfFiltSz) && (cf < filterSize);
                        ci++, cf++)
            {
                const int idxF = rf + cf * filterSize ;
                const int idxI = pixIdx + ri + ci * width;
                sum += filter[idxF]*srcPtr[idxI];
            }
        }
        int res = (int)sum;
        dstPtr[pixIdx] = saturate_8u(res);
    } else 
    {
        dstPtr[pixIdx] = srcPtr[pixIdx];
    }

}

__kernel void pixelate_packed (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            __global  float* filter,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int x1,
                            const unsigned int y1,
                            const unsigned int x2,
                            const unsigned int y2,
                            const unsigned int channel,
                            const unsigned int filterSize
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;
    //printf(" %d %d %d %d ",x1,y1,x2,y2);
    int pixIdx = id_x * channel+ id_y * width * channel + id_z;
    int hfFiltSz = filterSize/2;
    if ((x1 <= id_x) &&(id_x <= x2) && (y1 <= id_y) &&( id_y <= y2))
     {
        // Handle shd be padded area here or in validation
        if ( (id_x < hfFiltSz) || (id_y < hfFiltSz) ||
        (id_y >= (height-hfFiltSz)) || (id_x >= (width-hfFiltSz)) )
        {
            dstPtr[pixIdx] = srcPtr[pixIdx];
            return;
        }

        int res;

        float sum = 0.0;
        for (int ri = (-1 * hfFiltSz) , rf = 0;
                (ri <= hfFiltSz) && (rf < filterSize);
                    ri++, rf++)
        {
            for (int ci = (-1 * hfFiltSz) , cf = 0;
                    (ci <= hfFiltSz) && (cf < filterSize);
                        ci++, cf++)
            {
                const int idxF = rf + cf * filterSize ;
                const int idxI = pixIdx + ri * channel + ci * width *channel;
                sum += filter[idxF]*srcPtr[idxI];
            }
        }
        res = (int)sum;
        dstPtr[pixIdx] = saturate_8u(res);
    } else 
    {
        dstPtr[pixIdx] = srcPtr[pixIdx];
    }
}