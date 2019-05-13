__kernel void contrast_streach(  __global unsigned char *a,
                                __global unsigned char *c,
                                   const unsigned int min,
                                   const unsigned int max,
                               const unsigned int new_min,
                               const unsigned int new_max,
                               const unsigned int height,
                               const unsigned int width)
{
   //Get our global thread ID
   int id_x = get_global_id(0);
   int id_y = get_global_id(1);

   //Make sure we do not go out of bounds
   if (id_x < height && id_y < width )
       c[id_x * width + id_y ] =
               (a[id_x * width + id_y] - min) * (new_max - new_min)/((max - min) * 1.0) + new_min ;
}