__kernel void warp_affine_pln (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            __global  float* affine,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int channel
)
{
   int id_x = get_global_id(0);
   int id_y = get_global_id(1);
   int id_z = get_global_id(2);
   
   int xc = id_x - source_width/2;
   int yc = id_y - source_height/2;

   int k ;
   int l ;

   k = (int)((affine[0] * xc )+ (affine[1] * yc)) + affine[2];
   l = (int)((affine[3] * xc) + (affine[4] * yc)) + affine[5];

   k = k + source_width/2;
   l = l + source_height/2;
    
   if (l < source_height && l >=0 && k < source_width && k >=0 )
   dstPtr[(id_z * dest_height * dest_width) + (id_y * dest_width) + id_x] =
                            srcPtr[(id_z * source_height * source_width) + (l * source_width) + k];
   else
   dstPtr[(id_z * dest_height * dest_width) + (id_y * dest_width) + id_x] = 0;

}


__kernel void warp_affine_pkd (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            __global  float* affine,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int channel
)
{

   int id_x = get_global_id(0);
   int id_y = get_global_id(1);
   int id_z = get_global_id(2);
   
   int xc = id_x - source_width/2;
   int yc = id_y - source_height/2;

   int k ;
   int l ;

   k = (int)((affine[0] * xc )+ (affine[1] * yc)) + affine[2];
   l = (int)((affine[3] * xc) + (affine[4] * yc)) + affine[5];

   k = k + source_width/2;
   l = l + source_height/2;

   /*if (l < source_height && l >=0 && k < source_width && k >=0 )
   dstPtr[id_z + (channel * id_y * dest_width) + (channel * id_x)] =
                             srcPtr[id_z + (channel * l * source_width) + (channel * k)];
   else*/
   dstPtr[id_z + (channel * id_y * dest_width) + (channel * id_x)] = 0;

}
