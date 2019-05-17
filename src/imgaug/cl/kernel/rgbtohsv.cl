__kernel void rgb2hsv(  __global unsigned char *a,
                         __global double *c,
                         const unsigned short height,
                         const unsigned short width,
                         const unsigned short channel)
{
    //Get our global thread ID
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    ind id_z = get_global_id(2);

    int r,g,b, min, max;
    int pixIdx;
     
   // int id_y = get_global_id(1);
   //Make sure we do not go out of bounds

    if (id_x < height && id_y < width && id_z < channel ){
        pixIdx = id_x + id_y * width + id_z * width * height;
        r = a[pixIdx];
        g = a[pixIdx + height * width ];
        b = a[pixIdx + 2* height * width];

        min = (r < g && r< b)? r : ((g < b)? g: b);

        if (r > g && r > b){
            max = r;
            c[pixIdx] =  60 * ((g - b)/(max - min)) ;
        }
        else if (g > b){
            max = g;
            c[pixIdx] =  60 * (2 + (b - r)/(max - min));
        }
        else{
            max = b;
            c[pixIdx] =  60 * (4 + (r - g)/(max - min)) ;
        }
        
        c[pixIdx] = c[pixIdx] % 360; // To make sure 360 + x = x
        c[pixIdx +  height * width] = (max - min)/(max * 1.0);
        c[pixIdx + 2* height * width] = max;

    }

}  ;