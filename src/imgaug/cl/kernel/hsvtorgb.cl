__kernel void hsv2rgb(   __global const double *a,
                         __global  double *c,
                         const unsigned short height,
                         const unsigned short width,
                         const unsigned short channel)
{
    //Get our global thread ID
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    ind id_z = get_global_id(2);
      
    double      hh, p, q, t, ff;
    long        i;    
    //Make sure we do not go out of bounds

    if (id_x < height && id_y < width && id_z < channel ){
        pixIdx = id_x + id_y * width + id_z * width * height;
       
        h = a[pixIdx];
        s = a[pixIdx + height * width ];
        v = a[pixIdx + 2* height * width];
        
        if (s <= 0){
            c[pixIdx] = 0;
            c[pixIdx +  height * width] = 0;
            c[pixIdx + 2* height * width] = 0;
        }

        if(h >= 360.0) hh = 0.0;
        h /= 60.0;
        i = (long)hh;
        ff = hh - i;
        p = v * (1.0 - s);
        q = v * (1.0 - (s * ff));
        t = v * (1.0 - (s * (1.0 - ff)));
        
    switch(i){
    case 0:
        c[pixIdx] = v * 255;
        c[pixIdx +  height * width] = t * 255;
        c[pixIdx + 2* height * width] = p * 255;
        break;
    case 1:
        c[pixIdx] = q * 255;
        c[pixIdx +  height * width] = v * 255;
        c[pixIdx + 2* height * width] = p * 255;
        break;
    case 2:
        c[pixIdx] = p * 255;
        c[pixIdx +  height * width] = v * 255;
        c[pixIdx + 2* height * width] = t * 255;
        break;

    case 3:
        c[pixIdx] = p * 255;
        c[pixIdx +  height * width] = q * 255;
        c[pixIdx + 2* height * width] = v * 255;
        break;
    case 4:
        c[pixIdx] = t *255;
        c[pixIdx +  height * width] = p * 255;
        c[pixIdx + 2* height * width] = v * 255;
        break;
    case 5:
    default:
        c[pixIdx] = v * 255;
        c[pixIdx +  height * width] = p * 255;
        c[pixIdx + 2* height * width] = q * 255;
        break;
     }

    }

}  ;

