__kernel void contrast_stretch(  __global unsigned char *a,
                         __global double *c,
                         const unsigned int height,
                          const unsigned int width)
{
    //Get our global thread ID
    int id = get_global_id(0);
    int r,g,b, min, max;

   // int id_y = get_global_id(1);
    //Make sure we do not go out of bounds

    if (id < height * width ){
        r = a[id];
        g = a[id + height * width];
        b = a[id + 2* height * width];

        min = (r < g && r< b)? r : ((g < b)? g: b);

        if (r > g && r > b){
            max = r;
            c[id] =  60 * ((g - b)/(max - min)) ;
        }
        else if (g > b){
            max = g;
            c[id] =  60 * (2 + (b - r)/(max - min));
        }
        else{
            max = b;
            c[id] =  60 * (4 + (r - g)/(max - min)) ;
        }

        c[id +  height * width] = (max - min)/(max * 1.0);
        c[id + 2* height * width] = max;

    }

}  ;