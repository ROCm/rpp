//////////////////Conversion Functions ////////////////////////////_
_kernel void rgb2hsv_pln(__global unsigned char *a,                       
                         __global double *c,                       
                         const unsigned int height,              
                         const unsigned int width)                   
{                                                                 
    //Get our global thread ID                                    
    int id = get_global_id(0);                                    
    double r,g,b, min, max, delta;                               
                                                                                                  
    //Make sure we do not go out of bounds                        
                                                                  
    if (id < height * width ){                                    
        r = a[id] / 255.0;                                                
        g = a[id + height * width] / 255.0;                                
        b = a[id + 2* height * width]/ 255.0;                              
                                                                   
        min = (r < g && r< b)? r : ((g < b)? g: b); 
        max = (r > g && r > b)? r : ((g > b)? g: b);  

        delta = max - min;   

        if (delta == 0) c[id] = 0;
        else {
            if (max == r)
                c[id] = 60 * ((g - b)/delta);
            else if (max == g)
                c[id] = 60 * ((b - r)/delta + 2);
            else 
                c[id] = 60 * ((r - g)/delta + 4);
        }         


        if ( c[id] < 0) c[id] = c[id] +360;                            
        if (max == 0) c[id +  height * width] = 0;
        else c[id + height * width] = delta / max;                   
        c[id + 2* height * width] = max;                             
                                                                     
    }                                                                
                                                                     
}       

__kernel void rgb2hsv_pkd(  __global unsigned char *a,                       
                            __global double *c,                       
                              const unsigned int height,              
                              const unsigned int width)                   
{                                                                 
    //Get our global thread ID                                    
    int id = get_global_id(0);                                    
    double r,g,b, min, max, delta;                               
                                                                                                   
    //Make sure we do not go out of bounds                        
    id = id * 3;                                                           
    if (id < 3 *height * width ){                                    
        r = a[id] / 255.0;                                                
        g = a[id + 1] / 255.0;                                
        b = a[id + 2]/ 255.0;                              
                                                                   
        min = (r < g && r< b)? r : ((g < b)? g: b); 
        max = (r > g && r > b)? r : ((g > b)? g: b);  

        delta = max - min;   

        if (delta == 0) c[id] = 0;
        else {
            if (max == r)
                c[id] = 60 * ((g - b)/delta);
            else if (max == g)
                c[id] = 60 * ((b - r)/delta + 2);
            else 
                c[id] = 60 * ((r - g)/delta + 4);
        }         


        if ( c[id] < 0)  c[id] = c[id] +360;                            
        if (max == 0) c[id +  1] = 0;
        else c[id + 1] = delta / max;                   
        c[id + 2] = max;                             
                                                                     
    }                                                                
                                                                     
}     

__kernel void hsv2rgb_pln(   __global const double *a,         
                         __global  unsigned char *c,        
                         const unsigned int height,       
                        const unsigned int width)       
{       
    //Get our global thread ID       
       

    int pixIdx  = get_global_id(0);         
    double     hh, p, q, t, ff;        
    int        i;        
    double     h,s,v;   
    pixIdx = 3 * pixIdx;
    //Make sure we do not go out of bounds       
       
    if (id < height*width ){             
       
        h = a[pixIdx];       
        s = a[pixIdx + height * width ] ;       
        v = a[pixIdx + 2* height * width] ;       
        
        if (s <= 0){       
            c[pixIdx] = 0;       
            c[pixIdx +  height * width] = 0;        
            c[pixIdx + 2* height * width] = 0;        
        }       
        
        hh = h;
        if(h == 360.0) hh = 0.0;       
        hh /= 60.0;       
        i = (int)hh;       
        ff = hh - i;       
        p = v * (1.0 - s);       
        q = v * (1.0 - (s * ff));       
        t = v * (1.0 - (s * (1.0 - ff)));       
               
    switch(i){       
    case 0:       
        c[pixIdx] = v * 255;       
        c[pixIdx +  height * width] = t * 255  ;       
        c[pixIdx + 2* height * width] = p * 255;       
        break;       
    case 1:       
        c[pixIdx] = q * 255;       
        c[pixIdx +  height * width] = v * 255  ;       
        c[pixIdx + 2* height * width] = p * 255 ;       
        break;       
    case 2:       
        c[pixIdx] = p * 255 ;       
        c[pixIdx +  height * width] = v * 255;       
        c[pixIdx + 2* height * width] = t * 255;       
        break;       
       
    case 3:       
        c[pixIdx] = p * 255;       
        c[pixIdx +  height * width] = q * 255;       
        c[pixIdx + 2* height * width] = v * 255;       
        break;       

    case 4:       
        c[pixIdx] = t * 255;       
        c[pixIdx +  height * width] = p * 255 ;       
        c[pixIdx + 2* height * width] = v * 255;       
        break;       
    case 5:       
    default:       
        c[pixIdx] = v * 255;       
        c[pixIdx +  height * width] = p * 255 ;       
        c[pixIdx + 2* height * width] = q * 255;       
        break;       
     }       
       
    }       
           
}  

__kernel void hsv2rgb_pkd(__global const double *a,         
                          __global  unsigned char *c,        
                          const unsigned int height,       
                          const unsigned int width)       
{       
    //Get our global thread ID       
       

    int pixIdx = get_global_id(0);         
    double     hh, p, q, t, ff;        
    int        i;        
    double     h,s,v;  
    pixIdx = 3 * pixIdx;
    //Make sure we do not go out of bounds       
    
    if (pixIdx < height*width*3 ){             
       
        h = a[pixIdx];       
        s = a[pixIdx + 1 ] ;       
        v = a[pixIdx + 2] ;       
        
        if (s <= 0){       
            c[pixIdx] = 0;       
            c[pixIdx + 1] = 0;        
            c[pixIdx + 2] = 0;        
        }       
        
        hh = h;
        if(h == 360.0) { hh = 0.0; }       
        hh /= 60.0;       
        i = (int)hh;       
        ff = hh - i;       
        p = v * (1.0 - s);       
        q = v * (1.0 - (s * ff));       
        t = v * (1.0 - (s * (1.0 - ff)));       
               
    switch(i)
    {       
    case 0:       
        c[pixIdx] = v * 255;       
        c[pixIdx + 1] = t * 255 ;       
        c[pixIdx + 2] = p * 255;       
        break;       
    case 1:       
        c[pixIdx] = q * 255;       
        c[pixIdx + 1] = v * 255  ;       
        c[pixIdx + 2] = p * 255 ;       
        break;       
    case 2:       
        c[pixIdx] = p * 255;       
        c[pixIdx + 1] = v * 255;       
        c[pixIdx + 2] = t * 255;       
        break;       
       
    case 3:       
        c[pixIdx] = p * 255;       
        c[pixIdx +  1] = q * 255;       
        c[pixIdx + 2] = v * 255;       
        break;       

    case 4:       
        c[pixIdx] = t * 255;       
        c[pixIdx +  1] = p * 255 ;       
        c[pixIdx + 2] = v * 255;       
        break;       
    case 5:       
    default:       
        c[pixIdx] = v * 255;       
        c[pixIdx +  1] = p * 255 ;       
        c[pixIdx + 2] = q * 255;       
        break;       
     }       
       
    }       
           
}  


// Hue and Satutation Modification /////////////////////////////

__kernel void huergb_pln(   __global  unsigned char *a,         
                            __global  unsigned char *c,
                            __global  double *temp,
                            const  double hue,
                            const  double sat,         
                            const unsigned int height,       
                            const unsigned int width)       
{       
    //Get our global thread ID                                        
    int id = get_global_id(0);                                    
    double r,g,b, min, max, delta;                               
                                                                                                  
    //Make sure we do not go out of bounds                        
    id = id ;                                                           
    if (id < 3 *height * width ){                                    
        r = a[id] / 255.0;                                                
        g = a[id + height * width] / 255.0;                                
        b = a[id + 2 *height * width]/ 255.0;                              
                                                                   
        min = (r < g && r< b)? r : ((g < b)? g: b); 
        max = (r > g && r > b)? r : ((g > b)? g: b);  

        delta = max - min;   

        if (delta == 0) c[id] = 0;
        else {
            if (max == r)
                temp[id] = 60 * ((g - b)/delta);
            else if (max == g)
                temp[id] = 60 * ((b - r)/delta + 2);
            else 
                temp[id] = 60 * ((r - g)/delta + 4);
        }         

        temp[id] += hue;
        if ( temp[id] < 0)  temp[id] = temp[id] +360;                            
        if (max == 0) temp[id +  1] = 0;
        else temp[id + height * width] = delta / max;  
        temp[id + height * width] += sat;                 
        temp[id + 2*height * width] = max;   

    barrier(CLK_GLOBAL_MEM_FENCE);
            
    double     hh, p, q, t, ff;        
    int        i;        
    double     h,s,v;  
    
    int pixIdx = id;    
    //Make sure we do not go out of bounds       
         
       
        h = temp[pixIdx];       
        s = temp[pixIdx + height * width] ;       
        v = temp[pixIdx + 2*height * width] ;       
        
        if (s <= 0){       
            c[pixIdx] = 0;       
            c[pixIdx + height * width] = 0;        
            c[pixIdx + 2*height * width] = 0;        
        }       
        
        hh = h;
        if(h == 360.0) { hh = 0.0; }       
        hh /= 60.0;       
        i = (int)hh;       
        ff = hh - i;       
        p = v * (1.0 - s);       
        q = v * (1.0 - (s * ff));       
        t = v * (1.0 - (s * (1.0 - ff)));       
               
    switch(i)
    {       
    case 0:       
        c[pixIdx] = v * 255;       
        c[pixIdx + height * width] = t * 255 ;       
        c[pixIdx + 2*height * width] = p * 255;       
        break;       
    case 1:       
        c[pixIdx] = q * 255;       
        c[pixIdx + height * width] = v * 255  ;       
        c[pixIdx + 2*height * width] = p * 255 ;       
        break;       
    case 2:       
        c[pixIdx] = p * 255;       
        c[pixIdx + height * width] = v * 255;       
        c[pixIdx + 2*height * width] = t * 255;       
        break;       
       
    case 3:       
        c[pixIdx] = p * 255;       
        c[pixIdx + height * width] = q * 255;       
        c[pixIdx + 2* height * width] = v * 255;       
        break;       

    case 4:       
        c[pixIdx] = t * 255;       
        c[pixIdx +  height * width] = p * 255 ;       
        c[pixIdx + 2*height * width] = v * 255;       
        break;       
    case 5:       
    default:       
        c[pixIdx] = v * 255;       
        c[pixIdx +  height * width] = p * 255 ;       
        c[pixIdx + 2*height * width] = q * 255;       
        break;       
     }       
       
    }       
           
}  


__kernel void huergb_pkd(   __global  unsigned char *a,         
                            __global  unsigned char *c,
                            __global  double *temp,
                            const  double hue,
                            const  double sat,         
                            const unsigned int height,       
                            const unsigned int width)       
{       
    //Get our global thread ID                                        
    int id = get_global_id(0);                                    
    double r,g,b, min, max, delta;                               
                                                                                                  
    //Make sure we do not go out of bounds                        
    id = id * 3;                                                           
    if (id < 3 *height * width ){                                    
        r = a[id] / 255.0;                                                
        g = a[id + 1] / 255.0;                                
        b = a[id + 2]/ 255.0;                              
                                                                   
        min = (r < g && r< b)? r : ((g < b)? g: b); 
        max = (r > g && r > b)? r : ((g > b)? g: b);  

        delta = max - min;   

        if (delta == 0) c[id] = 0;
        else {
            if (max == r)
                temp[id] = 60 * ((g - b)/delta);
            else if (max == g)
                temp[id] = 60 * ((b - r)/delta + 2);
            else 
                temp[id] = 60 * ((r - g)/delta + 4);
        }         

        temp[id] += hue;
        if ( temp[id] < 0)  temp[id] = temp[id] +360;                            
        if (max == 0) temp[id +  1] = 0;
        else temp[id + 1] = delta / max;  
        temp[id + 1] += sat;                 
        temp[id + 2] = max;   

    barrier(CLK_GLOBAL_MEM_FENCE);
            
    double     hh, p, q, t, ff;        
    int        i;        
    double     h,s,v;  
    
    int pixIdx = id;    
    //Make sure we do not go out of bounds       
         
       
        h = temp[pixIdx];       
        s = temp[pixIdx + 1] ;       
        v = temp[pixIdx + 2] ;       
        
        if (s <= 0){       
            c[pixIdx] = 0;       
            c[pixIdx + 1] = 0;        
            c[pixIdx + 2] = 0;        
        }       
        
        hh = h;
        if(h == 360.0) { hh = 0.0; }       
        hh /= 60.0;       
        i = (int)hh;       
        ff = hh - i;       
        p = v * (1.0 - s);       
        q = v * (1.0 - (s * ff));       
        t = v * (1.0 - (s * (1.0 - ff)));       
               
    switch(i)
    {       
    case 0:       
        c[pixIdx] = v * 255;       
        c[pixIdx + 1] = t * 255 ;       
        c[pixIdx + 2] = p * 255;       
        break;       
    case 1:       
        c[pixIdx] = q * 255;       
        c[pixIdx + 1] = v * 255  ;       
        c[pixIdx + 2] = p * 255 ;       
        break;       
    case 2:       
        c[pixIdx] = p * 255;       
        c[pixIdx + 1] = v * 255;       
        c[pixIdx + 2] = t * 255;       
        break;       
       
    case 3:       
        c[pixIdx] = p * 255;       
        c[pixIdx +  1] = q * 255;       
        c[pixIdx + 2] = v * 255;       
        break;       

    case 4:       
        c[pixIdx] = t * 255;       
        c[pixIdx +  1] = p * 255 ;       
        c[pixIdx + 2] = v * 255;       
        break;       
    case 5:       
    default:       
        c[pixIdx] = v * 255;       
        c[pixIdx +  1] = p * 255 ;       
        c[pixIdx + 2] = q * 255;       
        break;       
     }       
       
    }       
           
}  



////////////////////////////////////////////