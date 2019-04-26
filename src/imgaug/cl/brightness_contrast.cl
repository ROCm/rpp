__kernel void brightness(  __global double *a,                   
                           __global double *c,                       
                                    int alpha,                   
                                    int  beta,                       
                         const unsigned int n)                    
{

    //Get our global thread ID                                  
    int id = get_global_id(0);                                  
                                                                
    //Make sure we do not go out of bounds                      
    if (id < n)                                                 
        c[id] = alpha * a[id] + beta;                                  
}        