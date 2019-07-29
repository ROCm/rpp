#define SWAP(a,b) {__local int *tmp=a;a=b;b=tmp;}

__kernel void scan(__global int *input,
                   __global int *output,
                   __local  int *b,
                   __local  int *c)
{
    //printf("Inside scan");
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint gs = get_local_size(0);

    c[lid] = b[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s = 1; s < gs; s <<= 1) {
        if(lid < (s-1)) {
            c[lid] = b[lid]+b[lid-s];
            c[lid + 256] = b[lid+ 256]+b[lid+ 256-s];
            c[lid + 512] = b[lid+ 512]+b[lid + 512-s];
        } else {
            c[lid] = b[lid];
            c[lid + 256] = b[lid + 256];
            c[lid + 512] = b[lid + 512];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(b,c);
    }
    output[gid] = b[lid];
    output[gid + 256] = b[lid + 256];
    output[gid + 512] = b[lid + 512];
    //printf("%d",output[gid]);
}
