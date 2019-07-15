#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
unsigned int xorshift(int pixid) {
    unsigned int x = 123456789;
    unsigned int w = 88675123;
    unsigned int seed = x + pixid;
    unsigned int t = seed ^ (seed << 11);
    unsigned int res = w ^ (w >> 19) ^ (t ^(t >> 8));
	return res;
}
__kernel void jitter(
	    const __global unsigned char* input,
	    __global  unsigned char* output,
	    const unsigned int height,
	    const unsigned int width,
	    const unsigned int channel,
	    const unsigned int minJitter,
        const unsigned int maxJitter
)
{

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;
    //output[pixIdx] = 0;
    int rand = xorshift(pixIdx) % (maxJitter - minJitter + 1) + minJitter;
    int res = input[pixIdx] + rand;
    //res = rand;
    output[pixIdx] = saturate_8u(res);

}
