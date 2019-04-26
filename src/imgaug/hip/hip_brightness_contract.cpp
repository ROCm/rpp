#include <hip/hip_rpp_comman.hpp>

#define COLS 10
#define ROWS 10

template <typename T>
__global__ void brightness_contrast_kernel( T* inImgPtr, T* outImgPtr,
                                            const int rows,const int cols, const int chns,
                                            const Rpp32f alpha, const Rpp32f beta )
{

    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;
    int z = hipThreadIdx_z + hipBlockIdx_z * hipBlockDim_z;
    if ( x >= cols || y >= rows || z >= chns)
        return;

    //Packed
    T result = inImgPtr[z + x*chns + y*cols*rows*chns] * alpha + beta;
    outImgPtr[z + x*chns + y*cols*rows*chns] = result;

    //Planar impl is same for 3 channel alone

}


template <typename T>
void brightness_contrast_caller(T *pSrc, RppiSize imgDim, T *pDst,
                                Rpp32f alpha, Rpp32f beta, RppiChnFormat chnFormat,
                                )
{

    if (KERNEL)
    {   dim3 grid;
        dim3 block;
        hipLaunchKernelGGL( (brightness_contrast_kernel<T>),
                            grid, block, 0, stream,
                            inputPtr, outputPtr,
                            imgDim.height, imgDim.width, imgDim.width,
                            alpha, beta );
    }
    else if (BLAS)
    {

    }

}