
# Radeon Performance Primitives Library

Radeon performance primitives(RPP) libaray is  a comprehensive high performance computer vision library for AMD(CPU and GPU) with HIP and OpenCL backend on the device side.


## Top level design
<p align="center"><img width="60%" src="rpp_new.png" /></p>



RPP is developed for __Linux__ operating system.
##### Prerequisites
1. Ubuntu `16.04`/`18.04`
2. [ROCm supported hardware](https://rocm.github.io/hardware.html)
3. [ROCm](https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)

## Functions Included
* Brightness
* Contrast
* Gamma
* Blend
* Warp Affine
* Resize
* CropResize
* Rotation
* Flip(Horizontal, Vertical and Both)
* Blur (Gaussian 3x3)
* Fisheye lens
* Vignette
* Jitter
* Salt and pepper noise
* Snow flakes
* Rain drops
* Fog
* Color temperature
* Lens correction
* Pixelization
* Exposure modification

## Variations
* Support for 3C(RGB) and 1C(Grayscale) images
* Planar and Packed
* Host and GPU 

## [Instructions to build the library](#rpp-installation)

```sh
$ git clone https://github.com/LokeshBonta/AMD-RPP.git
$ cd AMD-RPP
$ mkdir build
$ cd build
$ cmake -DBACKEND=OCL ..
$ make -j4
$ sudo make install
```
## MIVisionX(OpenVX) Support
Extended RPP support as a functionality through OpenVX
 [MIVisionX](https://github.com/shobana-mcw/MIVisionX) (Clone the repository from the link)
### To build OpenVX with RPP extension
* RPP should be installed, follow [Instructions to build the library](#rpp-installation)
```sh
$ git  clone https://github.com/shobana-mcw/MIVisionX.git
$ cd MIVisionX
$ git  checkout main-dev
$ mkdir build
$ cd build ; cmake .. ; make -j4 //For GPU support
        or
$ cd build ; cmake -DCMAKE_DISABLE_FIND_PACKAGE_OpenCL=TRUE;  //For CPU support
$ make -j4
$ sudo make install
```




## Miscellaneous
#### RPP stand-alone code snippet
```c
err = clGetPlatformIDs(1, &platform_id, NULL);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    theContext = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    theQueue = clCreateCommandQueue(theContext, device_id, 0, &err);

    d_a = clCreateBuffer(theContext, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_a, CL_TRUE, 0,  bytes, h_a, 0, NULL, NULL);
    cl_mem d_f;
    d_f = clCreateBuffer(theContext, CL_MEM_READ_ONLY, f_bytes, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, d_f, CL_TRUE, 0, f_bytes, h_f, 0, NULL, NULL)
    
    Rpp32f alpha=2;
    Rpp32s beta=1;
    
    RppiSize srcSize;
    srcSize.height=height;
    srcSize.width=width;
    rppi_brighten_8u_pln1_gpu( d_a, srcSize, d_c, alpha, beta, theQueue);//device side API call

```

#### RPP with [GDF](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/utilities/runvx/README.md#amd-runvx)(uses OpenVX) code snippet

```sh
# specify input source for input image and request for displaying input and output images
read input  ../images/face.jpg
view input  inputWindow
view output brightnessWindow

#import RPP library
import vx_rpp
# create input and output images
data input  = image:480,360,RGB2
data output = image:480,360,U008

# compute luma image channel from input RGB image
data yuv  = image-virtual:0,0,IYUV
data luma = image-virtual:0,0,U008
node org.khronos.openvx.color_convert input yuv
node org.khronos.openvx.channel_extract yuv !CHANNEL_Y luma

#compute brightness and contrast in luma image using Brightness function
data alpha = scalar:FLOAT32,1.0  #contrast control
data beta = scalar:INT32,30    #brightness control
node org.rpp.Brightness luma output alpha beta



```
