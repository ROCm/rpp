# Radeon Performance Primitives Library

Radeon Performance Primitives (RPP) library is a comprehensive high-performance computer vision library for AMD (CPU and GPU) with HIP and OpenCL backend on the device side.

## Top level design
<p align="center"><img width="60%" src="rpp_new.png" /></p>

RPP is developed for __Linux__ operating system.

##### Prerequisites
1. Ubuntu `16.04`/`18.04`
2. [ROCm supported hardware](https://rocm.github.io/hardware.html)
3. [ROCm](https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)

## Functions Included

### Image Augmentation Category

#### Enhancements
* Brightness modification
* Contrast modification
* Hue modification
* Saturation modification
* Color temperature modification
* Vignette effect
* Gamma Correction
* Histogram Balance

#### Self Driving Car Specs
* Exposure modifications
* Foggy
* Rainy
* Snowy
* RandomShadow

#### Geometric Distortion Nodes
* Rotate
* Warp-affine
* Flip (horizontally or vertically)
* Fish Eye Effect
* Lens correction

#### Other Augmentations
* Resize
* RandomResizeCrop
* Blending images
* Adding Occlusion
* Pixilation
* Adding Noise
* Blurring
* Adding Jitter
* RandomCropLetterBox

### Vision Functions
* Absolute Difference
* Accumulate
* Accumulate Squared
* Accumulate Weighted
* Arithmetic Addition
* Arithmetic Subtraction
* Bilateral Filter
* Bitwise AND
* Bitwise EXCLUSIVE OR
* Bitwise INCLUSIVE OR
* Bitwise NOT
* Box Filter
* Canny Edge Detector
* Channel Combine
* Channel Extract
* Control Flow
* Convert Bit Depth
* Custom Convolution
* Data Object Copy
* Dilate Image
* Equalize Histogram
* Erode Image
* Fast Corners
* Gaussian Filter
* Gaussian Image Pyramid
* Harris Corners
* Histogram
* Integral Image
* LBP
* Laplacian Image Pyramid
* Magnitude
* MatchTemplate
* Max
* Mean and Standard Deviation
* Median Filter
* Min
* Min, Max Location
* Non-Linear Filter
* Non-Maxima Suppression
* Phase
* Pixel-wise Multiplication
* Reconstruction from a Laplacian Image Pyramid
* Remap
* Scale Image
* Sobel 3x3
* TableLookup
* Tensor Add
* Tensor Convert Bit-Depth
* Tensor Matrix Multiply
* Tensor Multiply
* Tensor Subtract
* Tensor TableLookUp
* Tensor Transpose
* Thresholding
* Warp Affine
* Warp Perspective

## Variations
* Support for 3C(RGB) and 1C(Grayscale) images
* Planar and Packed
* Host and GPU 
* Batch Processing with 26 variations
* ROI variations
* Padded Variations

## [Instructions to build the library](#rpp-installation)

```
$ git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git
$ cd rpp
$ mkdir build
$ cd build
$ cmake -DBACKEND=OCL .. #for OCL and HOST
        or
$ cmake -DBACKEND=HIP -DCOMPILE=STATIC #for HIPSTATIC
        or
$ cmake -DBACKEND=HIP -DCOMPILE=HSACOO #for HIPHSACOO
        or
$ cmake -DBACKEND=HIP -DCOMPILE=HIPRTC #for HIPRTC        
$ make -j4
$ sudo make install
```

## MIVisionX(OpenVX) Support
Extended RPP support as a functionality through OpenVX [MIVisionX](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX) (Find build instructions and build the amd_rpp library)

## Miscellaneous
#### RPP stand-alone code snippet using OCL

````
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
````

#### RPP stand-alone code snippet using HOST

```
rppHandle_t handle;
rppCreateWithBatchSize(&handle, noOfImages);
rppi_resize_u8_pkd3_batchDD_host(input, srcSize, output, dstSize, noOfImages, handle);
Rpp32f alpha=2;
Rpp32s beta=1;
    
RppiSize srcSize;
srcSize.height=height;
srcSize.width=width;
rppi_brighten_8u_pln1_gpu( d_a, srcSize, d_c, alpha, beta, theQueue);//device side API call
```

#### RPP stand-alone code snippet using HIP

```
hipMalloc(&d_input, ioBufferSize * sizeof(Rpp8u));
hipMalloc(&d_output, ioBufferSize * sizeof(Rpp8u));
check_hip_error();
hipMemcpy(d_input, input, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
check_hip_error();
Rpp32f alpha=2;
Rpp32s beta=1;

RppiSize srcSize;
srcSize.height=height;
srcSize.width=width;
rppi_brightness_u8_pkd3_gpu(d_input, srcSize[0], d_output, alpha, beta, handle); //device side API call
```

#### RPP with [GDF](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/utilities/runvx/README.md#amd-runvx)(uses OpenVX) code snippet

```
# specify input source for input image and request for displaying input and output images
read input  ../images/face.jpg
view input  inputWindow
view output brightnessWindow

# import RPP library
import vx_rpp
# create input and output images
data input  = image:480,360,RGB2
data output = image:480,360,U008

# compute luma image channel from input RGB image
data yuv  = image-virtual:0,0,IYUV
data luma = image-virtual:0,0,U008
node org.khronos.openvx.color_convert input yuv
node org.khronos.openvx.channel_extract yuv !CHANNEL_Y luma

# compute brightness and contrast in luma image using Brightness function
data alpha = scalar:FLOAT32,1.0  #contrast control
data beta = scalar:INT32,30    #brightness control
node org.rpp.Brightness luma output alpha beta
```