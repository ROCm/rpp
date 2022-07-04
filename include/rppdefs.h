/*
   MulticoreWare Inc.
*/

#ifndef RPPDEFS_H
#define RPPDEFS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#ifdef OCL_COMPILE
#include <CL/cl.h>
#endif

#define RPP_MIN_8U      ( 0 )
#define RPP_MAX_8U      ( 255 )
#define RPP_MIN_16U     ( 0 )
#define RPP_MAX_16U     ( 65535 )




/******************** RPP typedefs ********************/

typedef unsigned char       Rpp8u;
typedef signed char         Rpp8s;
typedef unsigned short      Rpp16u;
typedef short               Rpp16s;
typedef unsigned int        Rpp32u;
typedef int                 Rpp32s;
typedef unsigned long long  Rpp64u;
typedef long long           Rpp64s;
typedef float               Rpp32f;
typedef double              Rpp64f;
typedef void*               RppPtr_t;
typedef size_t              RppSize_t;

typedef enum
{
    RPP_SUCCESS                  = 0,
    RPP_ERROR                    = -1,
    RPP_ERROR_INVALID_ARGUMENTS  = -2,
    RPP_ERROR_LOW_OFFSET         = -3,
    RPP_ERROR_HIGH_SRC_DIMENSION = -5,
    RPP_ERROR_NOT_IMPLEMENTED    = -6
} RppStatus;

typedef enum
{
    rppStatusSuccess        = 0,
    rppStatusBadParm        = -1,
    rppStatusUnknownError   = -2,
    rppStatusNotInitialized = -3,
    rppStatusInvalidValue   = -4,
    rppStatusAllocFailed    = -5,
    rppStatusInternalError  = -6,
    rppStatusNotImplemented = -7,
    rppStatusUnsupportedOp  = -8,
} rppStatus_t;

typedef enum
{
    RPPI_HORIZONTAL_AXIS,
    RPPI_VERTICAL_AXIS,
    RPPI_BOTH_AXIS
} RppiAxis;

typedef enum
{
    RPP_SCALAR_OP_AND       = 1,
    RPP_SCALAR_OP_OR,
    RPP_SCALAR_OP_XOR,
    RPP_SCALAR_OP_NAND,
    RPP_SCALAR_OP_EQUAL,
    RPP_SCALAR_OP_NOTEQUAL,
    RPP_SCALAR_OP_LESS,
    RPP_SCALAR_OP_LESSEQ,
    RPP_SCALAR_OP_GREATER,
    RPP_SCALAR_OP_GREATEREQ,
    RPP_SCALAR_OP_ADD,
    RPP_SCALAR_OP_SUBTRACT,
    RPP_SCALAR_OP_MULTIPLY,
    RPP_SCALAR_OP_DIVIDE,
    RPP_SCALAR_OP_MODULUS,
    RPP_SCALAR_OP_MIN,
    RPP_SCALAR_OP_MAX,
} RppOp;

typedef enum
{
    U8_S8,
    S8_U8,
} RppConvertBitDepthMode;

typedef struct
{
    Rpp32f rho;
    Rpp32f theta;
} RppPointPolar;

typedef struct
{
    Rpp32u channelParam;
    Rpp32u bufferMultiplier;
} RppLayoutParams;

typedef struct
{
    Rpp32f data[6];
} Rpp32f6;





/******************** RPPI typedefs ********************/

typedef enum
{
    RGB_HSV                 = 1,
    HSV_RGB
} RppiColorConvertMode;

typedef enum
{
    RPPI_LOW,
    RPPI_MEDIUM,
    RPPI_HIGH
} RppiFuzzyLevel;

typedef enum
{
    RPPI_CHN_PLANAR,
    RPPI_CHN_PACKED
} RppiChnFormat;

typedef struct
{
    unsigned int width;
    unsigned int height;
} RppiSize;

typedef struct
{
    int x;
    int y;
} RppiPoint;

typedef struct
{
    int x;
    int y;
    int width;
    int height;
} RppiRect;

typedef struct
{
    unsigned int x;
    unsigned int y;
    unsigned int roiWidth;
    unsigned int roiHeight;
} RppiROI;

typedef enum
{
    GAUSS3,
    GAUSS5,
    GAUSS3x1,
    GAUSS1x3,
    AVG3 = 10,
    AVG5
} RppiBlur;

typedef enum
{
    ZEROPAD,
    NOPAD
} RppiPad;

typedef enum
{
    RGB,
    HSV
} RppiFormat;





/******************** RPPT typedefs ********************/

typedef enum
{
    U8,
    F32,
    F16,
    I8
} RpptDataType;

typedef enum
{
    NCHW,
    NHWC
} RpptLayout;

typedef enum
{
    LTRB,
    XYWH

} RpptRoiType;

typedef enum
{
    NEAREST_NEIGHBOR = 0,
    BILINEAR,
    BICUBIC,
    LANCZOS,
    GAUSSIAN,
    TRIANGULAR
} RpptInterpolationType;

typedef struct
{
    RppiPoint lt, rb;

} RpptRoiLtrb;

typedef struct
{
    RppiPoint xy;
    int roiWidth, roiHeight;

} RpptRoiXywh;

typedef union
{
    RpptRoiLtrb ltrbROI;
    RpptRoiXywh xywhROI;

} RpptROI, *RpptROIPtr;

typedef struct
{
    Rpp32u nStride;
    Rpp32u cStride;
    Rpp32u hStride;
    Rpp32u wStride;
} RpptStrides;

typedef struct
{
    RppSize_t numDims;
    Rpp32u offsetInBytes;
    RpptDataType dataType;
    RpptLayout layout;
    Rpp32u n, c, h, w;
    RpptStrides strides;
} RpptDesc, *RpptDescPtr;

typedef struct
{
    Rpp8u R;
    Rpp8u G;
    Rpp8u B;
} RpptRGB;

typedef struct
{
    Rpp32f R;
    Rpp32f G;
    Rpp32f B;
} RpptFloatRGB;

typedef struct
{
    Rpp32u x;
    Rpp32u y;
} RpptUintVector2D;

typedef struct
{
    Rpp32f x;
    Rpp32f y;
} RpptFloatVector2D;

typedef struct
{
    Rpp32u width;
    Rpp32u height;
} RpptImagePatch, *RpptImagePatchPtr;

typedef struct
{   uint x[5];
    uint counter;
} RpptXorwowState;



/******************** HOST memory typedefs ********************/

typedef struct
{
    Rpp64f *doublemem;
} memRpp64f;

typedef struct
{
    Rpp32f *floatmem;
} memRpp32f;

typedef struct
{
    Rpp32u *uintmem;
} memRpp32u;

typedef struct
{
    Rpp32s *intmem;
} memRpp32s;

typedef struct
{
    Rpp8u *ucharmem;
} memRpp8u;

typedef struct
{
    Rpp8s *charmem;
} memRpp8s;

typedef struct
{
    RpptRGB* rgbmem;
} memRpptRGB;

typedef struct
{
    Rpp32u *height;
    Rpp32u *width;
} memSize;

typedef struct
{
    Rpp32u *x;
    Rpp32u *y;
    Rpp32u *roiHeight;
    Rpp32u *roiWidth;
} memROI;

typedef struct {
    RppiSize *srcSize;
    RppiSize *dstSize;
    RppiSize *maxSrcSize;
    RppiSize *maxDstSize;
    RppiROI *roiPoints;
    memRpp32f floatArr[10];
    memRpp64f doubleArr[10];
    memRpp32u uintArr[10];
    memRpp32s intArr[10];
    memRpp8u ucharArr[10];
    memRpp8s charArr[10];
    memRpptRGB rgbArr;
    Rpp64u *srcBatchIndex;
    Rpp64u *dstBatchIndex;
    Rpp32u *inc;
    Rpp32u *dstInc;
} memCPU;





#ifdef OCL_COMPILE

/******************** OCL memory typedefs ********************/

typedef struct
{
    cl_mem floatmem;
} clmemRpp32f;


typedef struct
{
    cl_mem doublemem;
} clmemRpp64f;

typedef struct
{
    cl_mem uintmem;
} clmemRpp32u;

typedef struct
{
    cl_mem intmem;
} clmemRpp32s;

typedef struct
{
    cl_mem ucharmem;
} clmemRpp8u;

typedef struct
{
    cl_mem charmem;
} clmemRpp8s;

typedef struct
{
    cl_mem height;
    cl_mem width;
} clmemSize;

typedef struct
{
    cl_mem x;
    cl_mem y;
    cl_mem roiHeight;
    cl_mem roiWidth;
} clmemROI;

typedef struct
{
    memSize csrcSize;
    memSize cdstSize;
    memSize cmaxSrcSize;
    memSize cmaxDstSize;
    memROI croiPoints;
    clmemSize srcSize;
    clmemSize dstSize;
    clmemSize maxSrcSize;
    clmemSize maxDstSize;
    clmemROI roiPoints;
    clmemRpp32f floatArr[10];
    clmemRpp64f doubleArr[10];
    clmemRpp32u uintArr[10];
    clmemRpp32s intArr[10];
    clmemRpp8u ucharArr[10];
    clmemRpp8s charArr[10];
    cl_mem srcBatchIndex;
    cl_mem dstBatchIndex;
    cl_mem inc;
    cl_mem dstInc;
} memGPU;

typedef struct
{
    memCPU mcpu;
    memGPU mgpu;
} memMgmt;

typedef struct
{
    RppPtr_t cpuHandle;
    Rpp32u nbatchSize;
    memMgmt mem;
} InitHandle;





#elif defined(HIP_COMPILE)

/******************** HIP memory typedefs ********************/

typedef struct
{
    Rpp32f* floatmem;
} hipMemRpp32f;

typedef struct
{
    Rpp64f* doublemem;
} hipMemRpp64f;

typedef struct
{
    Rpp32u* uintmem;
} hipMemRpp32u;

typedef struct
{
    Rpp32s* intmem;
} hipMemRpp32s;

typedef struct
{
    Rpp8u* ucharmem;
} hipMemRpp8u;

typedef struct
{
    Rpp8s* charmem;
} hipMemRpp8s;

typedef struct
{
    RpptRGB* rgbmem;
} hipMemRpptRGB;

typedef struct
{
    Rpp32u* height;
    Rpp32u* width;
} hipMemSize;

typedef struct
{
    Rpp32u* x;
    Rpp32u* y;
    Rpp32u* roiHeight;
    Rpp32u* roiWidth;
} hipMemROI;

typedef struct
{
    memSize csrcSize;
    memSize cdstSize;
    memSize cmaxSrcSize;
    memSize cmaxDstSize;
    memROI croiPoints;
    hipMemSize srcSize;
    hipMemSize dstSize;
    hipMemSize maxSrcSize;
    hipMemSize maxDstSize;
    hipMemROI roiPoints;
    hipMemRpp32f floatArr[10];
    hipMemRpp32f float3Arr[10];
    hipMemRpp64f doubleArr[10];
    hipMemRpp32u uintArr[10];
    hipMemRpp32s intArr[10];
    hipMemRpp8u ucharArr[10];
    hipMemRpp8s charArr[10];
    hipMemRpptRGB rgbArr;
    hipMemRpp32f maskArr;
    Rpp64u* srcBatchIndex;
    Rpp64u* dstBatchIndex;
    Rpp32u* inc;
    Rpp32u* dstInc;
} memGPU;

typedef struct
{
    memCPU mcpu;
    memGPU mgpu;
} memMgmt;

typedef struct
{
    RppPtr_t cpuHandle;
    Rpp32u nbatchSize;
    memMgmt mem;
} InitHandle;

#endif //BACKEND

#ifdef __cplusplus
}
#endif
#endif /* RPPDEFS_H */
