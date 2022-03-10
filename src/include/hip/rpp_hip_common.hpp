#ifndef RPP_HIP_COMMON_H
#define RPP_HIP_COMMON_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <half.hpp>
#include "rppdefs.h"
#include "hip/rpp/handle.hpp"
#include "hip/rpp_hip_roi_conversion.hpp"
using halfhpp = half_float::half;
typedef halfhpp Rpp16f;

// float
typedef struct d_float6
{
    float2 x;
    float2 y;
    float2 z;
} d_float6;
typedef struct d_float8
{
    float4 x;
    float4 y;
} d_float8;
typedef struct d_float12
{
    float4 x;
    float4 y;
    float4 z;
} d_float12;
typedef struct d_float16
{
    d_float8 x;
    d_float8 y;
} d_float16;
typedef struct d_float24
{
    d_float8 x;
    d_float8 y;
    d_float8 z;
} d_float24;
typedef struct d_float6_as_float3s
{
    float3 x;
    float3 y;
} d_float6_as_float3s;
typedef struct d_float12_as_float3s
{
    float3 x;
    float3 y;
    float3 z;
    float3 w;

} d_float12_as_float3s;
typedef struct d_float24_as_float3s
{
    d_float12_as_float3s x;
    d_float12_as_float3s y;

} d_float24_as_float3s;

// uint
typedef struct d_uint6
{
    uint2 x;
    uint2 y;
    uint2 z;
} d_uint6;

// int
typedef struct d_int4
{
    int2 x;
    int2 y;
} d_int4;
typedef struct d_int6
{
    int2 x;
    int2 y;
    int2 z;
} d_int6;

// half
typedef struct d_half3
{
    half x;
    half y;
    half z;
} d_half3;
typedef struct d_half4
{
    half2 x;
    half2 y;
} d_half4;
typedef struct d_half6
{
    half2 x;
    half2 y;
    half2 z;
} d_half6;
typedef struct d_half8
{
    d_half4 x;
    d_half4 y;
} d_half8;
typedef struct d_half24
{
    d_half8 x;
    d_half8 y;
    d_half8 z;
} d_half24;
typedef struct d_half4_as_halfs
{
    half x;
    half y;
    half z;
    half w;
} d_half4_as_halfs;
typedef struct d_half8_as_halfs
{
    d_half4_as_halfs x;
    d_half4_as_halfs y;
} d_half8_as_halfs;
typedef struct d_half24_as_halfs
{
    d_half8_as_halfs x;
    d_half8_as_halfs y;
    d_half8_as_halfs z;
} d_half24_as_halfs;
typedef struct d_half12_as_half3s
{
    d_half3 x;
    d_half3 y;
    d_half3 z;
    d_half3 w;

} d_half12_as_half3s;
typedef struct d_half24_as_half3s
{
    d_half12_as_half3s x;
    d_half12_as_half3s y;

} d_half24_as_half3s;

// uchar
typedef unsigned char uchar;
typedef struct d_uchar8
{
    uchar4 x;
    uchar4 y;
} d_uchar8;
typedef struct d_uchar24
{
    d_uchar8 x;
    d_uchar8 y;
    d_uchar8 z;
} d_uchar24;
typedef struct d_uchar12_as_uchar3s
{
    uchar3 x;
    uchar3 y;
    uchar3 z;
    uchar3 w;

} d_uchar12_as_uchar3s;
typedef struct d_uchar24_as_uchar3s
{
    d_uchar12_as_uchar3s x;
    d_uchar12_as_uchar3s y;

} d_uchar24_as_uchar3s;

// schar
typedef signed char schar;
typedef struct d_schar3
{
    schar x;
    schar y;
    schar z;
} d_schar3;
typedef struct d_schar4
{
    schar x;
    schar y;
    schar z;
    schar w;
} d_schar4;
typedef struct d_schar8
{
    d_schar4 x;
    d_schar4 y;
} d_schar8;
typedef struct d_schar24
{
    d_schar8 x;
    d_schar8 y;
    d_schar8 z;
} d_schar24;
typedef struct d_schar12_as_schar3s
{
    d_schar3 x;
    d_schar3 y;
    d_schar3 z;
    d_schar3 w;

} d_schar12_as_schar3s;
typedef struct d_schar24_as_schar3s
{
    d_schar12_as_schar3s x;
    d_schar12_as_schar3s y;

} d_schar24_as_schar3s;

enum class RPPTensorDataType
{
    U8 = 0,
    FP32,
    FP16,
    I8,
};

struct RPPTensorFunctionMetaData
{
    RPPTensorDataType _in_type = RPPTensorDataType::U8;
    RPPTensorDataType _out_type = RPPTensorDataType::U8;
    RppiChnFormat _in_format = RppiChnFormat::RPPI_CHN_PACKED;
    RppiChnFormat _out_format = RppiChnFormat::RPPI_CHN_PLANAR;
    Rpp32u _in_channels = 3;

    RPPTensorFunctionMetaData(RppiChnFormat in_chn_format, RPPTensorDataType in_tensor_type,
                              RPPTensorDataType out_tensor_type, Rpp32u in_channels,
                              bool out_format_change) : _in_format(in_chn_format), _in_type(in_tensor_type),
                                                        _out_type(out_tensor_type), _in_channels(in_channels)
    {
        if (out_format_change)
        {
            if (_in_format == RPPI_CHN_PLANAR)
                _out_format = RppiChnFormat::RPPI_CHN_PACKED;
            else
                _out_format = RppiChnFormat::RPPI_CHN_PLANAR;
        }
        else
            _out_format = _in_format;
    }
};

#define LOCAL_THREADS_X 16
#define LOCAL_THREADS_Y 16
#define LOCAL_THREADS_Z 1
#define ONE_OVER_255 0.00392157f
#define SIX_OVER_360 0.01666667f
#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0')

/******************** HOST FUNCTIONS ********************/

inline int getplnpkdind(RppiChnFormat &format)
{
    return format == RPPI_CHN_PLANAR ? 1 : 3;
}

inline RppStatus generate_gaussian_kernel_gpu(Rpp32f stdDev, Rpp32f* kernel, Rpp32u kernelSize)
{
    Rpp32f s, sum = 0.0, multiplier;
    int bound = ((kernelSize - 1) / 2);
    Rpp32u c = 0;
    s = 1 / (2 * stdDev * stdDev);
    multiplier = (1 / M_PI) * (s);
    for (int i = -bound; i <= bound; i++)
    {
        for (int j = -bound; j <= bound; j++)
        {
            kernel[c] = multiplier * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        kernel[i] /= sum;
    }

    return RPP_SUCCESS;
}

/******************** DEVICE FUNCTIONS ********************/

// -------------------- Set 0 - Range checks and Range adjustment --------------------

// float4 pixel check for 0-255 range

__device__ __forceinline__ float4 rpp_hip_pixel_check_0to255(float4 src_f4)
{
    return make_float4(fminf(fmaxf(src_f4.x, 0), 255),
                       fminf(fmaxf(src_f4.y, 0), 255),
                       fminf(fmaxf(src_f4.z, 0), 255),
                       fminf(fmaxf(src_f4.w, 0), 255));
}

// float4 pixel check for 0-1 range

__device__ __forceinline__ float4 rpp_hip_pixel_check_0to1(float4 src_f4)
{
    return make_float4(fminf(fmaxf(src_f4.x, 0), 1),
                       fminf(fmaxf(src_f4.y, 0), 1),
                       fminf(fmaxf(src_f4.z, 0), 1),
                       fminf(fmaxf(src_f4.w, 0), 1));
}

// d_float24 pixel check for 0-255 range

__device__ __forceinline__ void rpp_hip_pixel_check_0to255(d_float24 *pix_f24)
{
    pix_f24->x.x = rpp_hip_pixel_check_0to255(pix_f24->x.x);
    pix_f24->x.y = rpp_hip_pixel_check_0to255(pix_f24->x.y);
    pix_f24->y.x = rpp_hip_pixel_check_0to255(pix_f24->y.x);
    pix_f24->y.y = rpp_hip_pixel_check_0to255(pix_f24->y.y);
    pix_f24->z.x = rpp_hip_pixel_check_0to255(pix_f24->z.x);
    pix_f24->z.y = rpp_hip_pixel_check_0to255(pix_f24->z.y);
}

// d_float24 pixel check for 0-1 range

__device__ __forceinline__ void rpp_hip_pixel_check_0to1(d_float24 *pix_f24)
{
    pix_f24->x.x = rpp_hip_pixel_check_0to1(pix_f24->x.x);
    pix_f24->x.y = rpp_hip_pixel_check_0to1(pix_f24->x.y);
    pix_f24->y.x = rpp_hip_pixel_check_0to1(pix_f24->y.x);
    pix_f24->y.y = rpp_hip_pixel_check_0to1(pix_f24->y.y);
    pix_f24->z.x = rpp_hip_pixel_check_0to1(pix_f24->z.x);
    pix_f24->z.y = rpp_hip_pixel_check_0to1(pix_f24->z.y);
}

// d_float8 adjust pixel range for different bit depths

__device__ __forceinline__ void rpp_hip_adjust_range(uchar *dstPtr, d_float8 *sum_f8){}

__device__ __forceinline__ void rpp_hip_adjust_range(float *dstPtr, d_float8 *sum_f8)
{
    sum_f8->x = sum_f8->x * (float4) ONE_OVER_255;
    sum_f8->y = sum_f8->y * (float4) ONE_OVER_255;
}

__device__ __forceinline__ void rpp_hip_adjust_range(schar *dstPtr, d_float8 *sum_f8)
{
    sum_f8->x = sum_f8->x - (float4) 128;
    sum_f8->y = sum_f8->y - (float4) 128;
}

__device__ __forceinline__ void rpp_hip_adjust_range(half *dstPtr, d_float8 *sum_f8)
{
    sum_f8->x = sum_f8->x * (float4) ONE_OVER_255;
    sum_f8->y = sum_f8->y * (float4) ONE_OVER_255;
}

// d_float24 adjust pixel range for different bit depths

__device__ __forceinline__ void rpp_hip_adjust_range(uchar *dstPtr, d_float24 *sum_f24)
{
}

__device__ __forceinline__ void rpp_hip_adjust_range(float *dstPtr, d_float24 *sum_f24)
{
    sum_f24->x.x = sum_f24->x.x * (float4) ONE_OVER_255;
    sum_f24->x.y = sum_f24->x.y * (float4) ONE_OVER_255;
    sum_f24->y.x = sum_f24->y.x * (float4) ONE_OVER_255;
    sum_f24->y.y = sum_f24->y.y * (float4) ONE_OVER_255;
    sum_f24->z.x = sum_f24->z.x * (float4) ONE_OVER_255;
    sum_f24->z.y = sum_f24->z.y * (float4) ONE_OVER_255;
}

__device__ __forceinline__ void rpp_hip_adjust_range(schar *dstPtr, d_float24 *sum_f24)
{
    sum_f24->x.x = sum_f24->x.x - (float4) 128;
    sum_f24->x.y = sum_f24->x.y - (float4) 128;
    sum_f24->y.x = sum_f24->y.x - (float4) 128;
    sum_f24->y.y = sum_f24->y.y - (float4) 128;
    sum_f24->z.x = sum_f24->z.x - (float4) 128;
    sum_f24->z.y = sum_f24->z.y - (float4) 128;
}

__device__ __forceinline__ void rpp_hip_adjust_range(half *dstPtr, d_float24 *sum_f24)
{
    sum_f24->x.x = sum_f24->x.x * (float4) ONE_OVER_255;
    sum_f24->x.y = sum_f24->x.y * (float4) ONE_OVER_255;
    sum_f24->y.x = sum_f24->y.x * (float4) ONE_OVER_255;
    sum_f24->y.y = sum_f24->y.y * (float4) ONE_OVER_255;
    sum_f24->z.x = sum_f24->z.x * (float4) ONE_OVER_255;
    sum_f24->z.y = sum_f24->z.y * (float4) ONE_OVER_255;
}

// -------------------- Set 1 - Packing --------------------

// Packing to U8s

__device__ __forceinline__ uint rpp_hip_pack(float4 src)
{
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

// Packing to I8s

__device__ __forceinline__ uint rpp_hip_pack_i8(float4 src)
{
    char4 dst_c4;
    dst_c4.w = (schar)(src.w);
    dst_c4.z = (schar)(src.z);
    dst_c4.y = (schar)(src.y);
    dst_c4.x = (schar)(src.x);

    return *(uint *)&dst_c4;
}

// -------------------- Set 2 - Un-Packing --------------------

// Un-Packing from U8s

__device__ __forceinline__ float rpp_hip_unpack0(uint src)
{
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack1(uint src)
{
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack2(uint src)
{
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack3(uint src)
{
    return (float)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float4 rpp_hip_unpack(uint src)
{
    return make_float4(rpp_hip_unpack0(src), rpp_hip_unpack1(src), rpp_hip_unpack2(src), rpp_hip_unpack3(src));
}

__device__ __forceinline__ float4 rpp_hip_unpack_mirror(uint src)
{
    return make_float4(rpp_hip_unpack3(src), rpp_hip_unpack2(src), rpp_hip_unpack1(src), rpp_hip_unpack0(src));
}

// Un-Packing from I8s

__device__ __forceinline__ float rpp_hip_unpack0(int src)
{
    return (float)(schar)(src & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack1(int src)
{
    return (float)(schar)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack2(int src)
{
    return (float)(schar)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack3(int src)
{
    return (float)(schar)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float4 rpp_hip_unpack_from_i8(int src)
{
    return make_float4(rpp_hip_unpack0(src), rpp_hip_unpack1(src), rpp_hip_unpack2(src), rpp_hip_unpack3(src));
}

__device__ __forceinline__ float4 rpp_hip_unpack_from_i8_mirror(int src)
{
    return make_float4(rpp_hip_unpack3(src), rpp_hip_unpack2(src), rpp_hip_unpack1(src), rpp_hip_unpack0(src));
}

// Un-Packing from F32s

__device__ __forceinline__ float4 rpp_hip_unpack_mirror(float4 src)
{
    return make_float4(src.w, src.z, src.y, src.x);
}

// -------------------- Set 3 - Bit Depth Conversions --------------------

// I8 to U8 conversions (8 pixels)

__device__ __forceinline__ void rpp_hip_convert8_i8_to_u8(schar *srcPtr, uchar *dstPtr)
{
    int2 *srcPtr_8;
    srcPtr_8 = (int2 *)srcPtr;

    uint2 *dstPtr_8;
    dstPtr_8 = (uint2 *)dstPtr;

    dstPtr_8->x = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_8->x) + (float4) 128);
    dstPtr_8->y = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_8->y) + (float4) 128);
}

// I8 to U8 conversions (24 pixels)

__device__ __forceinline__ void rpp_hip_convert24_i8_to_u8(schar *srcPtr, uchar *dstPtr)
{
    d_int6 *srcPtr_24;
    srcPtr_24 = (d_int6 *)srcPtr;

    d_uint6 *dstPtr_24;
    dstPtr_24 = (d_uint6 *)dstPtr;

    dstPtr_24->x.x = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_24->x.x) + (float4) 128);
    dstPtr_24->x.y = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_24->x.y) + (float4) 128);
    dstPtr_24->y.x = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_24->y.x) + (float4) 128);
    dstPtr_24->y.y = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_24->y.y) + (float4) 128);
    dstPtr_24->z.x = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_24->z.x) + (float4) 128);
    dstPtr_24->z.y = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_24->z.y) + (float4) 128);
}

// -------------------- Set 4 - Loads to float --------------------

// WITHOUT LAYOUT TOGGLE

// U8 loads without layout toggle (8 U8 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(uchar *srcPtr, d_float8 *src_f8)
{
    uint2 src = *((uint2 *)(srcPtr));
    src_f8->x = rpp_hip_unpack(src.x);
    src_f8->y = rpp_hip_unpack(src.y);
}

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8_mirror(uchar *srcPtr, d_float8 *src_f8)
{
    uint2 src = *((uint2 *)(srcPtr));
    src_f8->x = rpp_hip_unpack_mirror(src.y);
    src_f8->y = rpp_hip_unpack_mirror(src.x);
}

// F32 loads without layout toggle (8 F32 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(float *srcPtr, d_float8 *src_f8)
{
    *src_f8 = *((d_float8 *)(srcPtr));
}

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8_mirror(float *srcPtr, d_float8 *src_f8)
{
    d_float8 src;
    src = *((d_float8 *)(srcPtr));
    src_f8->x = rpp_hip_unpack_mirror(src.y);
    src_f8->y = rpp_hip_unpack_mirror(src.x);
}

// I8 loads without layout toggle (8 I8 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(schar *srcPtr, d_float8 *src_f8)
{
    int2 src = *((int2 *)(srcPtr));
    src_f8->x = rpp_hip_unpack_from_i8(src.x);
    src_f8->y = rpp_hip_unpack_from_i8(src.y);
}

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8_mirror(schar *srcPtr, d_float8 *src_f8)
{
    int2 src = *((int2 *)(srcPtr));
    src_f8->x = rpp_hip_unpack_from_i8_mirror(src.y);
    src_f8->y = rpp_hip_unpack_from_i8_mirror(src.x);
}

// F16 loads without layout toggle (8 F16 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(half *srcPtr, d_float8 *src_f8)
{
    d_half8 src_h8;
    src_h8 = *((d_half8 *)(srcPtr));

    float2 src1_f2, src2_f2;

    src1_f2 = __half22float2(src_h8.x.x);
    src2_f2 = __half22float2(src_h8.x.y);
    src_f8->x = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);

    src1_f2 = __half22float2(src_h8.y.x);
    src2_f2 = __half22float2(src_h8.y.y);
    src_f8->y = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);
}

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8_mirror(half *srcPtr, d_float8 *src_f8)
{
    d_half8 src_h8;
    src_h8 = *((d_half8 *)(srcPtr));

    float2 src1_f2, src2_f2;

    src1_f2 = __half22float2(src_h8.x.x);
    src2_f2 = __half22float2(src_h8.x.y);
    src_f8->y = rpp_hip_unpack_mirror(make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y));

    src1_f2 = __half22float2(src_h8.y.x);
    src2_f2 = __half22float2(src_h8.y.y);
    src_f8->x = rpp_hip_unpack_mirror(make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y));
}

// U8 loads without layout toggle PLN3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3(uchar *srcPtr, uint increment, d_float24 *src_f24)
{
    d_uint6 src;
    uchar *srcTempPtr = srcPtr;

    src.x = *((uint2 *)(srcTempPtr));
    srcTempPtr += increment;
    src.y = *((uint2 *)(srcTempPtr));
    srcTempPtr += increment;
    src.z = *((uint2 *)(srcTempPtr));

    src_f24->x.x = rpp_hip_unpack(src.x.x);    // write R00-R03
    src_f24->x.y = rpp_hip_unpack(src.x.y);    // write R04-R07
    src_f24->y.x = rpp_hip_unpack(src.y.x);    // write G00-G03
    src_f24->y.y = rpp_hip_unpack(src.y.y);    // write G04-G07
    src_f24->z.x = rpp_hip_unpack(src.z.x);    // write B00-B03
    src_f24->z.y = rpp_hip_unpack(src.z.y);    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3_mirror(uchar *srcPtr, uint increment, d_float24 *src_f24)
{
    d_uint6 src;
    uchar *srcTempPtr = srcPtr;

    src.x = *((uint2 *)(srcTempPtr));
    srcTempPtr += increment;
    src.y = *((uint2 *)(srcTempPtr));
    srcTempPtr += increment;
    src.z = *((uint2 *)(srcTempPtr));

    src_f24->x.x = rpp_hip_unpack_mirror(src.x.y);    // write R07-R04 (mirrored load)
    src_f24->x.y = rpp_hip_unpack_mirror(src.x.x);    // write R03-R00 (mirrored load)
    src_f24->y.x = rpp_hip_unpack_mirror(src.y.y);    // write G07-G04 (mirrored load)
    src_f24->y.y = rpp_hip_unpack_mirror(src.y.x);    // write G03-G00 (mirrored load)
    src_f24->z.x = rpp_hip_unpack_mirror(src.z.y);    // write B07-B04 (mirrored load)
    src_f24->z.y = rpp_hip_unpack_mirror(src.z.x);    // write B03-B00 (mirrored load)
}

// F32 loads without layout toggle PLN3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3(float *srcPtr, uint increment, d_float24 *src_f24)
{
    float *srcPtrR, *srcPtrG, *srcPtrB;
    srcPtrR = srcPtr;
    srcPtrG = srcPtrR + increment;
    srcPtrB = srcPtrG + increment;

    src_f24->x = *(d_float8 *)srcPtrR;    // write R00-R07
    src_f24->y = *(d_float8 *)srcPtrG;    // write G00-G07
    src_f24->z = *(d_float8 *)srcPtrB;    // write B00-B07
}

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3_mirror(float *srcPtr, uint increment, d_float24 *src_f24)
{
    d_float24 src;
    float *srcPtrR, *srcPtrG, *srcPtrB;
    srcPtrR = srcPtr;
    srcPtrG = srcPtrR + increment;
    srcPtrB = srcPtrG + increment;

    src.x = *(d_float8 *)srcPtrR;
    src.y = *(d_float8 *)srcPtrG;
    src.z = *(d_float8 *)srcPtrB;

    src_f24->x.x = rpp_hip_unpack_mirror(src.x.y);    // write R07-R04 (mirrored load)
    src_f24->x.y = rpp_hip_unpack_mirror(src.x.x);    // write R03-R00 (mirrored load)
    src_f24->y.x = rpp_hip_unpack_mirror(src.y.y);    // write G07-G04 (mirrored load)
    src_f24->y.y = rpp_hip_unpack_mirror(src.y.x);    // write G03-G00 (mirrored load)
    src_f24->z.x = rpp_hip_unpack_mirror(src.z.y);    // write B07-B04 (mirrored load)
    src_f24->z.y = rpp_hip_unpack_mirror(src.z.x);    // write B03-B00 (mirrored load)
}

// I8 loads without layout toggle PLN3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3(schar *srcPtr, uint increment, d_float24 *src_f24)
{
    d_int6 src;
    schar *srcTempPtr = srcPtr;

    src.x = *((int2 *)(srcTempPtr));
    srcTempPtr += increment;
    src.y = *((int2 *)(srcTempPtr));
    srcTempPtr += increment;
    src.z = *((int2 *)(srcTempPtr));

    src_f24->x.x = rpp_hip_unpack_from_i8(src.x.x);    // write R00-R03
    src_f24->x.y = rpp_hip_unpack_from_i8(src.x.y);    // write R04-R07
    src_f24->y.x = rpp_hip_unpack_from_i8(src.y.x);    // write G00-G03
    src_f24->y.y = rpp_hip_unpack_from_i8(src.y.y);    // write G04-G07
    src_f24->z.x = rpp_hip_unpack_from_i8(src.z.x);    // write B00-B03
    src_f24->z.y = rpp_hip_unpack_from_i8(src.z.y);    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3_mirror(schar *srcPtr, uint increment, d_float24 *src_f24)
{
    d_int6 src;
    schar *srcTempPtr = srcPtr;

    src.x = *((int2 *)(srcTempPtr));
    srcTempPtr += increment;
    src.y = *((int2 *)(srcTempPtr));
    srcTempPtr += increment;
    src.z = *((int2 *)(srcTempPtr));

    src_f24->x.x = rpp_hip_unpack_from_i8_mirror(src.x.y);    // write R07-R04 (mirrored load)
    src_f24->x.y = rpp_hip_unpack_from_i8_mirror(src.x.x);    // write R03-R00 (mirrored load)
    src_f24->y.x = rpp_hip_unpack_from_i8_mirror(src.y.y);    // write G07-G04 (mirrored load)
    src_f24->y.y = rpp_hip_unpack_from_i8_mirror(src.y.x);    // write G03-G00 (mirrored load)
    src_f24->z.x = rpp_hip_unpack_from_i8_mirror(src.z.y);    // write B07-B04 (mirrored load)
    src_f24->z.y = rpp_hip_unpack_from_i8_mirror(src.z.x);    // write B03-B00 (mirrored load)
}

// F16 loads without layout toggle PLN3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3(half *srcPtr, uint increment, d_float24 *src_f24)
{
    half *srcPtrR, *srcPtrG, *srcPtrB;
    srcPtrR = srcPtr;
    srcPtrG = srcPtrR + increment;
    srcPtrB = srcPtrG + increment;

    d_half8 *srcR_h8, *srcG_h8, *srcB_h8;
    srcR_h8 = (d_half8 *)srcPtrR;
    srcG_h8 = (d_half8 *)srcPtrG;
    srcB_h8 = (d_half8 *)srcPtrB;

    src_f24->x.x = make_float4(__half2float(__low2half(srcR_h8->x.x)), __half2float(__high2half(srcR_h8->x.x)), __half2float(__low2half(srcR_h8->x.y)), __half2float(__high2half(srcR_h8->x.y)));    // write R00-R03
    src_f24->x.y = make_float4(__half2float(__low2half(srcR_h8->y.x)), __half2float(__high2half(srcR_h8->y.x)), __half2float(__low2half(srcR_h8->y.y)), __half2float(__high2half(srcR_h8->y.y)));    // write R04-R07
    src_f24->y.x = make_float4(__half2float(__low2half(srcG_h8->x.x)), __half2float(__high2half(srcG_h8->x.x)), __half2float(__low2half(srcG_h8->x.y)), __half2float(__high2half(srcG_h8->x.y)));    // write G00-G03
    src_f24->y.y = make_float4(__half2float(__low2half(srcG_h8->y.x)), __half2float(__high2half(srcG_h8->y.x)), __half2float(__low2half(srcG_h8->y.y)), __half2float(__high2half(srcG_h8->y.y)));    // write G04-G07
    src_f24->z.x = make_float4(__half2float(__low2half(srcB_h8->x.x)), __half2float(__high2half(srcB_h8->x.x)), __half2float(__low2half(srcB_h8->x.y)), __half2float(__high2half(srcB_h8->x.y)));    // write B00-B03
    src_f24->z.y = make_float4(__half2float(__low2half(srcB_h8->y.x)), __half2float(__high2half(srcB_h8->y.x)), __half2float(__low2half(srcB_h8->y.y)), __half2float(__high2half(srcB_h8->y.y)));    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3_mirror(half *srcPtr, uint increment, d_float24 *src_f24)
{
    half *srcPtrR, *srcPtrG, *srcPtrB;
    srcPtrR = srcPtr;
    srcPtrG = srcPtrR + increment;
    srcPtrB = srcPtrG + increment;

    d_half8 *srcR_h8, *srcG_h8, *srcB_h8;
    srcR_h8 = (d_half8 *)srcPtrR;
    srcG_h8 = (d_half8 *)srcPtrG;
    srcB_h8 = (d_half8 *)srcPtrB;

    src_f24->x.x = make_float4(__half2float(__high2half(srcR_h8->y.y)), __half2float(__low2half(srcR_h8->y.y)), __half2float(__high2half(srcR_h8->y.x)), __half2float(__low2half(srcR_h8->y.x)));    // write R07-R04 (mirrored load)
    src_f24->x.y = make_float4(__half2float(__high2half(srcR_h8->x.y)), __half2float(__low2half(srcR_h8->x.y)), __half2float(__high2half(srcR_h8->x.x)), __half2float(__low2half(srcR_h8->x.x)));    // write R03-R00 (mirrored load)
    src_f24->y.x = make_float4(__half2float(__high2half(srcG_h8->y.y)), __half2float(__low2half(srcG_h8->y.y)), __half2float(__high2half(srcG_h8->y.x)), __half2float(__low2half(srcG_h8->y.x)));    // write G07-G04 (mirrored load)
    src_f24->y.y = make_float4(__half2float(__high2half(srcG_h8->x.y)), __half2float(__low2half(srcG_h8->x.y)), __half2float(__high2half(srcG_h8->x.x)), __half2float(__low2half(srcG_h8->x.x)));    // write G03-G00 (mirrored load)
    src_f24->z.x = make_float4(__half2float(__high2half(srcB_h8->y.y)), __half2float(__low2half(srcB_h8->y.y)), __half2float(__high2half(srcB_h8->y.x)), __half2float(__low2half(srcB_h8->y.x)));    // write B07-B04 (mirrored load)
    src_f24->z.y = make_float4(__half2float(__high2half(srcB_h8->x.y)), __half2float(__low2half(srcB_h8->x.y)), __half2float(__high2half(srcB_h8->x.x)), __half2float(__low2half(srcB_h8->x.x)));    // write B03-B00 (mirrored load)
}

// WITH LAYOUT TOGGLE

// U8 loads with layout toggle PKD3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(uchar *srcPtr, d_float24 *src_f24)
{
    d_uint6 src = *((d_uint6 *)(srcPtr));

    src_f24->x.x = make_float4(rpp_hip_unpack0(src.x.x), rpp_hip_unpack3(src.x.x), rpp_hip_unpack2(src.x.y), rpp_hip_unpack1(src.y.x));    // write R00-R03
    src_f24->x.y = make_float4(rpp_hip_unpack0(src.y.y), rpp_hip_unpack3(src.y.y), rpp_hip_unpack2(src.z.x), rpp_hip_unpack1(src.z.y));    // write R04-R07
    src_f24->y.x = make_float4(rpp_hip_unpack1(src.x.x), rpp_hip_unpack0(src.x.y), rpp_hip_unpack3(src.x.y), rpp_hip_unpack2(src.y.x));    // write G00-G03
    src_f24->y.y = make_float4(rpp_hip_unpack1(src.y.y), rpp_hip_unpack0(src.z.x), rpp_hip_unpack3(src.z.x), rpp_hip_unpack2(src.z.y));    // write G04-G07
    src_f24->z.x = make_float4(rpp_hip_unpack2(src.x.x), rpp_hip_unpack1(src.x.y), rpp_hip_unpack0(src.y.x), rpp_hip_unpack3(src.y.x));    // write B00-B03
    src_f24->z.y = make_float4(rpp_hip_unpack2(src.y.y), rpp_hip_unpack1(src.z.x), rpp_hip_unpack0(src.z.y), rpp_hip_unpack3(src.z.y));    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(uchar *srcPtr, d_float24 *src_f24)
{
    d_uint6 src = *((d_uint6 *)(srcPtr));

    src_f24->x.x = make_float4(rpp_hip_unpack1(src.z.y), rpp_hip_unpack2(src.z.x), rpp_hip_unpack3(src.y.y), rpp_hip_unpack0(src.y.y));    // write R07-R04 (mirrored load)
    src_f24->x.y = make_float4(rpp_hip_unpack1(src.y.x), rpp_hip_unpack2(src.x.y), rpp_hip_unpack3(src.x.x), rpp_hip_unpack0(src.x.x));    // write R03-R00 (mirrored load)
    src_f24->y.x = make_float4(rpp_hip_unpack2(src.z.y), rpp_hip_unpack3(src.z.x), rpp_hip_unpack0(src.z.x), rpp_hip_unpack1(src.y.y));    // write G07-G04 (mirrored load)
    src_f24->y.y = make_float4(rpp_hip_unpack2(src.y.x), rpp_hip_unpack3(src.x.y), rpp_hip_unpack0(src.x.y), rpp_hip_unpack1(src.x.x));    // write G03-G00 (mirrored load)
    src_f24->z.x = make_float4(rpp_hip_unpack3(src.z.y), rpp_hip_unpack0(src.z.y), rpp_hip_unpack1(src.z.x), rpp_hip_unpack2(src.y.y));    // write B07-B04 (mirrored load)
    src_f24->z.y = make_float4(rpp_hip_unpack3(src.y.x), rpp_hip_unpack0(src.y.x), rpp_hip_unpack1(src.x.y), rpp_hip_unpack2(src.x.x));    // write B03-B00 (mirrored load)
}

// F32 loads with layout toggle PKD3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(float *srcPtr, d_float24 *src_f24)
{
    d_float24 *srcPtr_f24;
    srcPtr_f24 = (d_float24 *)srcPtr;

    src_f24->x.x = make_float4(srcPtr_f24->x.x.x, srcPtr_f24->x.x.w, srcPtr_f24->x.y.z, srcPtr_f24->y.x.y);    // write R00-R03
    src_f24->x.y = make_float4(srcPtr_f24->y.y.x, srcPtr_f24->y.y.w, srcPtr_f24->z.x.z, srcPtr_f24->z.y.y);    // write R04-R07
    src_f24->y.x = make_float4(srcPtr_f24->x.x.y, srcPtr_f24->x.y.x, srcPtr_f24->x.y.w, srcPtr_f24->y.x.z);    // write G00-G03
    src_f24->y.y = make_float4(srcPtr_f24->y.y.y, srcPtr_f24->z.x.x, srcPtr_f24->z.x.w, srcPtr_f24->z.y.z);    // write G04-G07
    src_f24->z.x = make_float4(srcPtr_f24->x.x.z, srcPtr_f24->x.y.y, srcPtr_f24->y.x.x, srcPtr_f24->y.x.w);    // write B00-B03
    src_f24->z.y = make_float4(srcPtr_f24->y.y.z, srcPtr_f24->z.x.y, srcPtr_f24->z.y.x, srcPtr_f24->z.y.w);    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(float *srcPtr, d_float24 *src_f24)
{
    d_float24 *srcPtr_f24;
    srcPtr_f24 = (d_float24 *)srcPtr;

    src_f24->x.x = make_float4(srcPtr_f24->z.y.y, srcPtr_f24->z.x.z, srcPtr_f24->y.y.w, srcPtr_f24->y.y.x);    // write R07-R04 (mirrored load)
    src_f24->x.y = make_float4(srcPtr_f24->y.x.y, srcPtr_f24->x.y.z, srcPtr_f24->x.x.w, srcPtr_f24->x.x.x);    // write R03-R00 (mirrored load)
    src_f24->y.x = make_float4(srcPtr_f24->z.y.z, srcPtr_f24->z.x.w, srcPtr_f24->z.x.x, srcPtr_f24->y.y.y);    // write G07-G04 (mirrored load)
    src_f24->y.y = make_float4(srcPtr_f24->y.x.z, srcPtr_f24->x.y.w, srcPtr_f24->x.y.x, srcPtr_f24->x.x.y);    // write G03-G00 (mirrored load)
    src_f24->z.x = make_float4(srcPtr_f24->z.y.w, srcPtr_f24->z.y.x, srcPtr_f24->z.x.y, srcPtr_f24->y.y.z);    // write B07-B04 (mirrored load)
    src_f24->z.y = make_float4(srcPtr_f24->y.x.w, srcPtr_f24->y.x.x, srcPtr_f24->x.y.y, srcPtr_f24->x.x.z);    // write B03-B00 (mirrored load)
}

// I8 loads with layout toggle PKD3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(schar *srcPtr, d_float24 *src_f24)
{
    d_int6 src = *((d_int6 *)(srcPtr));

    src_f24->x.x = make_float4(rpp_hip_unpack0(src.x.x), rpp_hip_unpack3(src.x.x), rpp_hip_unpack2(src.x.y), rpp_hip_unpack1(src.y.x));    // write R00-R03
    src_f24->x.y = make_float4(rpp_hip_unpack0(src.y.y), rpp_hip_unpack3(src.y.y), rpp_hip_unpack2(src.z.x), rpp_hip_unpack1(src.z.y));    // write R04-R07
    src_f24->y.x = make_float4(rpp_hip_unpack1(src.x.x), rpp_hip_unpack0(src.x.y), rpp_hip_unpack3(src.x.y), rpp_hip_unpack2(src.y.x));    // write G00-G03
    src_f24->y.y = make_float4(rpp_hip_unpack1(src.y.y), rpp_hip_unpack0(src.z.x), rpp_hip_unpack3(src.z.x), rpp_hip_unpack2(src.z.y));    // write G04-G07
    src_f24->z.x = make_float4(rpp_hip_unpack2(src.x.x), rpp_hip_unpack1(src.x.y), rpp_hip_unpack0(src.y.x), rpp_hip_unpack3(src.y.x));    // write B00-B03
    src_f24->z.y = make_float4(rpp_hip_unpack2(src.y.y), rpp_hip_unpack1(src.z.x), rpp_hip_unpack0(src.z.y), rpp_hip_unpack3(src.z.y));    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(schar *srcPtr, d_float24 *src_f24)
{
    d_int6 src = *((d_int6 *)(srcPtr));

    src_f24->x.x = make_float4(rpp_hip_unpack1(src.z.y), rpp_hip_unpack2(src.z.x), rpp_hip_unpack3(src.y.y), rpp_hip_unpack0(src.y.y));    // write R07-R04 (mirrored load)
    src_f24->x.y = make_float4(rpp_hip_unpack1(src.y.x), rpp_hip_unpack2(src.x.y), rpp_hip_unpack3(src.x.x), rpp_hip_unpack0(src.x.x));    // write R03-R00 (mirrored load)
    src_f24->y.x = make_float4(rpp_hip_unpack2(src.z.y), rpp_hip_unpack3(src.z.x), rpp_hip_unpack0(src.z.x), rpp_hip_unpack1(src.y.y));    // write G07-G04 (mirrored load)
    src_f24->y.y = make_float4(rpp_hip_unpack2(src.y.x), rpp_hip_unpack3(src.x.y), rpp_hip_unpack0(src.x.y), rpp_hip_unpack1(src.x.x));    // write G03-G00 (mirrored load)
    src_f24->z.x = make_float4(rpp_hip_unpack3(src.z.y), rpp_hip_unpack0(src.z.y), rpp_hip_unpack1(src.z.x), rpp_hip_unpack2(src.y.y));    // write B07-B04 (mirrored load)
    src_f24->z.y = make_float4(rpp_hip_unpack3(src.y.x), rpp_hip_unpack0(src.y.x), rpp_hip_unpack1(src.x.y), rpp_hip_unpack2(src.x.x));    // write B03-B00 (mirrored load)
}

// F16 loads with layout toggle PKD3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(half *srcPtr, d_float24 *src_f24)
{
    d_half24 *src_h24;
    src_h24 = (d_half24 *)srcPtr;

    src_f24->x.x = make_float4(__half2float(__low2half(src_h24->x.x.x)), __half2float(__high2half(src_h24->x.x.y)), __half2float(__low2half(src_h24->x.y.y)), __half2float(__high2half(src_h24->y.x.x)));    // write R00-R03
    src_f24->x.y = make_float4(__half2float(__low2half(src_h24->y.y.x)), __half2float(__high2half(src_h24->y.y.y)), __half2float(__low2half(src_h24->z.x.y)), __half2float(__high2half(src_h24->z.y.x)));    // write R04-R07
    src_f24->y.x = make_float4(__half2float(__high2half(src_h24->x.x.x)), __half2float(__low2half(src_h24->x.y.x)), __half2float(__high2half(src_h24->x.y.y)), __half2float(__low2half(src_h24->y.x.y)));    // write G00-G03
    src_f24->y.y = make_float4(__half2float(__high2half(src_h24->y.y.x)), __half2float(__low2half(src_h24->z.x.x)), __half2float(__high2half(src_h24->z.x.y)), __half2float(__low2half(src_h24->z.y.y)));    // write G04-G07
    src_f24->z.x = make_float4(__half2float(__low2half(src_h24->x.x.y)), __half2float(__high2half(src_h24->x.y.x)), __half2float(__low2half(src_h24->y.x.x)), __half2float(__high2half(src_h24->y.x.y)));    // write B00-B03
    src_f24->z.y = make_float4(__half2float(__low2half(src_h24->y.y.y)), __half2float(__high2half(src_h24->z.x.x)), __half2float(__low2half(src_h24->z.y.x)), __half2float(__high2half(src_h24->z.y.y)));    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(half *srcPtr, d_float24 *src_f24)
{
    d_half24 *src_h24;
    src_h24 = (d_half24 *)srcPtr;

    src_f24->x.x = make_float4(__half2float(__high2half(src_h24->z.y.x)), __half2float(__low2half(src_h24->z.x.y)), __half2float(__high2half(src_h24->y.y.y)), __half2float(__low2half(src_h24->y.y.x)));    // write R07-R04 (mirrored load)
    src_f24->x.y = make_float4(__half2float(__high2half(src_h24->y.x.x)), __half2float(__low2half(src_h24->x.y.y)), __half2float(__high2half(src_h24->x.x.y)), __half2float(__low2half(src_h24->x.x.x)));    // write R03-R00 (mirrored load)
    src_f24->y.x = make_float4(__half2float(__low2half(src_h24->z.y.y)), __half2float(__high2half(src_h24->z.x.y)), __half2float(__low2half(src_h24->z.x.x)), __half2float(__high2half(src_h24->y.y.x)));    // write G07-G04 (mirrored load)
    src_f24->y.y = make_float4(__half2float(__low2half(src_h24->y.x.y)), __half2float(__high2half(src_h24->x.y.y)), __half2float(__low2half(src_h24->x.y.x)), __half2float(__high2half(src_h24->x.x.x)));    // write G03-G00 (mirrored load)
    src_f24->z.x = make_float4(__half2float(__high2half(src_h24->z.y.y)), __half2float(__low2half(src_h24->z.y.x)), __half2float(__high2half(src_h24->z.x.x)), __half2float(__low2half(src_h24->y.y.y)));    // write B07-B04 (mirrored load)
    src_f24->z.y = make_float4(__half2float(__high2half(src_h24->y.x.y)), __half2float(__low2half(src_h24->y.x.x)), __half2float(__high2half(src_h24->x.y.x)), __half2float(__low2half(src_h24->x.x.y)));    // write B03-B00 (mirrored load)
}

// U8 loads with layout toggle PLN3 to PKD3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(uchar *srcPtr, uint increment, d_float24 *src_f24)
{
    d_uint6 src;
    uchar *srcTempPtr = srcPtr;

    src.x = *((uint2 *)(srcTempPtr));
    srcTempPtr += increment;
    src.y = *((uint2 *)(srcTempPtr));
    srcTempPtr += increment;
    src.z = *((uint2 *)(srcTempPtr));

    src_f24->x.x = make_float4(rpp_hip_unpack0(src.x.x), rpp_hip_unpack0(src.y.x), rpp_hip_unpack0(src.z.x), rpp_hip_unpack1(src.x.x));    // write R00G00B00R01
    src_f24->x.y = make_float4(rpp_hip_unpack1(src.y.x), rpp_hip_unpack1(src.z.x), rpp_hip_unpack2(src.x.x), rpp_hip_unpack2(src.y.x));    // write G01B01R02G02
    src_f24->y.x = make_float4(rpp_hip_unpack2(src.z.x), rpp_hip_unpack3(src.x.x), rpp_hip_unpack3(src.y.x), rpp_hip_unpack3(src.z.x));    // write B02R03G03B03
    src_f24->y.y = make_float4(rpp_hip_unpack0(src.x.y), rpp_hip_unpack0(src.y.y), rpp_hip_unpack0(src.z.y), rpp_hip_unpack1(src.x.y));    // write R04G04B04R05
    src_f24->z.x = make_float4(rpp_hip_unpack1(src.y.y), rpp_hip_unpack1(src.z.y), rpp_hip_unpack2(src.x.y), rpp_hip_unpack2(src.y.y));    // write G05B05R06G06
    src_f24->z.y = make_float4(rpp_hip_unpack2(src.z.y), rpp_hip_unpack3(src.x.y), rpp_hip_unpack3(src.y.y), rpp_hip_unpack3(src.z.y));    // write B06R07G07B07
}

// F32 loads with layout toggle PLN3 to PKD3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(float *srcPtr, uint increment, d_float24 *src_f24)
{
    float *srcPtrR, *srcPtrG, *srcPtrB;
    srcPtrR = srcPtr;
    srcPtrG = srcPtrR + increment;
    srcPtrB = srcPtrG + increment;

    d_float8 *srcPtrR_f8, *srcPtrG_f8, *srcPtrB_f8;

    srcPtrR_f8 = (d_float8 *)srcPtrR;
    srcPtrG_f8 = (d_float8 *)srcPtrG;
    srcPtrB_f8 = (d_float8 *)srcPtrB;

    src_f24->x.x = make_float4(srcPtrR_f8->x.x, srcPtrG_f8->x.x, srcPtrB_f8->x.x, srcPtrR_f8->x.y);    // write R00G00B00R01
    src_f24->x.y = make_float4(srcPtrG_f8->x.y, srcPtrB_f8->x.y, srcPtrR_f8->x.z, srcPtrG_f8->x.z);    // write G01B01R02G02
    src_f24->y.x = make_float4(srcPtrB_f8->x.z, srcPtrR_f8->x.w, srcPtrG_f8->x.w, srcPtrB_f8->x.w);    // write B02R03G03B03
    src_f24->y.y = make_float4(srcPtrR_f8->y.x, srcPtrG_f8->y.x, srcPtrB_f8->y.x, srcPtrR_f8->y.y);    // write R04G04B04R05
    src_f24->z.x = make_float4(srcPtrG_f8->y.y, srcPtrB_f8->y.y, srcPtrR_f8->y.z, srcPtrG_f8->y.z);    // write G05B05R06G06
    src_f24->z.y = make_float4(srcPtrB_f8->y.z, srcPtrR_f8->y.w, srcPtrG_f8->y.w, srcPtrB_f8->y.w);    // write B06R07G07B07
}

// I8 loads with layout toggle PLN3 to PKD3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(schar *srcPtr, uint increment, d_float24 *src_f24)
{
    d_int6 src;
    schar *srcTempPtr = srcPtr;

    src.x = *((int2 *)(srcTempPtr));
    srcTempPtr += increment;
    src.y = *((int2 *)(srcTempPtr));
    srcTempPtr += increment;
    src.z = *((int2 *)(srcTempPtr));

    src_f24->x.x = make_float4(rpp_hip_unpack0(src.x.x), rpp_hip_unpack0(src.y.x), rpp_hip_unpack0(src.z.x), rpp_hip_unpack1(src.x.x));    // write R00G00B00R01
    src_f24->x.y = make_float4(rpp_hip_unpack1(src.y.x), rpp_hip_unpack1(src.z.x), rpp_hip_unpack2(src.x.x), rpp_hip_unpack2(src.y.x));    // write G01B01R02G02
    src_f24->y.x = make_float4(rpp_hip_unpack2(src.z.x), rpp_hip_unpack3(src.x.x), rpp_hip_unpack3(src.y.x), rpp_hip_unpack3(src.z.x));    // write B02R03G03B03
    src_f24->y.y = make_float4(rpp_hip_unpack0(src.x.y), rpp_hip_unpack0(src.y.y), rpp_hip_unpack0(src.z.y), rpp_hip_unpack1(src.x.y));    // write R04G04B04R05
    src_f24->z.x = make_float4(rpp_hip_unpack1(src.y.y), rpp_hip_unpack1(src.z.y), rpp_hip_unpack2(src.x.y), rpp_hip_unpack2(src.y.y));    // write G05B05R06G06
    src_f24->z.y = make_float4(rpp_hip_unpack2(src.z.y), rpp_hip_unpack3(src.x.y), rpp_hip_unpack3(src.y.y), rpp_hip_unpack3(src.z.y));    // write B06R07G07B07
}

// F16 loads with layout toggle PLN3 to PKD3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(half *srcPtr, uint increment, d_float24 *src_f24)
{
    half *srcPtrR, *srcPtrG, *srcPtrB;
    srcPtrR = srcPtr;
    srcPtrG = srcPtrR + increment;
    srcPtrB = srcPtrG + increment;

    d_half8 *srcR_h8, *srcG_h8, *srcB_h8;
    srcR_h8 = (d_half8 *)srcPtrR;
    srcG_h8 = (d_half8 *)srcPtrG;
    srcB_h8 = (d_half8 *)srcPtrB;

    src_f24->x.x = make_float4(__half2float(__low2half(srcR_h8->x.x)), __half2float(__low2half(srcG_h8->x.x)), __half2float(__low2half(srcB_h8->x.x)), __half2float(__high2half(srcR_h8->x.x)));      // write R00G00B00R01
    src_f24->x.y = make_float4(__half2float(__high2half(srcG_h8->x.x)), __half2float(__high2half(srcB_h8->x.x)), __half2float(__low2half(srcR_h8->x.y)), __half2float(__low2half(srcG_h8->x.y)));     // write G01B01R02G02
    src_f24->y.x = make_float4(__half2float(__low2half(srcB_h8->x.y)), __half2float(__high2half(srcR_h8->x.y)), __half2float(__high2half(srcG_h8->x.y)), __half2float(__high2half(srcB_h8->x.y)));    // write B02R03G03B03
    src_f24->y.y = make_float4(__half2float(__low2half(srcR_h8->y.x)), __half2float(__low2half(srcG_h8->y.x)), __half2float(__low2half(srcB_h8->y.x)), __half2float(__high2half(srcR_h8->y.x)));      // write R04G04B04R05
    src_f24->z.x = make_float4(__half2float(__high2half(srcG_h8->y.x)), __half2float(__high2half(srcB_h8->y.x)), __half2float(__low2half(srcR_h8->y.y)), __half2float(__low2half(srcG_h8->y.y)));     // write G05B05R06G06
    src_f24->z.y = make_float4(__half2float(__low2half(srcB_h8->y.y)), __half2float(__high2half(srcR_h8->y.y)), __half2float(__high2half(srcG_h8->y.y)), __half2float(__high2half(srcB_h8->y.y)));    // write B06R07G07B07
}

// -------------------- Set 5 - Stores from float --------------------

// WITHOUT LAYOUT TOGGLE

// U8 stores without layout toggle (8 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(uchar *dstPtr, d_float8 *dst_f8)
{
    uint2 dst;
    dst.x = rpp_hip_pack(dst_f8->x);
    dst.y = rpp_hip_pack(dst_f8->y);
    *((uint2 *)(dstPtr)) = dst;
}

// F32 stores without layout toggle (8 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(float *dstPtr, d_float8 *dst_f8)
{
    *((d_float8 *)(dstPtr)) = *dst_f8;
}

// I8 stores without layout toggle (8 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(schar *dstPtr, d_float8 *dst_f8)
{
    uint2 dst;
    dst.x = rpp_hip_pack_i8(dst_f8->x);
    dst.y = rpp_hip_pack_i8(dst_f8->y);
    *((uint2 *)(dstPtr)) = dst;
}

// F16 stores without layout toggle (8 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(half *dstPtr, d_float8 *dst_f8)
{
    d_half8 dst_h8;

    dst_h8.x.x = __float22half2_rn(make_float2(dst_f8->x.x, dst_f8->x.y));
    dst_h8.x.y = __float22half2_rn(make_float2(dst_f8->x.z, dst_f8->x.w));
    dst_h8.y.x = __float22half2_rn(make_float2(dst_f8->y.x, dst_f8->y.y));
    dst_h8.y.y = __float22half2_rn(make_float2(dst_f8->y.z, dst_f8->y.w));

    *((d_half8 *)(dstPtr)) = dst_h8;
}

// U8 stores without layout toggle PKD3 to PKD3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(uchar *dstPtr, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack(dst_f24->x.x);    // write R00G00B00R01
    dst.x.y = rpp_hip_pack(dst_f24->x.y);    // write G01B01R02G02
    dst.y.x = rpp_hip_pack(dst_f24->y.x);    // write B02R03G03B03
    dst.y.y = rpp_hip_pack(dst_f24->y.y);    // write R04G04B04R05
    dst.z.x = rpp_hip_pack(dst_f24->z.x);    // write G05B05R06G06
    dst.z.y = rpp_hip_pack(dst_f24->z.y);    // write B06R07G07B07

    *((d_uint6 *)(dstPtr)) = dst;
}

// F32 stores without layout toggle PKD3 to PKD3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(float *dstPtr, d_float24 *dst_f24)
{
    *((d_float24 *)(dstPtr)) = *dst_f24;
}

// I8 stores without layout toggle PKD3 to PKD3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(schar *dstPtr, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack_i8(dst_f24->x.x);    // write R00G00B00R01
    dst.x.y = rpp_hip_pack_i8(dst_f24->x.y);    // write G01B01R02G02
    dst.y.x = rpp_hip_pack_i8(dst_f24->y.x);    // write B02R03G03B03
    dst.y.y = rpp_hip_pack_i8(dst_f24->y.y);    // write R04G04B04R05
    dst.z.x = rpp_hip_pack_i8(dst_f24->z.x);    // write G05B05R06G06
    dst.z.y = rpp_hip_pack_i8(dst_f24->z.y);    // write B06R07G07B07

    *((d_uint6 *)(dstPtr)) = dst;
}

// F16 stores without layout toggle PKD3 to PKD3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(half *dstPtr, d_float24 *dst_f24)
{
    d_half24 dst_h24;

    dst_h24.x.x.x = __float22half2_rn(make_float2(dst_f24->x.x.x, dst_f24->x.x.y));    // write R00G00
    dst_h24.x.x.y = __float22half2_rn(make_float2(dst_f24->x.x.z, dst_f24->x.x.w));    // write B00R01
    dst_h24.x.y.x = __float22half2_rn(make_float2(dst_f24->x.y.x, dst_f24->x.y.y));    // write G01B01
    dst_h24.x.y.y = __float22half2_rn(make_float2(dst_f24->x.y.z, dst_f24->x.y.w));    // write R02G02
    dst_h24.y.x.x = __float22half2_rn(make_float2(dst_f24->y.x.x, dst_f24->y.x.y));    // write B02R03
    dst_h24.y.x.y = __float22half2_rn(make_float2(dst_f24->y.x.z, dst_f24->y.x.w));    // write G03B03
    dst_h24.y.y.x = __float22half2_rn(make_float2(dst_f24->y.y.x, dst_f24->y.y.y));    // write R04G04
    dst_h24.y.y.y = __float22half2_rn(make_float2(dst_f24->y.y.z, dst_f24->y.y.w));    // write B04R05
    dst_h24.z.x.x = __float22half2_rn(make_float2(dst_f24->z.x.x, dst_f24->z.x.y));    // write G05B05
    dst_h24.z.x.y = __float22half2_rn(make_float2(dst_f24->z.x.z, dst_f24->z.x.w));    // write R06G06
    dst_h24.z.y.x = __float22half2_rn(make_float2(dst_f24->z.y.x, dst_f24->z.y.y));    // write B06R07
    dst_h24.z.y.y = __float22half2_rn(make_float2(dst_f24->z.y.z, dst_f24->z.y.w));    // write G07B07

    *((d_half24 *)(dstPtr)) = dst_h24;
}

// U8 stores without layout toggle PLN3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pln3(uchar *dstPtr, uint increment, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack(dst_f24->x.x);    // write R00-R03
    dst.x.y = rpp_hip_pack(dst_f24->x.y);    // write R04-R07
    dst.y.x = rpp_hip_pack(dst_f24->y.x);    // write G00-G03
    dst.y.y = rpp_hip_pack(dst_f24->y.y);    // write G04-G07
    dst.z.x = rpp_hip_pack(dst_f24->z.x);    // write B00-B03
    dst.z.y = rpp_hip_pack(dst_f24->z.y);    // write B04-B07

    *((uint2 *)(dstPtr)) = dst.x;
    dstPtr += increment;
    *((uint2 *)(dstPtr)) = dst.y;
    dstPtr += increment;
    *((uint2 *)(dstPtr)) = dst.z;
}

// F32 stores without layout toggle PLN3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pln3(float *dstPtr, uint increment, d_float24 *dst_f24)
{
    *((d_float8 *)(dstPtr)) = dst_f24->x;
    dstPtr += increment;
    *((d_float8 *)(dstPtr)) = dst_f24->y;
    dstPtr += increment;
    *((d_float8 *)(dstPtr)) = dst_f24->z;
}

// I8 stores without layout toggle PLN3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pln3(schar *dstPtr, uint increment, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack_i8(dst_f24->x.x);    // write R00-R03
    dst.x.y = rpp_hip_pack_i8(dst_f24->x.y);    // write R04-R07
    dst.y.x = rpp_hip_pack_i8(dst_f24->y.x);    // write G00-G03
    dst.y.y = rpp_hip_pack_i8(dst_f24->y.y);    // write G04-G07
    dst.z.x = rpp_hip_pack_i8(dst_f24->z.x);    // write B00-B03
    dst.z.y = rpp_hip_pack_i8(dst_f24->z.y);    // write B04-B07

    *((uint2 *)(dstPtr)) = dst.x;
    dstPtr += increment;
    *((uint2 *)(dstPtr)) = dst.y;
    dstPtr += increment;
    *((uint2 *)(dstPtr)) = dst.z;
}

// F16 stores without layout toggle PLN3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pln3(half *dstPtr, uint increment, d_float24 *dst_f24)
{
    d_half24 dst_h24;

    dst_h24.x.x.x = __float22half2_rn(make_float2(dst_f24->x.x.x, dst_f24->x.x.y));    // write R00R01
    dst_h24.x.x.y = __float22half2_rn(make_float2(dst_f24->x.x.z, dst_f24->x.x.w));    // write R02R03
    dst_h24.x.y.x = __float22half2_rn(make_float2(dst_f24->x.y.x, dst_f24->x.y.y));    // write R04R05
    dst_h24.x.y.y = __float22half2_rn(make_float2(dst_f24->x.y.z, dst_f24->x.y.w));    // write R06R07
    dst_h24.y.x.x = __float22half2_rn(make_float2(dst_f24->y.x.x, dst_f24->y.x.y));    // write G00G01
    dst_h24.y.x.y = __float22half2_rn(make_float2(dst_f24->y.x.z, dst_f24->y.x.w));    // write G02G03
    dst_h24.y.y.x = __float22half2_rn(make_float2(dst_f24->y.y.x, dst_f24->y.y.y));    // write G04G05
    dst_h24.y.y.y = __float22half2_rn(make_float2(dst_f24->y.y.z, dst_f24->y.y.w));    // write G06G07
    dst_h24.z.x.x = __float22half2_rn(make_float2(dst_f24->z.x.x, dst_f24->z.x.y));    // write B00B01
    dst_h24.z.x.y = __float22half2_rn(make_float2(dst_f24->z.x.z, dst_f24->z.x.w));    // write B02B03
    dst_h24.z.y.x = __float22half2_rn(make_float2(dst_f24->z.y.x, dst_f24->z.y.y));    // write B04B05
    dst_h24.z.y.y = __float22half2_rn(make_float2(dst_f24->z.y.z, dst_f24->z.y.w));    // write B06B07

    *((d_half8 *)(dstPtr)) = dst_h24.x;
    dstPtr += increment;
    *((d_half8 *)(dstPtr)) = dst_h24.y;
    dstPtr += increment;
    *((d_half8 *)(dstPtr)) = dst_h24.z;
}

// WITH LAYOUT TOGGLE

// U8 stores with layout toggle PLN3 to PKD3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pkd3(uchar *dstPtr, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack(make_float4(dst_f24->x.x.x, dst_f24->y.x.x, dst_f24->z.x.x, dst_f24->x.x.y));    // write R00G00B00R01
    dst.x.y = rpp_hip_pack(make_float4(dst_f24->y.x.y, dst_f24->z.x.y, dst_f24->x.x.z, dst_f24->y.x.z));    // write G01B01R02G02
    dst.y.x = rpp_hip_pack(make_float4(dst_f24->z.x.z, dst_f24->x.x.w, dst_f24->y.x.w, dst_f24->z.x.w));    // write B02R03G03B03
    dst.y.y = rpp_hip_pack(make_float4(dst_f24->x.y.x, dst_f24->y.y.x, dst_f24->z.y.x, dst_f24->x.y.y));    // write R04G04B04R05
    dst.z.x = rpp_hip_pack(make_float4(dst_f24->y.y.y, dst_f24->z.y.y, dst_f24->x.y.z, dst_f24->y.y.z));    // write G05B05R06G06
    dst.z.y = rpp_hip_pack(make_float4(dst_f24->z.y.z, dst_f24->x.y.w, dst_f24->y.y.w, dst_f24->z.y.w));    // write B06R07G07B07

    *((d_uint6 *)(dstPtr)) = dst;
}

// F32 stores with layout toggle PLN3 to PKD3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pkd3(float *dstPtr, d_float24 *dst_f24)
{
    d_float24 dstPtr_f24;

    dstPtr_f24.x.x = make_float4(dst_f24->x.x.x, dst_f24->y.x.x, dst_f24->z.x.x, dst_f24->x.x.y);    // write R00G00B00R01
    dstPtr_f24.x.y = make_float4(dst_f24->y.x.y, dst_f24->z.x.y, dst_f24->x.x.z, dst_f24->y.x.z);    // write G01B01R02G02
    dstPtr_f24.y.x = make_float4(dst_f24->z.x.z, dst_f24->x.x.w, dst_f24->y.x.w, dst_f24->z.x.w);    // write B02R03G03B03
    dstPtr_f24.y.y = make_float4(dst_f24->x.y.x, dst_f24->y.y.x, dst_f24->z.y.x, dst_f24->x.y.y);    // write R04G04B04R05
    dstPtr_f24.z.x = make_float4(dst_f24->y.y.y, dst_f24->z.y.y, dst_f24->x.y.z, dst_f24->y.y.z);    // write G05B05R06G06
    dstPtr_f24.z.y = make_float4(dst_f24->z.y.z, dst_f24->x.y.w, dst_f24->y.y.w, dst_f24->z.y.w);    // write B06R07G07B07

    *((d_float24 *)(dstPtr)) = dstPtr_f24;
}

// I8 stores with layout toggle PLN3 to PKD3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pkd3(schar *dstPtr, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack_i8(make_float4(dst_f24->x.x.x, dst_f24->y.x.x, dst_f24->z.x.x, dst_f24->x.x.y));    // write R00G00B00R01
    dst.x.y = rpp_hip_pack_i8(make_float4(dst_f24->y.x.y, dst_f24->z.x.y, dst_f24->x.x.z, dst_f24->y.x.z));    // write G01B01R02G02
    dst.y.x = rpp_hip_pack_i8(make_float4(dst_f24->z.x.z, dst_f24->x.x.w, dst_f24->y.x.w, dst_f24->z.x.w));    // write B02R03G03B03
    dst.y.y = rpp_hip_pack_i8(make_float4(dst_f24->x.y.x, dst_f24->y.y.x, dst_f24->z.y.x, dst_f24->x.y.y));    // write R04G04B04R05
    dst.z.x = rpp_hip_pack_i8(make_float4(dst_f24->y.y.y, dst_f24->z.y.y, dst_f24->x.y.z, dst_f24->y.y.z));    // write G05B05R06G06
    dst.z.y = rpp_hip_pack_i8(make_float4(dst_f24->z.y.z, dst_f24->x.y.w, dst_f24->y.y.w, dst_f24->z.y.w));    // write B06R07G07B07

    *((d_uint6 *)(dstPtr)) = dst;
}

// F16 stores with layout toggle PLN3 to PKD3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pkd3(half *dstPtr, d_float24 *dst_f24)
{
    d_half24 dst_h24;

    dst_h24.x.x.x = __float22half2_rn(make_float2(dst_f24->x.x.x, dst_f24->y.x.x));    // write R00G00
    dst_h24.x.x.y = __float22half2_rn(make_float2(dst_f24->z.x.x, dst_f24->x.x.y));    // write B00R01
    dst_h24.x.y.x = __float22half2_rn(make_float2(dst_f24->y.x.y, dst_f24->z.x.y));    // write G01B01
    dst_h24.x.y.y = __float22half2_rn(make_float2(dst_f24->x.x.z, dst_f24->y.x.z));    // write R02G02
    dst_h24.y.x.x = __float22half2_rn(make_float2(dst_f24->z.x.z, dst_f24->x.x.w));    // write B02R03
    dst_h24.y.x.y = __float22half2_rn(make_float2(dst_f24->y.x.w, dst_f24->z.x.w));    // write G03B03
    dst_h24.y.y.x = __float22half2_rn(make_float2(dst_f24->x.y.x, dst_f24->y.y.x));    // write R04G04
    dst_h24.y.y.y = __float22half2_rn(make_float2(dst_f24->z.y.x, dst_f24->x.y.y));    // write B04R05
    dst_h24.z.x.x = __float22half2_rn(make_float2(dst_f24->y.y.y, dst_f24->z.y.y));    // write G05B05
    dst_h24.z.x.y = __float22half2_rn(make_float2(dst_f24->x.y.z, dst_f24->y.y.z));    // write R06G06
    dst_h24.z.y.x = __float22half2_rn(make_float2(dst_f24->z.y.z, dst_f24->x.y.w));    // write B06R07
    dst_h24.z.y.y = __float22half2_rn(make_float2(dst_f24->y.y.w, dst_f24->z.y.w));    // write G07B07

    *((d_half24 *)(dstPtr)) = dst_h24;
}

// U8 stores with layout toggle PKD3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pln3(uchar *dstPtr, uint increment, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack(make_float4(dst_f24->x.x.x, dst_f24->x.x.w, dst_f24->x.y.z, dst_f24->y.x.y));    // write R00-R03
    dst.x.y = rpp_hip_pack(make_float4(dst_f24->y.y.x, dst_f24->y.y.w, dst_f24->z.x.z, dst_f24->z.y.y));    // write R04-R07
    dst.y.x = rpp_hip_pack(make_float4(dst_f24->x.x.y, dst_f24->x.y.x, dst_f24->x.y.w, dst_f24->y.x.z));    // write G00-G03
    dst.y.y = rpp_hip_pack(make_float4(dst_f24->y.y.y, dst_f24->z.x.x, dst_f24->z.x.w, dst_f24->z.y.z));    // write G04-G07
    dst.z.x = rpp_hip_pack(make_float4(dst_f24->x.x.z, dst_f24->x.y.y, dst_f24->y.x.x, dst_f24->y.x.w));    // write B00-B03
    dst.z.y = rpp_hip_pack(make_float4(dst_f24->y.y.z, dst_f24->z.x.y, dst_f24->z.y.x, dst_f24->z.y.w));    // write B04-B07

    *((uint2 *)(dstPtr)) = dst.x;
    dstPtr += increment;
    *((uint2 *)(dstPtr)) = dst.y;
    dstPtr += increment;
    *((uint2 *)(dstPtr)) = dst.z;
}

// F32 stores with layout toggle PKD3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pln3(float *dstPtr, uint increment, d_float24 *dst_f24)
{
    d_float24 dstPtr_f24;

    dstPtr_f24.x.x = make_float4(dst_f24->x.x.x, dst_f24->x.x.w, dst_f24->x.y.z, dst_f24->y.x.y);    // write R00-R03
    dstPtr_f24.x.y = make_float4(dst_f24->y.y.x, dst_f24->y.y.w, dst_f24->z.x.z, dst_f24->z.y.y);    // write R04-R07
    dstPtr_f24.y.x = make_float4(dst_f24->x.x.y, dst_f24->x.y.x, dst_f24->x.y.w, dst_f24->y.x.z);    // write G00-G03
    dstPtr_f24.y.y = make_float4(dst_f24->y.y.y, dst_f24->z.x.x, dst_f24->z.x.w, dst_f24->z.y.z);    // write G04-G07
    dstPtr_f24.z.x = make_float4(dst_f24->x.x.z, dst_f24->x.y.y, dst_f24->y.x.x, dst_f24->y.x.w);    // write B00-B03
    dstPtr_f24.z.y = make_float4(dst_f24->y.y.z, dst_f24->z.x.y, dst_f24->z.y.x, dst_f24->z.y.w);    // write B04-B07

    *(d_float8 *)dstPtr = dstPtr_f24.x;
    dstPtr += increment;
    *(d_float8 *)dstPtr = dstPtr_f24.y;
    dstPtr += increment;
    *(d_float8 *)dstPtr = dstPtr_f24.z;
}

// I8 stores with layout toggle PKD3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pln3(schar *dstPtr, uint increment, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack_i8(make_float4(dst_f24->x.x.x, dst_f24->x.x.w, dst_f24->x.y.z, dst_f24->y.x.y));    // write R00-R03
    dst.x.y = rpp_hip_pack_i8(make_float4(dst_f24->y.y.x, dst_f24->y.y.w, dst_f24->z.x.z, dst_f24->z.y.y));    // write R04-R07
    dst.y.x = rpp_hip_pack_i8(make_float4(dst_f24->x.x.y, dst_f24->x.y.x, dst_f24->x.y.w, dst_f24->y.x.z));    // write G00-G03
    dst.y.y = rpp_hip_pack_i8(make_float4(dst_f24->y.y.y, dst_f24->z.x.x, dst_f24->z.x.w, dst_f24->z.y.z));    // write G04-G07
    dst.z.x = rpp_hip_pack_i8(make_float4(dst_f24->x.x.z, dst_f24->x.y.y, dst_f24->y.x.x, dst_f24->y.x.w));    // write B00-B03
    dst.z.y = rpp_hip_pack_i8(make_float4(dst_f24->y.y.z, dst_f24->z.x.y, dst_f24->z.y.x, dst_f24->z.y.w));    // write B04-B07

    *((uint2 *)(dstPtr)) = dst.x;
    dstPtr += increment;
    *((uint2 *)(dstPtr)) = dst.y;
    dstPtr += increment;
    *((uint2 *)(dstPtr)) = dst.z;
}

// F16 stores with layout toggle PKD3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pln3(half *dstPtr, uint increment, d_float24 *dst_f24)
{
    d_half24 dst_h24;

    dst_h24.x.x.x = __float22half2_rn(make_float2(dst_f24->x.x.x, dst_f24->x.x.w));    // write R00R01
    dst_h24.x.x.y = __float22half2_rn(make_float2(dst_f24->x.y.z, dst_f24->y.x.y));    // write R02R03
    dst_h24.x.y.x = __float22half2_rn(make_float2(dst_f24->y.y.x, dst_f24->y.y.w));    // write R04R05
    dst_h24.x.y.y = __float22half2_rn(make_float2(dst_f24->z.x.z, dst_f24->z.y.y));    // write R06R07
    dst_h24.y.x.x = __float22half2_rn(make_float2(dst_f24->x.x.y, dst_f24->x.y.x));    // write G00G01
    dst_h24.y.x.y = __float22half2_rn(make_float2(dst_f24->x.y.w, dst_f24->y.x.z));    // write G02G03
    dst_h24.y.y.x = __float22half2_rn(make_float2(dst_f24->y.y.y, dst_f24->z.x.x));    // write G04G05
    dst_h24.y.y.y = __float22half2_rn(make_float2(dst_f24->z.x.w, dst_f24->z.y.z));    // write G06G07
    dst_h24.z.x.x = __float22half2_rn(make_float2(dst_f24->x.x.z, dst_f24->x.y.y));    // write B00B01
    dst_h24.z.x.y = __float22half2_rn(make_float2(dst_f24->y.x.x, dst_f24->y.x.w));    // write B02B03
    dst_h24.z.y.x = __float22half2_rn(make_float2(dst_f24->y.y.z, dst_f24->z.x.y));    // write B04B05
    dst_h24.z.y.y = __float22half2_rn(make_float2(dst_f24->z.y.x, dst_f24->z.y.w));    // write B06B07

    *((d_half8 *)(dstPtr)) = dst_h24.x;
    dstPtr += increment;
    *((d_half8 *)(dstPtr)) = dst_h24.y;
    dstPtr += increment;
    *((d_half8 *)(dstPtr)) = dst_h24.z;
}

// -------------------- Set 6 - Loads to uchar --------------------

// WITHOUT LAYOUT TOGGLE

// U8 loads without layout toggle (8 U8 pixels)

__device__ __forceinline__ void rpp_hip_load8_to_uchar8(uchar *srcPtr, uchar *src_uchar8)
{
    *(uint2 *)src_uchar8 = *(uint2 *)srcPtr;
}

// F32 loads without layout toggle (8 F32 pixels)

__device__ __forceinline__ void rpp_hip_load8_to_uchar8(float *srcPtr, uchar *src_uchar8)
{
    d_float8 *srcPtr_f8;
    srcPtr_f8 = (d_float8 *)srcPtr;

    d_float8 src_f8;
    src_f8.x = rpp_hip_pixel_check_0to255(srcPtr_f8->x * (float4) 255.0);
    src_f8.y = rpp_hip_pixel_check_0to255(srcPtr_f8->y * (float4) 255.0);

    uint2 *srcPtr_lds;
    srcPtr_lds = (uint2 *)src_uchar8;
    srcPtr_lds->x = rpp_hip_pack(src_f8.x);
    srcPtr_lds->y = rpp_hip_pack(src_f8.y);
}

// I8 loads without layout toggle (8 I8 pixels)

__device__ __forceinline__ void rpp_hip_load8_to_uchar8(schar *srcPtr, uchar *src_uchar8)
{
    rpp_hip_convert8_i8_to_u8(srcPtr, src_uchar8);
}

// F16 loads without layout toggle (8 F16 pixels)

__device__ __forceinline__ void rpp_hip_load8_to_uchar8(half *srcPtr, uchar *src_uchar8)
{
    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr, &src_f8);
    rpp_hip_load8_to_uchar8((float *)&src_f8, src_uchar8);
}

// WITH LAYOUT TOGGLE

// U8 loads with layout toggle PKD3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_to_uchar8_pln3(uchar *srcPtr, uchar **srcPtrs_uchar8)
{
    d_uchar24 *srcPtr_uchar24;
    srcPtr_uchar24 = (d_uchar24 *)srcPtr;

    d_uchar8 *src_c1_uchar8, *src_c2_uchar8, *src_c3_uchar8;
    src_c1_uchar8 = (d_uchar8 *)srcPtrs_uchar8[0];
    src_c2_uchar8 = (d_uchar8 *)srcPtrs_uchar8[1];
    src_c3_uchar8 = (d_uchar8 *)srcPtrs_uchar8[2];

    src_c1_uchar8->x = make_uchar4(srcPtr_uchar24->x.x.x, srcPtr_uchar24->x.x.w, srcPtr_uchar24->x.y.z, srcPtr_uchar24->y.x.y);    // write R00-R03
    src_c1_uchar8->y = make_uchar4(srcPtr_uchar24->y.y.x, srcPtr_uchar24->y.y.w, srcPtr_uchar24->z.x.z, srcPtr_uchar24->z.y.y);    // write R04-R07
    src_c2_uchar8->x = make_uchar4(srcPtr_uchar24->x.x.y, srcPtr_uchar24->x.y.x, srcPtr_uchar24->x.y.w, srcPtr_uchar24->y.x.z);    // write G00-G03
    src_c2_uchar8->y = make_uchar4(srcPtr_uchar24->y.y.y, srcPtr_uchar24->z.x.x, srcPtr_uchar24->z.x.w, srcPtr_uchar24->z.y.z);    // write G04-G07
    src_c3_uchar8->x = make_uchar4(srcPtr_uchar24->x.x.z, srcPtr_uchar24->x.y.y, srcPtr_uchar24->y.x.x, srcPtr_uchar24->y.x.w);    // write B00-B03
    src_c3_uchar8->y = make_uchar4(srcPtr_uchar24->y.y.z, srcPtr_uchar24->z.x.y, srcPtr_uchar24->z.y.x, srcPtr_uchar24->z.y.w);    // write B04-B07
}

// F32 loads with layout toggle PKD3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_to_uchar8_pln3(float *srcPtr, uchar **srcPtrs_uchar8)
{
    d_float24 *srcPtr_f24;
    srcPtr_f24 = (d_float24 *)srcPtr;

    d_uint6 src_uchar24;
    src_uchar24.x.x = rpp_hip_pack(rpp_hip_pixel_check_0to255(srcPtr_f24->x.x * (float4) 255.0));    // write R00G00B00R01
    src_uchar24.x.y = rpp_hip_pack(rpp_hip_pixel_check_0to255(srcPtr_f24->x.y * (float4) 255.0));    // write G01B01R02G02
    src_uchar24.y.x = rpp_hip_pack(rpp_hip_pixel_check_0to255(srcPtr_f24->y.x * (float4) 255.0));    // write B02R03G03B03
    src_uchar24.y.y = rpp_hip_pack(rpp_hip_pixel_check_0to255(srcPtr_f24->y.y * (float4) 255.0));    // write R04G04B04R05
    src_uchar24.z.x = rpp_hip_pack(rpp_hip_pixel_check_0to255(srcPtr_f24->z.x * (float4) 255.0));    // write G05B05R06G06
    src_uchar24.z.y = rpp_hip_pack(rpp_hip_pixel_check_0to255(srcPtr_f24->z.y * (float4) 255.0));    // write B06R07G07B07

    d_uchar8 *src_c1_uchar8, *src_c2_uchar8, *src_c3_uchar8;
    src_c1_uchar8 = (d_uchar8 *)srcPtrs_uchar8[0];
    src_c2_uchar8 = (d_uchar8 *)srcPtrs_uchar8[1];
    src_c3_uchar8 = (d_uchar8 *)srcPtrs_uchar8[2];

    d_uchar24 *srcPtr_uchar24;
    srcPtr_uchar24 = (d_uchar24 *)&src_uchar24;

    src_c1_uchar8->x = make_uchar4(srcPtr_uchar24->x.x.x, srcPtr_uchar24->x.x.w, srcPtr_uchar24->x.y.z, srcPtr_uchar24->y.x.y);    // write R00-R03
    src_c1_uchar8->y = make_uchar4(srcPtr_uchar24->y.y.x, srcPtr_uchar24->y.y.w, srcPtr_uchar24->z.x.z, srcPtr_uchar24->z.y.y);    // write R04-R07
    src_c2_uchar8->x = make_uchar4(srcPtr_uchar24->x.x.y, srcPtr_uchar24->x.y.x, srcPtr_uchar24->x.y.w, srcPtr_uchar24->y.x.z);    // write G00-G03
    src_c2_uchar8->y = make_uchar4(srcPtr_uchar24->y.y.y, srcPtr_uchar24->z.x.x, srcPtr_uchar24->z.x.w, srcPtr_uchar24->z.y.z);    // write G04-G07
    src_c3_uchar8->x = make_uchar4(srcPtr_uchar24->x.x.z, srcPtr_uchar24->x.y.y, srcPtr_uchar24->y.x.x, srcPtr_uchar24->y.x.w);    // write B00-B03
    src_c3_uchar8->y = make_uchar4(srcPtr_uchar24->y.y.z, srcPtr_uchar24->z.x.y, srcPtr_uchar24->z.y.x, srcPtr_uchar24->z.y.w);    // write B04-B07
}

// F16 loads with layout toggle PKD3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_to_uchar8_pln3(half *srcPtr, uchar **srcPtrs_uchar8)
{
    d_half24 *srcPtr_h24;
    srcPtr_h24 = (d_half24 *)srcPtr;

    d_float24 src_f24;
    float2 src1_f2, src2_f2;

    src1_f2 = __half22float2(srcPtr_h24->x.x.x);
    src2_f2 = __half22float2(srcPtr_h24->x.x.y);
    src_f24.x.x = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write R00-R03
    src1_f2 = __half22float2(srcPtr_h24->x.y.x);
    src2_f2 = __half22float2(srcPtr_h24->x.y.y);
    src_f24.x.y = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write R04-R07
    src1_f2 = __half22float2(srcPtr_h24->y.x.x);
    src2_f2 = __half22float2(srcPtr_h24->y.x.y);
    src_f24.y.x = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write G00-G03
    src1_f2 = __half22float2(srcPtr_h24->y.y.x);
    src2_f2 = __half22float2(srcPtr_h24->y.y.y);
    src_f24.y.y = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write G04-G07
    src1_f2 = __half22float2(srcPtr_h24->z.x.x);
    src2_f2 = __half22float2(srcPtr_h24->z.x.y);
    src_f24.z.x = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write B00-B03
    src1_f2 = __half22float2(srcPtr_h24->z.y.x);
    src2_f2 = __half22float2(srcPtr_h24->z.y.y);
    src_f24.z.y = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write B04-B07

    rpp_hip_load24_pkd3_to_uchar8_pln3((float *)&src_f24, srcPtrs_uchar8);
}

// I8 loads with layout toggle PKD3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_to_uchar8_pln3(schar *srcPtr, uchar **srcPtrs_uchar8)
{
    d_uchar24 src_uchar24;
    rpp_hip_convert24_i8_to_u8(srcPtr, (uchar *)&src_uchar24);
    rpp_hip_load24_pkd3_to_uchar8_pln3((uchar *)&src_uchar24, srcPtrs_uchar8);
}

// -------------------- Set 7 - Templated layout toggles --------------------

// PKD3 to PLN3

template <typename T>
__device__ __forceinline__ void rpp_hip_layouttoggle24_pkd3_to_pln3(T *src)
{
    T pixpln3;

    pixpln3.x.x.x = src->x.x.x;
    pixpln3.x.x.y = src->x.x.w;
    pixpln3.x.x.z = src->x.y.z;
    pixpln3.x.x.w = src->y.x.y;
    pixpln3.x.y.x = src->y.y.x;
    pixpln3.x.y.y = src->y.y.w;
    pixpln3.x.y.z = src->z.x.z;
    pixpln3.x.y.w = src->z.y.y;

    pixpln3.y.x.x = src->x.x.y;
    pixpln3.y.x.y = src->x.y.x;
    pixpln3.y.x.z = src->x.y.w;
    pixpln3.y.x.w = src->y.x.z;
    pixpln3.y.y.x = src->y.y.y;
    pixpln3.y.y.y = src->z.x.x;
    pixpln3.y.y.z = src->z.x.w;
    pixpln3.y.y.w = src->z.y.z;

    pixpln3.z.x.x = src->x.x.z;
    pixpln3.z.x.y = src->x.y.y;
    pixpln3.z.x.z = src->y.x.x;
    pixpln3.z.x.w = src->y.x.w;
    pixpln3.z.y.x = src->y.y.z;
    pixpln3.z.y.y = src->z.x.y;
    pixpln3.z.y.z = src->z.y.x;
    pixpln3.z.y.w = src->z.y.w;

    *src = pixpln3;
}

// PLN3 to PKD3

template <typename T>
__device__ __forceinline__ void rpp_hip_layouttoggle24_pln3_to_pkd3(T *src)
{
    T pixpkd3;

    pixpkd3.x.x.x = src->x.x.x;
    pixpkd3.x.x.y = src->y.x.x;
    pixpkd3.x.x.z = src->z.x.x;

    pixpkd3.x.x.w = src->x.x.y;
    pixpkd3.x.y.x = src->y.x.y;
    pixpkd3.x.y.y = src->z.x.y;

    pixpkd3.x.y.z = src->x.x.z;
    pixpkd3.x.y.w = src->y.x.z;
    pixpkd3.y.x.x = src->z.x.z;

    pixpkd3.y.x.y = src->x.x.w;
    pixpkd3.y.x.z = src->y.x.w;
    pixpkd3.y.x.w = src->z.x.w;

    pixpkd3.y.y.x = src->x.y.x;
    pixpkd3.y.y.y = src->y.y.x;
    pixpkd3.y.y.z = src->z.y.x;

    pixpkd3.y.y.w = src->x.y.y;
    pixpkd3.z.x.x = src->y.y.y;
    pixpkd3.z.x.y = src->z.y.y;

    pixpkd3.z.x.z = src->x.y.z;
    pixpkd3.z.x.w = src->y.y.z;
    pixpkd3.z.y.x = src->z.y.z;

    pixpkd3.z.y.y = src->x.y.w;
    pixpkd3.z.y.z = src->y.y.w;
    pixpkd3.z.y.w = src->z.y.w;

    *src = pixpkd3;
}

/******************** DEVICE MATH HELPER FUNCTIONS ********************/

// d_float16 floor

__device__ __forceinline__ void rpp_hip_math_floor16(d_float16 *src_f16, d_float16 *dst_f16)
{
    dst_f16->x.x.x = floorf(src_f16->x.x.x);
    dst_f16->x.x.y = floorf(src_f16->x.x.y);
    dst_f16->x.x.z = floorf(src_f16->x.x.z);
    dst_f16->x.x.w = floorf(src_f16->x.x.w);
    dst_f16->x.y.x = floorf(src_f16->x.y.x);
    dst_f16->x.y.y = floorf(src_f16->x.y.y);
    dst_f16->x.y.z = floorf(src_f16->x.y.z);
    dst_f16->x.y.w = floorf(src_f16->x.y.w);

    dst_f16->y.x.x = floorf(src_f16->y.x.x);
    dst_f16->y.x.y = floorf(src_f16->y.x.y);
    dst_f16->y.x.z = floorf(src_f16->y.x.z);
    dst_f16->y.x.w = floorf(src_f16->y.x.w);
    dst_f16->y.y.x = floorf(src_f16->y.y.x);
    dst_f16->y.y.y = floorf(src_f16->y.y.y);
    dst_f16->y.y.z = floorf(src_f16->y.y.z);
    dst_f16->y.y.w = floorf(src_f16->y.y.w);
}

// d_float16 subtract

__device__ __forceinline__ void rpp_hip_math_subtract16(d_float16 *src1_f16, d_float16 *src2_f16, d_float16 *dst_f16)
{
    dst_f16->x.x = src1_f16->x.x - src2_f16->x.x;
    dst_f16->x.y = src1_f16->x.y - src2_f16->x.y;
    dst_f16->y.x = src1_f16->y.x - src2_f16->y.x;
    dst_f16->y.y = src1_f16->y.y - src2_f16->y.y;
}

// d_float24 multiply with constant

__device__ __forceinline__ void rpp_hip_math_multiply24_const(d_float24 *src_f24, d_float24 *dst_f24, float4 multiplier_f4)
{
    dst_f24->x.x = src_f24->x.x * multiplier_f4;
    dst_f24->x.y = src_f24->x.y * multiplier_f4;
    dst_f24->y.x = src_f24->y.x * multiplier_f4;
    dst_f24->y.y = src_f24->y.y * multiplier_f4;
    dst_f24->z.x = src_f24->z.x * multiplier_f4;
    dst_f24->z.y = src_f24->z.y * multiplier_f4;
}

// d_float24 add with constant

__device__ __forceinline__ void rpp_hip_math_add24_const(d_float24 *src_f24, d_float24 *dst_f24, float4 addend_f4)
{
    dst_f24->x.x = src_f24->x.x + addend_f4;
    dst_f24->x.y = src_f24->x.y + addend_f4;
    dst_f24->y.x = src_f24->y.x + addend_f4;
    dst_f24->y.y = src_f24->y.y + addend_f4;
    dst_f24->z.x = src_f24->z.x + addend_f4;
    dst_f24->z.y = src_f24->z.y + addend_f4;
}

// d_float24 subtract with constant

__device__ __forceinline__ void rpp_hip_math_subtract24_const(d_float24 *src_f24, d_float24 *dst_f24, float4 subtrahend_f4)
{
    dst_f24->x.x = src_f24->x.x - subtrahend_f4;
    dst_f24->x.y = src_f24->x.y - subtrahend_f4;
    dst_f24->y.x = src_f24->y.x - subtrahend_f4;
    dst_f24->y.y = src_f24->y.y - subtrahend_f4;
    dst_f24->z.x = src_f24->z.x - subtrahend_f4;
    dst_f24->z.y = src_f24->z.y - subtrahend_f4;
}

/******************** DEVICE INTERPOLATION HELPER FUNCTIONS ********************/

// BILINEAR INTERPOLATION LOAD HELPERS (separate load routines for each bit depth)

// U8 loads for bilinear interpolation (4 U8 pixels)

__device__ __forceinline__ void rpp_hip_interpolate1_bilinear_load_pln1(uchar *srcPtr, uint srcStrideH, float2 *locSrcFloor, float4 *srcNeighborhood_f4)
{
    uint src;
    int srcIdx = (int)locSrcFloor->y * srcStrideH + (int)locSrcFloor->x;
    src = *(uint *)&srcPtr[srcIdx];
    srcNeighborhood_f4->x = rpp_hip_unpack0(src);
    srcNeighborhood_f4->y = rpp_hip_unpack1(src);
    srcIdx += srcStrideH;
    src = *(uint *)&srcPtr[srcIdx];
    srcNeighborhood_f4->z = rpp_hip_unpack0(src);
    srcNeighborhood_f4->w = rpp_hip_unpack1(src);
}

// F32 loads for bilinear interpolation (4 F32 pixels)

__device__ __forceinline__ void rpp_hip_interpolate1_bilinear_load_pln1(float *srcPtr, uint srcStrideH, float2 *locSrcFloor, float4 *srcNeighborhood_f4)
{
    float2 src_f2;
    int srcIdx = (int)locSrcFloor->y * srcStrideH + (int)locSrcFloor->x;
    src_f2 = *(float2 *)&srcPtr[srcIdx];
    srcNeighborhood_f4->x = src_f2.x;
    srcNeighborhood_f4->y = src_f2.y;
    srcIdx += srcStrideH;
    src_f2 = *(float2 *)&srcPtr[srcIdx];
    srcNeighborhood_f4->z = src_f2.x;
    srcNeighborhood_f4->w = src_f2.y;
}

// I8 loads for bilinear interpolation (4 I8 pixels)

__device__ __forceinline__ void rpp_hip_interpolate1_bilinear_load_pln1(schar *srcPtr, uint srcStrideH, float2 *locSrcFloor, float4 *srcNeighborhood_f4)
{
    int src;
    int srcIdx = (int)locSrcFloor->y * srcStrideH + (int)locSrcFloor->x;
    src = *(int *)&srcPtr[srcIdx];
    srcNeighborhood_f4->x = rpp_hip_unpack0(src);
    srcNeighborhood_f4->y = rpp_hip_unpack1(src);
    srcIdx += srcStrideH;
    src = *(int *)&srcPtr[srcIdx];
    srcNeighborhood_f4->z = rpp_hip_unpack0(src);
    srcNeighborhood_f4->w = rpp_hip_unpack1(src);
}

// F16 loads for bilinear interpolation (4 F16 pixels)

__device__ __forceinline__ void rpp_hip_interpolate1_bilinear_load_pln1(half *srcPtr, uint srcStrideH, float2 *locSrcFloor, float4 *srcNeighborhood_f4)
{
    float2 srcUpper_f2, srcLower_f2;
    int srcIdx = (int)locSrcFloor->y * srcStrideH + (int)locSrcFloor->x;
    srcUpper_f2 = __half22float2(*(half2 *)&srcPtr[srcIdx]);
    srcIdx += srcStrideH;
    srcLower_f2 = __half22float2(*(half2 *)&srcPtr[srcIdx]);
    *srcNeighborhood_f4 = make_float4(srcUpper_f2.x, srcUpper_f2.y, srcLower_f2.x, srcLower_f2.y);
}

// U8 loads for bilinear interpolation (12 U8 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_load_pkd3(uchar *srcPtr, uint srcStrideH, float2 *locSrcFloor, d_float12 *srcNeighborhood_f12)
{
    uint2 src_u2;
    int srcIdx = (int)locSrcFloor->y * srcStrideH + (int)locSrcFloor->x * 3;
    src_u2 = *(uint2 *)&srcPtr[srcIdx];
    srcNeighborhood_f12->x.x = rpp_hip_unpack0(src_u2.x);
    srcNeighborhood_f12->x.y = rpp_hip_unpack3(src_u2.x);
    srcNeighborhood_f12->y.x = rpp_hip_unpack1(src_u2.x);
    srcNeighborhood_f12->y.y = rpp_hip_unpack0(src_u2.y);
    srcNeighborhood_f12->z.x = rpp_hip_unpack2(src_u2.x);
    srcNeighborhood_f12->z.y = rpp_hip_unpack1(src_u2.y);
    srcIdx += srcStrideH;
    src_u2 = *(uint2 *)&srcPtr[srcIdx];
    srcNeighborhood_f12->x.z = rpp_hip_unpack0(src_u2.x);
    srcNeighborhood_f12->x.w = rpp_hip_unpack3(src_u2.x);
    srcNeighborhood_f12->y.z = rpp_hip_unpack1(src_u2.x);
    srcNeighborhood_f12->y.w = rpp_hip_unpack0(src_u2.y);
    srcNeighborhood_f12->z.z = rpp_hip_unpack2(src_u2.x);
    srcNeighborhood_f12->z.w = rpp_hip_unpack1(src_u2.y);
}

// F32 loads for bilinear interpolation (12 F32 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_load_pkd3(float *srcPtr, uint srcStrideH, float2 *locSrcFloor, d_float12 *srcNeighborhood_f12)
{
    d_float6_as_float3s src_f6;
    int srcIdx = (int)locSrcFloor->y * srcStrideH + (int)locSrcFloor->x * 3;
    src_f6 = *(d_float6_as_float3s *)&srcPtr[srcIdx];
    srcNeighborhood_f12->x.x = src_f6.x.x;
    srcNeighborhood_f12->x.y = src_f6.y.x;
    srcNeighborhood_f12->y.x = src_f6.x.y;
    srcNeighborhood_f12->y.y = src_f6.y.y;
    srcNeighborhood_f12->z.x = src_f6.x.z;
    srcNeighborhood_f12->z.y = src_f6.y.z;
    srcIdx += srcStrideH;
    src_f6 = *(d_float6_as_float3s *)&srcPtr[srcIdx];
    srcNeighborhood_f12->x.z = src_f6.x.x;
    srcNeighborhood_f12->x.w = src_f6.y.x;
    srcNeighborhood_f12->y.z = src_f6.x.y;
    srcNeighborhood_f12->y.w = src_f6.y.y;
    srcNeighborhood_f12->z.z = src_f6.x.z;
    srcNeighborhood_f12->z.w = src_f6.y.z;
}

// I8 loads for bilinear interpolation (12 I8 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_load_pkd3(schar *srcPtr, uint srcStrideH, float2 *locSrcFloor, d_float12 *srcNeighborhood_f12)
{
    int2 src_i2;
    int srcIdx = (int)locSrcFloor->y * srcStrideH + (int)locSrcFloor->x * 3;
    src_i2 = *(int2 *)&srcPtr[srcIdx];
    srcNeighborhood_f12->x.x = rpp_hip_unpack0(src_i2.x);
    srcNeighborhood_f12->x.y = rpp_hip_unpack3(src_i2.x);
    srcNeighborhood_f12->y.x = rpp_hip_unpack1(src_i2.x);
    srcNeighborhood_f12->y.y = rpp_hip_unpack0(src_i2.y);
    srcNeighborhood_f12->z.x = rpp_hip_unpack2(src_i2.x);
    srcNeighborhood_f12->z.y = rpp_hip_unpack1(src_i2.y);
    srcIdx += srcStrideH;
    src_i2 = *(int2 *)&srcPtr[srcIdx];
    srcNeighborhood_f12->x.z = rpp_hip_unpack0(src_i2.x);
    srcNeighborhood_f12->x.w = rpp_hip_unpack3(src_i2.x);
    srcNeighborhood_f12->y.z = rpp_hip_unpack1(src_i2.x);
    srcNeighborhood_f12->y.w = rpp_hip_unpack0(src_i2.y);
    srcNeighborhood_f12->z.z = rpp_hip_unpack2(src_i2.x);
    srcNeighborhood_f12->z.w = rpp_hip_unpack1(src_i2.y);
}

// F16 loads for bilinear interpolation (12 F16 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_load_pkd3(half *srcPtr, uint srcStrideH, float2 *locSrcFloor, d_float12 *srcNeighborhood_f12)
{
    d_half6 src_h6;
    d_float6 src_f6;
    int srcIdx = (int)locSrcFloor->y * srcStrideH + (int)locSrcFloor->x * 3;
    src_h6 = *(d_half6 *)&srcPtr[srcIdx];
    src_f6.x = __half22float2(src_h6.x);
    src_f6.y = __half22float2(src_h6.y);
    src_f6.z = __half22float2(src_h6.z);
    srcNeighborhood_f12->x.x = src_f6.x.x;
    srcNeighborhood_f12->x.y = src_f6.y.y;
    srcNeighborhood_f12->y.x = src_f6.x.y;
    srcNeighborhood_f12->y.y = src_f6.z.x;
    srcNeighborhood_f12->z.x = src_f6.y.x;
    srcNeighborhood_f12->z.y = src_f6.z.y;
    srcIdx += srcStrideH;
    src_h6 = *(d_half6 *)&srcPtr[srcIdx];
    src_f6.x = __half22float2(src_h6.x);
    src_f6.y = __half22float2(src_h6.y);
    src_f6.z = __half22float2(src_h6.z);
    srcNeighborhood_f12->x.z = src_f6.x.x;
    srcNeighborhood_f12->x.w = src_f6.y.y;
    srcNeighborhood_f12->y.z = src_f6.x.y;
    srcNeighborhood_f12->y.w = src_f6.z.x;
    srcNeighborhood_f12->z.z = src_f6.y.x;
    srcNeighborhood_f12->z.w = src_f6.z.y;
}

// BILINEAR INTERPOLATION EXECUTION HELPERS (templated execution routines for all bit depths)

// float bilinear interpolation computation

__device__ __forceinline__ void rpp_hip_interpolate_bilinear(float4 *srcNeighborhood_f4, float2 *weightedWH, float2 *oneMinusWeightedWH, float *dst)
{
    *dst = fmaf(srcNeighborhood_f4->x, oneMinusWeightedWH->y * oneMinusWeightedWH->x,
                fmaf(srcNeighborhood_f4->y, oneMinusWeightedWH->y * weightedWH->x,
                    fmaf(srcNeighborhood_f4->z, weightedWH->y * oneMinusWeightedWH->x,
                        srcNeighborhood_f4->w * weightedWH->y * weightedWH->x)));
}

// float bilinear interpolation pln1

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate1_bilinear_pln1(T *srcPtr, uint srcStrideH, float locSrcX, float locSrcY, d_int4 *roiPtrSrc, float *dst)
{
    float2 locSrcFloor, weightedWH, oneMinusWeightedWH;
    locSrcFloor.x = floorf(locSrcX);
    locSrcFloor.y = floorf(locSrcY);
    if ((locSrcFloor.x < roiPtrSrc->x.x) || (locSrcFloor.y < roiPtrSrc->x.y) || (locSrcFloor.x > roiPtrSrc->y.x) || (locSrcFloor.y > roiPtrSrc->y.y))
    {
        *dst = 0.0f;
    }
    else
    {
        weightedWH.x = locSrcX - locSrcFloor.x;
        weightedWH.y = locSrcY - locSrcFloor.y;
        oneMinusWeightedWH.x = 1.0f - weightedWH.x;
        oneMinusWeightedWH.y = 1.0f - weightedWH.y;
        float4 srcNeighborhood_f4;
        rpp_hip_interpolate1_bilinear_load_pln1(srcPtr, srcStrideH, &locSrcFloor, &srcNeighborhood_f4);
        rpp_hip_interpolate_bilinear(&srcNeighborhood_f4, &weightedWH, &oneMinusWeightedWH, dst);
    }
}

// float3 bilinear interpolation pkd3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_pkd3(T *srcPtr, uint srcStrideH, float locSrcX, float locSrcY, d_int4 *roiPtrSrc, float3 *dst_f3)
{
    float2 locSrcFloor, weightedWH, oneMinusWeightedWH;
    locSrcFloor.x = floorf(locSrcX);
    locSrcFloor.y = floorf(locSrcY);
    if ((locSrcFloor.x < roiPtrSrc->x.x) || (locSrcFloor.y < roiPtrSrc->x.y) || (locSrcFloor.x > roiPtrSrc->y.x) || (locSrcFloor.y > roiPtrSrc->y.y))
    {
        *dst_f3 = (float3) 0.0f;
    }
    else
    {
        weightedWH.x = locSrcX - locSrcFloor.x;
        weightedWH.y = locSrcY - locSrcFloor.y;
        oneMinusWeightedWH.x = 1.0f - weightedWH.x;
        oneMinusWeightedWH.y = 1.0f - weightedWH.y;
        d_float12 srcNeighborhood_f12;
        rpp_hip_interpolate3_bilinear_load_pkd3(srcPtr, srcStrideH, &locSrcFloor, &srcNeighborhood_f12);
        rpp_hip_interpolate_bilinear(&srcNeighborhood_f12.x, &weightedWH, &oneMinusWeightedWH, &(dst_f3->x));
        rpp_hip_interpolate_bilinear(&srcNeighborhood_f12.y, &weightedWH, &oneMinusWeightedWH, &(dst_f3->y));
        rpp_hip_interpolate_bilinear(&srcNeighborhood_f12.z, &weightedWH, &oneMinusWeightedWH, &(dst_f3->z));
    }
}

// d_float8 bilinear interpolation in pln1

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate8_bilinear_pln1(T *srcPtr, uint srcStrideH, d_float16 *locPtrSrc_f16, d_int4 *roiPtrSrc, d_float8 *dst_f8)
{
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.x.x, locPtrSrc_f16->y.x.x, roiPtrSrc, &(dst_f8->x.x));
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.x.y, locPtrSrc_f16->y.x.y, roiPtrSrc, &(dst_f8->x.y));
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.x.z, locPtrSrc_f16->y.x.z, roiPtrSrc, &(dst_f8->x.z));
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.x.w, locPtrSrc_f16->y.x.w, roiPtrSrc, &(dst_f8->x.w));
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.y.x, locPtrSrc_f16->y.y.x, roiPtrSrc, &(dst_f8->y.x));
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.y.y, locPtrSrc_f16->y.y.y, roiPtrSrc, &(dst_f8->y.y));
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.y.z, locPtrSrc_f16->y.y.z, roiPtrSrc, &(dst_f8->y.z));
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.y.w, locPtrSrc_f16->y.y.w, roiPtrSrc, &(dst_f8->y.w));
}

// d_float24 bilinear interpolation in pln3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate24_bilinear_pln3(T *srcPtr, uint3 *srcStridesNCH, d_float16 *locPtrSrc_f16, d_int4 *roiPtrSrc, d_float24 *dst_f24)
{
    rpp_hip_interpolate8_bilinear_pln1(srcPtr, srcStridesNCH->z, locPtrSrc_f16, roiPtrSrc, &(dst_f24->x));
    srcPtr += srcStridesNCH->y;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr, srcStridesNCH->z, locPtrSrc_f16, roiPtrSrc, &(dst_f24->y));
    srcPtr += srcStridesNCH->y;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr, srcStridesNCH->z, locPtrSrc_f16, roiPtrSrc, &(dst_f24->z));
}

// d_float24 bilinear interpolation in pkd3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate24_bilinear_pkd3(T *srcPtr, uint srcStrideH, d_float16 *locPtrSrc_f16, d_int4 *roiPtrSrc, d_float24_as_float3s *dst_f24)
{
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.x.x, locPtrSrc_f16->y.x.x, roiPtrSrc, &(dst_f24->x.x));
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.x.y, locPtrSrc_f16->y.x.y, roiPtrSrc, &(dst_f24->x.y));
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.x.z, locPtrSrc_f16->y.x.z, roiPtrSrc, &(dst_f24->x.z));
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.x.w, locPtrSrc_f16->y.x.w, roiPtrSrc, &(dst_f24->x.w));
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.y.x, locPtrSrc_f16->y.y.x, roiPtrSrc, &(dst_f24->y.x));
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.y.y, locPtrSrc_f16->y.y.y, roiPtrSrc, &(dst_f24->y.y));
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.y.z, locPtrSrc_f16->y.y.z, roiPtrSrc, &(dst_f24->y.z));
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.y.w, locPtrSrc_f16->y.y.w, roiPtrSrc, &(dst_f24->y.w));
}

// NEAREST NEIGHBOR INTERPOLATION LOAD HELPERS (separate load routines for each bit depth)

// U8 loads for nearest_neighbor interpolation (1 U8 pixel)

__device__ __forceinline__ void rpp_hip_interpolate1_nearest_neighbor_load_pln1(uchar *srcPtr, int srcIdx, float *dst)
{
    uint src = *(uint *)&srcPtr[srcIdx];
    *dst = rpp_hip_unpack0(src);
}

// F32 loads for nearest_neighbor interpolation (1 F32 pixel)

__device__ __forceinline__ void rpp_hip_interpolate1_nearest_neighbor_load_pln1(float *srcPtr, int srcIdx, float *dst)
{
    *dst = srcPtr[srcIdx];
}

// I8 loads for nearest_neighbor interpolation (1 I8 pixel)

__device__ __forceinline__ void rpp_hip_interpolate1_nearest_neighbor_load_pln1(schar *srcPtr, int srcIdx, float *dst)
{
    int src = *(int *)&srcPtr[srcIdx];
    *dst = rpp_hip_unpack0(src);
}

// F16 loads for nearest_neighbor interpolation (1 F16 pixel)

__device__ __forceinline__ void rpp_hip_interpolate1_nearest_neighbor_load_pln1(half *srcPtr, int srcIdx, float *dst)
{
    *dst = __half2float(srcPtr[srcIdx]);
}

// U8 loads for nearest_neighbor interpolation (3 U8 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_nearest_neighbor_load_pkd3(uchar *srcPtr, int srcIdx, float3 *dst_f3)
{
    uint src = *(uint *)&srcPtr[srcIdx];
    *dst_f3 = make_float3(rpp_hip_unpack0(src), rpp_hip_unpack1(src), rpp_hip_unpack2(src));
}

// F32 loads for nearest_neighbor interpolation (3 F32 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_nearest_neighbor_load_pkd3(float *srcPtr, int srcIdx, float3 *dst_f3)
{
    float3 src_f3 = *(float3 *)&srcPtr[srcIdx];
    *dst_f3 = src_f3;
}

// I8 loads for nearest_neighbor interpolation (3 I8 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_nearest_neighbor_load_pkd3(schar *srcPtr, int srcIdx, float3 *dst_f3)
{
    int src = *(int *)&srcPtr[srcIdx];
    *dst_f3 = make_float3(rpp_hip_unpack0(src), rpp_hip_unpack1(src), rpp_hip_unpack2(src));
}

// F16 loads for nearest_neighbor interpolation (3 F16 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_nearest_neighbor_load_pkd3(half *srcPtr, int srcIdx, float3 *dst_f3)
{
    d_half3 src_h3 = *(d_half3 *)&srcPtr[srcIdx];
    dst_f3->x = __half2float(src_h3.x);
    dst_f3->y = __half2float(src_h3.y);
    dst_f3->z = __half2float(src_h3.z);
}

// NEAREST NEIGHBOR INTERPOLATION EXECUTION HELPERS (templated execution routines for all bit depths)

// float nearest neighbor interpolation pln1

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate1_nearest_neighbor_pln1(T *srcPtr, uint srcStrideH, float locSrcX, float locSrcY, d_int4 *roiPtrSrc, float *dst)
{
    int2 locSrc;
    locSrc.x = roundf(locSrcX);
    locSrc.y = roundf(locSrcY);

    if ((locSrc.x < roiPtrSrc->x.x) || (locSrc.y < roiPtrSrc->x.y) || (locSrc.x > roiPtrSrc->y.x) || (locSrc.y > roiPtrSrc->y.y))
    {
        *dst = 0.0f;
    }
    else
    {
        int srcIdx = locSrc.y * srcStrideH + locSrc.x;
        rpp_hip_interpolate1_nearest_neighbor_load_pln1(srcPtr, srcIdx, dst);
    }
}

// float3 nearest neighbor interpolation pkd3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate3_nearest_neighbor_pkd3(T *srcPtr, uint srcStrideH, float locSrcX, float locSrcY, d_int4 *roiPtrSrc, float3 *dst_f3)
{
    int2 locSrc;
    locSrc.x = roundf(locSrcX);
    locSrc.y = roundf(locSrcY);

    if ((locSrc.x < roiPtrSrc->x.x) || (locSrc.y < roiPtrSrc->x.y) || (locSrc.x > roiPtrSrc->y.x) || (locSrc.y > roiPtrSrc->y.y))
    {
        *dst_f3 = (float3) 0.0f;
    }
    else
    {
        uint src;
        int srcIdx = locSrc.y * srcStrideH + locSrc.x * 3;
        rpp_hip_interpolate3_nearest_neighbor_load_pkd3(srcPtr, srcIdx, dst_f3);
    }
}

// d_float8 nearest neighbor interpolation in pln1

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate8_nearest_neighbor_pln1(T *srcPtr, uint srcStrideH, d_float16 *locPtrSrc_f16, d_int4 *roiPtrSrc, d_float8 *dst_f8)
{
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.x.x, locPtrSrc_f16->y.x.x, roiPtrSrc, &(dst_f8->x.x));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.x.y, locPtrSrc_f16->y.x.y, roiPtrSrc, &(dst_f8->x.y));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.x.z, locPtrSrc_f16->y.x.z, roiPtrSrc, &(dst_f8->x.z));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.x.w, locPtrSrc_f16->y.x.w, roiPtrSrc, &(dst_f8->x.w));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.y.x, locPtrSrc_f16->y.y.x, roiPtrSrc, &(dst_f8->y.x));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.y.y, locPtrSrc_f16->y.y.y, roiPtrSrc, &(dst_f8->y.y));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.y.z, locPtrSrc_f16->y.y.z, roiPtrSrc, &(dst_f8->y.z));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->x.y.w, locPtrSrc_f16->y.y.w, roiPtrSrc, &(dst_f8->y.w));
}

// d_float24 nearest neighbor interpolation in pln3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate24_nearest_neighbor_pln3(T *srcPtr, uint3 *srcStridesNCH, d_float16 *locPtrSrc_f16, d_int4 *roiPtrSrc, d_float24 *dst_f24)
{
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr, srcStridesNCH->z, locPtrSrc_f16, roiPtrSrc, &(dst_f24->x));
    srcPtr += srcStridesNCH->y;
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr, srcStridesNCH->z, locPtrSrc_f16, roiPtrSrc, &(dst_f24->y));
    srcPtr += srcStridesNCH->y;
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr, srcStridesNCH->z, locPtrSrc_f16, roiPtrSrc, &(dst_f24->z));
}

// d_float24 nearest neighbor interpolation in pkd3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate24_nearest_neighbor_pkd3(T *srcPtr, uint srcStrideH, d_float16 *locPtrSrc_f16, d_int4 *roiPtrSrc, d_float24_as_float3s *dst_f24)
{
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.x.x, locPtrSrc_f16->y.x.x, roiPtrSrc, &(dst_f24->x.x));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.x.y, locPtrSrc_f16->y.x.y, roiPtrSrc, &(dst_f24->x.y));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.x.z, locPtrSrc_f16->y.x.z, roiPtrSrc, &(dst_f24->x.z));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.x.w, locPtrSrc_f16->y.x.w, roiPtrSrc, &(dst_f24->x.w));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.y.x, locPtrSrc_f16->y.y.x, roiPtrSrc, &(dst_f24->y.x));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.y.y, locPtrSrc_f16->y.y.y, roiPtrSrc, &(dst_f24->y.y));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.y.z, locPtrSrc_f16->y.y.z, roiPtrSrc, &(dst_f24->y.z));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->x.y.w, locPtrSrc_f16->y.y.w, roiPtrSrc, &(dst_f24->y.w));
}

#endif // RPP_HIP_COMMON_H