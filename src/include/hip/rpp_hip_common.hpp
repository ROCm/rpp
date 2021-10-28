#ifndef RPP_HIP_COMMON_H
#define RPP_HIP_COMMON_H

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <rppdefs.h>
#include <vector>
#include <half.hpp>
using halfhpp = half_float::half;
typedef halfhpp Rpp16f;

// float
typedef struct d_float8
{
    float4 x;
    float4 y;
} d_float8;
typedef struct d_float24
{
    d_float8 x;
    d_float8 y;
    d_float8 z;
} d_float24;

// uint
typedef struct d_uint6
{
    uint2 x;
    uint2 y;
    uint2 z;
} d_uint6;

// int
typedef struct d_int6
{
    int2 x;
    int2 y;
    int2 z;
} d_int6;

// half
typedef struct d_half4
{
    half2 x;
    half2 y;
} d_half4;
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

// uchar
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

__device__ __forceinline__ float4 rpp_hip_pixel_check(float4 src_f4)
{
    return make_float4(fminf(fmaxf(src_f4.x, 0), 255),
                       fminf(fmaxf(src_f4.y, 0), 255),
                       fminf(fmaxf(src_f4.z, 0), 255),
                       fminf(fmaxf(src_f4.w, 0), 255));
}

// d_float8 adjust pixel range for different bit depths

__device__ __forceinline__ void rpp_hip_adjust_range(uchar *dstPtr, d_float8 *sum_f8)
{
}

__device__ __forceinline__ void rpp_hip_adjust_range(float *dstPtr, d_float8 *sum_f8)
{
    sum_f8->x = sum_f8->x * (float4) 0.00392157;
    sum_f8->y = sum_f8->y * (float4) 0.00392157;
}

__device__ __forceinline__ void rpp_hip_adjust_range(signed char *dstPtr, d_float8 *sum_f8)
{
    sum_f8->x = sum_f8->x - (float4) 128;
    sum_f8->y = sum_f8->y - (float4) 128;
}

__device__ __forceinline__ void rpp_hip_adjust_range(half *dstPtr, d_float8 *sum_f8)
{
    sum_f8->x = sum_f8->x * (float4) 0.00392157;
    sum_f8->y = sum_f8->y * (float4) 0.00392157;
}

// d_float24 adjust pixel range for different bit depths

__device__ __forceinline__ void rpp_hip_adjust_range(uchar *dstPtr, d_float24 *sum_f24)
{
}

__device__ __forceinline__ void rpp_hip_adjust_range(float *dstPtr, d_float24 *sum_f24)
{
    sum_f24->x.x = sum_f24->x.x * (float4) 0.00392157;
    sum_f24->x.y = sum_f24->x.y * (float4) 0.00392157;
    sum_f24->y.x = sum_f24->y.x * (float4) 0.00392157;
    sum_f24->y.y = sum_f24->y.y * (float4) 0.00392157;
    sum_f24->z.x = sum_f24->z.x * (float4) 0.00392157;
    sum_f24->z.y = sum_f24->z.y * (float4) 0.00392157;
}

__device__ __forceinline__ void rpp_hip_adjust_range(signed char *dstPtr, d_float24 *sum_f24)
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
    sum_f24->x.x = sum_f24->x.x * (float4) 0.00392157;
    sum_f24->x.y = sum_f24->x.y * (float4) 0.00392157;
    sum_f24->y.x = sum_f24->y.x * (float4) 0.00392157;
    sum_f24->y.y = sum_f24->y.y * (float4) 0.00392157;
    sum_f24->z.x = sum_f24->z.x * (float4) 0.00392157;
    sum_f24->z.y = sum_f24->z.y * (float4) 0.00392157;
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
    dst_c4.w = (signed char)(src.w);
    dst_c4.z = (signed char)(src.z);
    dst_c4.y = (signed char)(src.y);
    dst_c4.x = (signed char)(src.x);

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

// Un-Packing from I8s

__device__ __forceinline__ float rpp_hip_unpack0(int src)
{
    return (float)(signed char)(src & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack1(int src)
{
    return (float)(signed char)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack2(int src)
{
    return (float)(signed char)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack3(int src)
{
    return (float)(signed char)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float4 rpp_hip_unpack_from_i8(int src)
{
    return make_float4(rpp_hip_unpack0(src), rpp_hip_unpack1(src), rpp_hip_unpack2(src), rpp_hip_unpack3(src));
}

// -------------------- Set 3 - Bit Depth Conversions --------------------

// I8 to U8 conversions (8 pixels)

__device__ __forceinline__ void rpp_hip_convert8_i8_to_u8(signed char *srcPtr, uchar *dstPtr)
{
    int2 *srcPtr_8;
    srcPtr_8 = (int2 *)srcPtr;

    uint2 *dstPtr_8;
    dstPtr_8 = (uint2 *)dstPtr;

    dstPtr_8->x = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_8->x) + (float4) 128);
    dstPtr_8->y = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_8->y) + (float4) 128);
}

// I8 to U8 conversions (24 pixels)

__device__ __forceinline__ void rpp_hip_convert24_i8_to_u8(signed char *srcPtr, uchar *dstPtr)
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

// -------------------- Set 4 - Loads --------------------

// WITHOUT LAYOUT TOGGLE

// U8 loads without layout toggle (8 U8 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(uchar *srcPtr, int srcIdx, d_float8 *src_f8)
{
    uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
    src_f8->x = rpp_hip_unpack(src.x);
    src_f8->y = rpp_hip_unpack(src.y);
}

// F32 loads without layout toggle (8 F32 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(float *srcPtr, int srcIdx, d_float8 *src_f8)
{
    *src_f8 = *((d_float8 *)(&srcPtr[srcIdx]));
}

// I8 loads without layout toggle (8 I8 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(signed char *srcPtr, int srcIdx, d_float8 *src_f8)
{
    int2 src = *((int2 *)(&srcPtr[srcIdx]));
    src_f8->x = rpp_hip_unpack_from_i8(src.x);
    src_f8->y = rpp_hip_unpack_from_i8(src.y);
}

// F16 loads without layout toggle (8 F16 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(half *srcPtr, int srcIdx, d_float8 *src_f8)
{
    d_half8 src_h8;
    src_h8 = *((d_half8 *)(&srcPtr[srcIdx]));

    float2 src1_f2, src2_f2;

    src1_f2 = __half22float2(src_h8.x.x);
    src2_f2 = __half22float2(src_h8.x.y);
    src_f8->x = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);

    src1_f2 = __half22float2(src_h8.y.x);
    src2_f2 = __half22float2(src_h8.y.y);
    src_f8->y = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);
}

// U8 loads without layout toggle PLN3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3(uchar *srcPtr, int srcIdx, uint increment, d_float24 *src_f24)
{
    d_uint6 src;

    src.x = *((uint2 *)(&srcPtr[srcIdx]));
    srcIdx += increment;
    src.y = *((uint2 *)(&srcPtr[srcIdx]));
    srcIdx += increment;
    src.z = *((uint2 *)(&srcPtr[srcIdx]));

    src_f24->x.x = rpp_hip_unpack(src.x.x);
    src_f24->x.y = rpp_hip_unpack(src.x.y);
    src_f24->y.x = rpp_hip_unpack(src.y.x);
    src_f24->y.y = rpp_hip_unpack(src.y.y);
    src_f24->z.x = rpp_hip_unpack(src.z.x);
    src_f24->z.y = rpp_hip_unpack(src.z.y);
}

// F32 loads without layout toggle PLN3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3(float *srcPtr, int srcIdx, uint increment, d_float24 *src_f24)
{
    float *srcPtrR, *srcPtrG, *srcPtrB;
    srcPtrR = srcPtr + srcIdx;
    srcPtrG = srcPtrR + increment;
    srcPtrB = srcPtrG + increment;

    src_f24->x = *(d_float8 *)srcPtrR;
    src_f24->y = *(d_float8 *)srcPtrG;
    src_f24->z = *(d_float8 *)srcPtrB;
}

// I8 loads without layout toggle PLN3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3(signed char *srcPtr, int srcIdx, uint increment, d_float24 *src_f24)
{
    d_int6 src;

    src.x = *((int2 *)(&srcPtr[srcIdx]));
    srcIdx += increment;
    src.y = *((int2 *)(&srcPtr[srcIdx]));
    srcIdx += increment;
    src.z = *((int2 *)(&srcPtr[srcIdx]));

    src_f24->x.x = rpp_hip_unpack_from_i8(src.x.x);
    src_f24->x.y = rpp_hip_unpack_from_i8(src.x.y);
    src_f24->y.x = rpp_hip_unpack_from_i8(src.y.x);
    src_f24->y.y = rpp_hip_unpack_from_i8(src.y.y);
    src_f24->z.x = rpp_hip_unpack_from_i8(src.z.x);
    src_f24->z.y = rpp_hip_unpack_from_i8(src.z.y);
}

// F16 loads without layout toggle PLN3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3(half *srcPtr, int srcIdx, uint increment, d_float24 *src_f24)
{
    half *srcPtrR, *srcPtrG, *srcPtrB;
    srcPtrR = srcPtr + srcIdx;
    srcPtrG = srcPtrR + increment;
    srcPtrB = srcPtrG + increment;

    d_half8 *srcR_h8, *srcG_h8, *srcB_h8;
    srcR_h8 = (d_half8 *)srcPtrR;
    srcG_h8 = (d_half8 *)srcPtrG;
    srcB_h8 = (d_half8 *)srcPtrB;

    src_f24->x.x.x = __half2float(__low2half(srcR_h8->x.x));
    src_f24->x.x.y = __half2float(__high2half(srcR_h8->x.x));
    src_f24->x.x.z = __half2float(__low2half(srcR_h8->x.y));
    src_f24->x.x.w = __half2float(__high2half(srcR_h8->x.y));
    src_f24->x.y.x = __half2float(__low2half(srcR_h8->y.x));
    src_f24->x.y.y = __half2float(__high2half(srcR_h8->y.x));
    src_f24->x.y.z = __half2float(__low2half(srcR_h8->y.y));
    src_f24->x.y.w = __half2float(__high2half(srcR_h8->y.y));

    src_f24->y.x.x = __half2float(__low2half(srcG_h8->x.x));
    src_f24->y.x.y = __half2float(__high2half(srcG_h8->x.x));
    src_f24->y.x.z = __half2float(__low2half(srcG_h8->x.y));
    src_f24->y.x.w = __half2float(__high2half(srcG_h8->x.y));
    src_f24->y.y.x = __half2float(__low2half(srcG_h8->y.x));
    src_f24->y.y.y = __half2float(__high2half(srcG_h8->y.x));
    src_f24->y.y.z = __half2float(__low2half(srcG_h8->y.y));
    src_f24->y.y.w = __half2float(__high2half(srcG_h8->y.y));

    src_f24->z.x.x = __half2float(__low2half(srcB_h8->x.x));
    src_f24->z.x.y = __half2float(__high2half(srcB_h8->x.x));
    src_f24->z.x.z = __half2float(__low2half(srcB_h8->x.y));
    src_f24->z.x.w = __half2float(__high2half(srcB_h8->x.y));
    src_f24->z.y.x = __half2float(__low2half(srcB_h8->y.x));
    src_f24->z.y.y = __half2float(__high2half(srcB_h8->y.x));
    src_f24->z.y.z = __half2float(__low2half(srcB_h8->y.y));
    src_f24->z.y.w = __half2float(__high2half(srcB_h8->y.y));
}

// WITH LAYOUT TOGGLE

// U8 loads with layout toggle PKD3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(uchar *srcPtr, int srcIdx, d_float24 *src_f24)
{
    d_uint6 src = *((d_uint6 *)(&srcPtr[srcIdx]));

    src_f24->x.x = make_float4(rpp_hip_unpack0(src.x.x), rpp_hip_unpack3(src.x.x), rpp_hip_unpack2(src.x.y), rpp_hip_unpack1(src.y.x));
    src_f24->x.y = make_float4(rpp_hip_unpack0(src.y.y), rpp_hip_unpack3(src.y.y), rpp_hip_unpack2(src.z.x), rpp_hip_unpack1(src.z.y));

    src_f24->y.x = make_float4(rpp_hip_unpack1(src.x.x), rpp_hip_unpack0(src.x.y), rpp_hip_unpack3(src.x.y), rpp_hip_unpack2(src.y.x));
    src_f24->y.y = make_float4(rpp_hip_unpack1(src.y.y), rpp_hip_unpack0(src.z.x), rpp_hip_unpack3(src.z.x), rpp_hip_unpack2(src.z.y));

    src_f24->z.x = make_float4(rpp_hip_unpack2(src.x.x), rpp_hip_unpack1(src.x.y), rpp_hip_unpack0(src.y.x), rpp_hip_unpack3(src.y.x));
    src_f24->z.y = make_float4(rpp_hip_unpack2(src.y.y), rpp_hip_unpack1(src.z.x), rpp_hip_unpack0(src.z.y), rpp_hip_unpack3(src.z.y));
}

// F32 loads with layout toggle PKD3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(float *srcPtr, int srcIdx, d_float24 *src_f24)
{
    d_float24 *srcPtr_f24;
    srcPtr_f24 = (d_float24 *)&srcPtr[srcIdx];

    src_f24->x.x.x = srcPtr_f24->x.x.x;
    src_f24->y.x.x = srcPtr_f24->x.x.y;
    src_f24->z.x.x = srcPtr_f24->x.x.z;
    src_f24->x.x.y = srcPtr_f24->x.x.w;
    src_f24->y.x.y = srcPtr_f24->x.y.x;
    src_f24->z.x.y = srcPtr_f24->x.y.y;
    src_f24->x.x.z = srcPtr_f24->x.y.z;
    src_f24->y.x.z = srcPtr_f24->x.y.w;

    src_f24->z.x.z = srcPtr_f24->y.x.x;
    src_f24->x.x.w = srcPtr_f24->y.x.y;
    src_f24->y.x.w = srcPtr_f24->y.x.z;
    src_f24->z.x.w = srcPtr_f24->y.x.w;
    src_f24->x.y.x = srcPtr_f24->y.y.x;
    src_f24->y.y.x = srcPtr_f24->y.y.y;
    src_f24->z.y.x = srcPtr_f24->y.y.z;
    src_f24->x.y.y = srcPtr_f24->y.y.w;

    src_f24->y.y.y = srcPtr_f24->z.x.x;
    src_f24->z.y.y = srcPtr_f24->z.x.y;
    src_f24->x.y.z = srcPtr_f24->z.x.z;
    src_f24->y.y.z = srcPtr_f24->z.x.w;
    src_f24->z.y.z = srcPtr_f24->z.y.x;
    src_f24->x.y.w = srcPtr_f24->z.y.y;
    src_f24->y.y.w = srcPtr_f24->z.y.z;
    src_f24->z.y.w = srcPtr_f24->z.y.w;
}

// I8 loads with layout toggle PKD3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(signed char *srcPtr, int srcIdx, d_float24 *src_f24)
{
    d_int6 src = *((d_int6 *)(&srcPtr[srcIdx]));

    src_f24->x.x = make_float4(rpp_hip_unpack0(src.x.x), rpp_hip_unpack3(src.x.x), rpp_hip_unpack2(src.x.y), rpp_hip_unpack1(src.y.x));
    src_f24->x.y = make_float4(rpp_hip_unpack0(src.y.y), rpp_hip_unpack3(src.y.y), rpp_hip_unpack2(src.z.x), rpp_hip_unpack1(src.z.y));

    src_f24->y.x = make_float4(rpp_hip_unpack1(src.x.x), rpp_hip_unpack0(src.x.y), rpp_hip_unpack3(src.x.y), rpp_hip_unpack2(src.y.x));
    src_f24->y.y = make_float4(rpp_hip_unpack1(src.y.y), rpp_hip_unpack0(src.z.x), rpp_hip_unpack3(src.z.x), rpp_hip_unpack2(src.z.y));

    src_f24->z.x = make_float4(rpp_hip_unpack2(src.x.x), rpp_hip_unpack1(src.x.y), rpp_hip_unpack0(src.y.x), rpp_hip_unpack3(src.y.x));
    src_f24->z.y = make_float4(rpp_hip_unpack2(src.y.y), rpp_hip_unpack1(src.z.x), rpp_hip_unpack0(src.z.y), rpp_hip_unpack3(src.z.y));
}

// F16 loads with layout toggle PKD3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(half *srcPtr, int srcIdx, d_float24 *src_f24)
{
    d_half24 *src_h24;
    src_h24 = (d_half24 *)&srcPtr[srcIdx];

    src_f24->x.x.x = __half2float(__low2half(src_h24->x.x.x));
    src_f24->x.x.y = __half2float(__high2half(src_h24->x.x.y));
    src_f24->x.x.z = __half2float(__low2half(src_h24->x.y.y));
    src_f24->x.x.w = __half2float(__high2half(src_h24->y.x.x));
    src_f24->x.y.x = __half2float(__low2half(src_h24->y.y.x));
    src_f24->x.y.y = __half2float(__high2half(src_h24->y.y.y));
    src_f24->x.y.z = __half2float(__low2half(src_h24->z.x.y));
    src_f24->x.y.w = __half2float(__high2half(src_h24->z.y.x));

    src_f24->y.x.x = __half2float(__high2half(src_h24->x.x.x));
    src_f24->y.x.y = __half2float(__low2half(src_h24->x.y.x));
    src_f24->y.x.z = __half2float(__high2half(src_h24->x.y.y));
    src_f24->y.x.w = __half2float(__low2half(src_h24->y.x.y));
    src_f24->y.y.x = __half2float(__high2half(src_h24->y.y.x));
    src_f24->y.y.y = __half2float(__low2half(src_h24->z.x.x));
    src_f24->y.y.z = __half2float(__high2half(src_h24->z.x.y));
    src_f24->y.y.w = __half2float(__low2half(src_h24->z.y.y));

    src_f24->z.x.x = __half2float(__low2half(src_h24->x.x.y));
    src_f24->z.x.y = __half2float(__high2half(src_h24->x.y.x));
    src_f24->z.x.z = __half2float(__low2half(src_h24->y.x.x));
    src_f24->z.x.w = __half2float(__high2half(src_h24->y.x.y));
    src_f24->z.y.x = __half2float(__low2half(src_h24->y.y.y));
    src_f24->z.y.y = __half2float(__high2half(src_h24->z.x.x));
    src_f24->z.y.z = __half2float(__low2half(src_h24->z.y.x));
    src_f24->z.y.w = __half2float(__high2half(src_h24->z.y.y));
}

// U8 loads with layout toggle PLN3 to PKD3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(uchar *srcPtr, int srcIdx, uint increment, d_float24 *src_f24)
{
    d_uint6 src;

    src.x = *((uint2 *)(&srcPtr[srcIdx]));
    srcIdx += increment;
    src.y = *((uint2 *)(&srcPtr[srcIdx]));
    srcIdx += increment;
    src.z = *((uint2 *)(&srcPtr[srcIdx]));

    src_f24->x.x = make_float4(rpp_hip_unpack0(src.x.x), rpp_hip_unpack0(src.y.x), rpp_hip_unpack0(src.z.x), rpp_hip_unpack1(src.x.x));
    src_f24->x.y = make_float4(rpp_hip_unpack1(src.y.x), rpp_hip_unpack1(src.z.x), rpp_hip_unpack2(src.x.x), rpp_hip_unpack2(src.y.x));
    src_f24->y.x = make_float4(rpp_hip_unpack2(src.z.x), rpp_hip_unpack3(src.x.x), rpp_hip_unpack3(src.y.x), rpp_hip_unpack3(src.z.x));
    src_f24->y.y = make_float4(rpp_hip_unpack0(src.x.y), rpp_hip_unpack0(src.y.y), rpp_hip_unpack0(src.z.y), rpp_hip_unpack1(src.x.y));
    src_f24->z.x = make_float4(rpp_hip_unpack1(src.y.y), rpp_hip_unpack1(src.z.y), rpp_hip_unpack2(src.x.y), rpp_hip_unpack2(src.y.y));
    src_f24->z.y = make_float4(rpp_hip_unpack2(src.z.y), rpp_hip_unpack3(src.x.y), rpp_hip_unpack3(src.y.y), rpp_hip_unpack3(src.z.y));
}

// F32 loads with layout toggle PLN3 to PKD3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(float *srcPtr, int srcIdx, uint increment, d_float24 *src_f24)
{
    float *srcPtrR, *srcPtrG, *srcPtrB;
    srcPtrR = srcPtr + srcIdx;
    srcPtrG = srcPtrR + increment;
    srcPtrB = srcPtrG + increment;

    d_float8 *srcPtrR_f8, *srcPtrG_f8, *srcPtrB_f8;

    srcPtrR_f8 = (d_float8 *)srcPtrR;
    srcPtrG_f8 = (d_float8 *)srcPtrG;
    srcPtrB_f8 = (d_float8 *)srcPtrB;

    src_f24->x.x.x = srcPtrR_f8->x.x;
    src_f24->x.x.y = srcPtrG_f8->x.x;
    src_f24->x.x.z = srcPtrB_f8->x.x;

    src_f24->x.x.w = srcPtrR_f8->x.y;
    src_f24->x.y.x = srcPtrG_f8->x.y;
    src_f24->x.y.y = srcPtrB_f8->x.y;

    src_f24->x.y.z = srcPtrR_f8->x.z;
    src_f24->x.y.w = srcPtrG_f8->x.z;
    src_f24->y.x.x = srcPtrB_f8->x.z;

    src_f24->y.x.y = srcPtrR_f8->x.w;
    src_f24->y.x.z = srcPtrG_f8->x.w;
    src_f24->y.x.w = srcPtrB_f8->x.w;

    src_f24->y.y.x = srcPtrR_f8->y.x;
    src_f24->y.y.y = srcPtrG_f8->y.x;
    src_f24->y.y.z = srcPtrB_f8->y.x;

    src_f24->y.y.w = srcPtrR_f8->y.y;
    src_f24->z.x.x = srcPtrG_f8->y.y;
    src_f24->z.x.y = srcPtrB_f8->y.y;

    src_f24->z.x.z = srcPtrR_f8->y.z;
    src_f24->z.x.w = srcPtrG_f8->y.z;
    src_f24->z.y.x = srcPtrB_f8->y.z;

    src_f24->z.y.y = srcPtrR_f8->y.w;
    src_f24->z.y.z = srcPtrG_f8->y.w;
    src_f24->z.y.w = srcPtrB_f8->y.w;
}

// I8 loads with layout toggle PLN3 to PKD3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(signed char *srcPtr, int srcIdx, uint increment, d_float24 *src_f24)
{
    d_int6 src;

    src.x = *((int2 *)(&srcPtr[srcIdx]));
    srcIdx += increment;
    src.y = *((int2 *)(&srcPtr[srcIdx]));
    srcIdx += increment;
    src.z = *((int2 *)(&srcPtr[srcIdx]));

    src_f24->x.x = make_float4(rpp_hip_unpack0(src.x.x), rpp_hip_unpack0(src.y.x), rpp_hip_unpack0(src.z.x), rpp_hip_unpack1(src.x.x));
    src_f24->x.y = make_float4(rpp_hip_unpack1(src.y.x), rpp_hip_unpack1(src.z.x), rpp_hip_unpack2(src.x.x), rpp_hip_unpack2(src.y.x));
    src_f24->y.x = make_float4(rpp_hip_unpack2(src.z.x), rpp_hip_unpack3(src.x.x), rpp_hip_unpack3(src.y.x), rpp_hip_unpack3(src.z.x));
    src_f24->y.y = make_float4(rpp_hip_unpack0(src.x.y), rpp_hip_unpack0(src.y.y), rpp_hip_unpack0(src.z.y), rpp_hip_unpack1(src.x.y));
    src_f24->z.x = make_float4(rpp_hip_unpack1(src.y.y), rpp_hip_unpack1(src.z.y), rpp_hip_unpack2(src.x.y), rpp_hip_unpack2(src.y.y));
    src_f24->z.y = make_float4(rpp_hip_unpack2(src.z.y), rpp_hip_unpack3(src.x.y), rpp_hip_unpack3(src.y.y), rpp_hip_unpack3(src.z.y));
}

// F16 loads with layout toggle PLN3 to PKD3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(half *srcPtr, int srcIdx, uint increment, d_float24 *src_f24)
{
    half *srcPtrR, *srcPtrG, *srcPtrB;
    srcPtrR = srcPtr + srcIdx;
    srcPtrG = srcPtrR + increment;
    srcPtrB = srcPtrG + increment;

    d_half8 *srcR_h8, *srcG_h8, *srcB_h8;
    srcR_h8 = (d_half8 *)srcPtrR;
    srcG_h8 = (d_half8 *)srcPtrG;
    srcB_h8 = (d_half8 *)srcPtrB;

    src_f24->x.x.x = __half2float(__low2half(srcR_h8->x.x));
    src_f24->x.x.y = __half2float(__low2half(srcG_h8->x.x));
    src_f24->x.x.z = __half2float(__low2half(srcB_h8->x.x));

    src_f24->x.x.w = __half2float(__high2half(srcR_h8->x.x));
    src_f24->x.y.x = __half2float(__high2half(srcG_h8->x.x));
    src_f24->x.y.y = __half2float(__high2half(srcB_h8->x.x));

    src_f24->x.y.z = __half2float(__low2half(srcR_h8->x.y));
    src_f24->x.y.w = __half2float(__low2half(srcG_h8->x.y));
    src_f24->y.x.x = __half2float(__low2half(srcB_h8->x.y));

    src_f24->y.x.y = __half2float(__high2half(srcR_h8->x.y));
    src_f24->y.x.z = __half2float(__high2half(srcG_h8->x.y));
    src_f24->y.x.w = __half2float(__high2half(srcB_h8->x.y));

    src_f24->y.y.x = __half2float(__low2half(srcR_h8->y.x));
    src_f24->y.y.y = __half2float(__low2half(srcG_h8->y.x));
    src_f24->y.y.z = __half2float(__low2half(srcB_h8->y.x));

    src_f24->y.y.w = __half2float(__high2half(srcR_h8->y.x));
    src_f24->z.x.x = __half2float(__high2half(srcG_h8->y.x));
    src_f24->z.x.y = __half2float(__high2half(srcB_h8->y.x));

    src_f24->z.x.z = __half2float(__low2half(srcR_h8->y.y));
    src_f24->z.x.w = __half2float(__low2half(srcG_h8->y.y));
    src_f24->z.y.x = __half2float(__low2half(srcB_h8->y.y));

    src_f24->z.y.y = __half2float(__high2half(srcR_h8->y.y));
    src_f24->z.y.z = __half2float(__high2half(srcG_h8->y.y));
    src_f24->z.y.w = __half2float(__high2half(srcB_h8->y.y));
}

// -------------------- Set 5 - Stores --------------------

// WITHOUT LAYOUT TOGGLE

// U8 stores without layout toggle (8 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(uchar *dstPtr, uint dstIdx, d_float8 *dst_f8)
{
    uint2 dst;
    dst.x = rpp_hip_pack(dst_f8->x);
    dst.y = rpp_hip_pack(dst_f8->y);
    *((uint2 *)(&dstPtr[dstIdx])) = dst;
}

// F32 stores without layout toggle (8 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(float *dstPtr, uint dstIdx, d_float8 *dst_f8)
{
    *((d_float8 *)(&dstPtr[dstIdx])) = *dst_f8;
}

// I8 stores without layout toggle (8 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(signed char *dstPtr, uint dstIdx, d_float8 *dst_f8)
{
    uint2 dst;
    dst.x = rpp_hip_pack_i8(dst_f8->x);
    dst.y = rpp_hip_pack_i8(dst_f8->y);
    *((uint2 *)(&dstPtr[dstIdx])) = dst;
}

// F16 stores without layout toggle (8 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(half *dstPtr, uint dstIdx, d_float8 *dst_f8)
{
    d_half8 dst_h8;

    dst_h8.x.x = __float22half2_rn(make_float2(dst_f8->x.x, dst_f8->x.y));
    dst_h8.x.y = __float22half2_rn(make_float2(dst_f8->x.z, dst_f8->x.w));
    dst_h8.y.x = __float22half2_rn(make_float2(dst_f8->y.x, dst_f8->y.y));
    dst_h8.y.y = __float22half2_rn(make_float2(dst_f8->y.z, dst_f8->y.w));

    *((d_half8 *)(&dstPtr[dstIdx])) = dst_h8;
}

// U8 stores without layout toggle PKD3 to PKD3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(uchar *dstPtr, uint dstIdx, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack(dst_f24->x.x);
    dst.x.y = rpp_hip_pack(dst_f24->x.y);
    dst.y.x = rpp_hip_pack(dst_f24->y.x);
    dst.y.y = rpp_hip_pack(dst_f24->y.y);
    dst.z.x = rpp_hip_pack(dst_f24->z.x);
    dst.z.y = rpp_hip_pack(dst_f24->z.y);

    *((d_uint6 *)(&dstPtr[dstIdx])) = dst;
}

// F32 stores without layout toggle PKD3 to PKD3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(float *dstPtr, uint dstIdx, d_float24 *dst_f24)
{
    *((d_float24 *)(&dstPtr[dstIdx])) = *dst_f24;
}

// I8 stores without layout toggle PKD3 to PKD3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(signed char *dstPtr, uint dstIdx, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack_i8(dst_f24->x.x);
    dst.x.y = rpp_hip_pack_i8(dst_f24->x.y);
    dst.y.x = rpp_hip_pack_i8(dst_f24->y.x);
    dst.y.y = rpp_hip_pack_i8(dst_f24->y.y);
    dst.z.x = rpp_hip_pack_i8(dst_f24->z.x);
    dst.z.y = rpp_hip_pack_i8(dst_f24->z.y);

    *((d_uint6 *)(&dstPtr[dstIdx])) = dst;
}

// F16 stores without layout toggle PKD3 to PKD3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(half *dstPtr, uint dstIdx, d_float24 *dst_f24)
{
    d_half24 dst_h24;

    dst_h24.x.x.x = __float22half2_rn(make_float2(dst_f24->x.x.x, dst_f24->x.x.y));
    dst_h24.x.x.y = __float22half2_rn(make_float2(dst_f24->x.x.z, dst_f24->x.x.w));
    dst_h24.x.y.x = __float22half2_rn(make_float2(dst_f24->x.y.x, dst_f24->x.y.y));
    dst_h24.x.y.y = __float22half2_rn(make_float2(dst_f24->x.y.z, dst_f24->x.y.w));

    dst_h24.y.x.x = __float22half2_rn(make_float2(dst_f24->y.x.x, dst_f24->y.x.y));
    dst_h24.y.x.y = __float22half2_rn(make_float2(dst_f24->y.x.z, dst_f24->y.x.w));
    dst_h24.y.y.x = __float22half2_rn(make_float2(dst_f24->y.y.x, dst_f24->y.y.y));
    dst_h24.y.y.y = __float22half2_rn(make_float2(dst_f24->y.y.z, dst_f24->y.y.w));

    dst_h24.z.x.x = __float22half2_rn(make_float2(dst_f24->z.x.x, dst_f24->z.x.y));
    dst_h24.z.x.y = __float22half2_rn(make_float2(dst_f24->z.x.z, dst_f24->z.x.w));
    dst_h24.z.y.x = __float22half2_rn(make_float2(dst_f24->z.y.x, dst_f24->z.y.y));
    dst_h24.z.y.y = __float22half2_rn(make_float2(dst_f24->z.y.z, dst_f24->z.y.w));

    *((d_half24 *)(&dstPtr[dstIdx])) = dst_h24;
}

// U8 stores without layout toggle PLN3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pln3(uchar *dstPtr, uint dstIdx, uint increment, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack(dst_f24->x.x);
    dst.x.y = rpp_hip_pack(dst_f24->x.y);
    dst.y.x = rpp_hip_pack(dst_f24->y.x);
    dst.y.y = rpp_hip_pack(dst_f24->y.y);
    dst.z.x = rpp_hip_pack(dst_f24->z.x);
    dst.z.y = rpp_hip_pack(dst_f24->z.y);

    *((uint2 *)(&dstPtr[dstIdx])) = dst.x;
    dstIdx += increment;
    *((uint2 *)(&dstPtr[dstIdx])) = dst.y;
    dstIdx += increment;
    *((uint2 *)(&dstPtr[dstIdx])) = dst.z;
}

// F32 stores without layout toggle PLN3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pln3(float *dstPtr, uint dstIdx, uint increment, d_float24 *dst_f24)
{
    *((d_float8 *)(&dstPtr[dstIdx])) = dst_f24->x;
    dstIdx += increment;
    *((d_float8 *)(&dstPtr[dstIdx])) = dst_f24->y;
    dstIdx += increment;
    *((d_float8 *)(&dstPtr[dstIdx])) = dst_f24->z;
}

// I8 stores without layout toggle PLN3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pln3(signed char *dstPtr, uint dstIdx, uint increment, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack_i8(dst_f24->x.x);
    dst.x.y = rpp_hip_pack_i8(dst_f24->x.y);
    dst.y.x = rpp_hip_pack_i8(dst_f24->y.x);
    dst.y.y = rpp_hip_pack_i8(dst_f24->y.y);
    dst.z.x = rpp_hip_pack_i8(dst_f24->z.x);
    dst.z.y = rpp_hip_pack_i8(dst_f24->z.y);

    *((uint2 *)(&dstPtr[dstIdx])) = dst.x;
    dstIdx += increment;
    *((uint2 *)(&dstPtr[dstIdx])) = dst.y;
    dstIdx += increment;
    *((uint2 *)(&dstPtr[dstIdx])) = dst.z;
}

// F16 stores without layout toggle PLN3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pln3(half *dstPtr, uint dstIdx, uint increment, d_float24 *dst_f24)
{
    d_half24 dst_h24;

    dst_h24.x.x.x = __float22half2_rn(make_float2(dst_f24->x.x.x, dst_f24->x.x.y));
    dst_h24.x.x.y = __float22half2_rn(make_float2(dst_f24->x.x.z, dst_f24->x.x.w));
    dst_h24.x.y.x = __float22half2_rn(make_float2(dst_f24->x.y.x, dst_f24->x.y.y));
    dst_h24.x.y.y = __float22half2_rn(make_float2(dst_f24->x.y.z, dst_f24->x.y.w));

    dst_h24.y.x.x = __float22half2_rn(make_float2(dst_f24->y.x.x, dst_f24->y.x.y));
    dst_h24.y.x.y = __float22half2_rn(make_float2(dst_f24->y.x.z, dst_f24->y.x.w));
    dst_h24.y.y.x = __float22half2_rn(make_float2(dst_f24->y.y.x, dst_f24->y.y.y));
    dst_h24.y.y.y = __float22half2_rn(make_float2(dst_f24->y.y.z, dst_f24->y.y.w));

    dst_h24.z.x.x = __float22half2_rn(make_float2(dst_f24->z.x.x, dst_f24->z.x.y));
    dst_h24.z.x.y = __float22half2_rn(make_float2(dst_f24->z.x.z, dst_f24->z.x.w));
    dst_h24.z.y.x = __float22half2_rn(make_float2(dst_f24->z.y.x, dst_f24->z.y.y));
    dst_h24.z.y.y = __float22half2_rn(make_float2(dst_f24->z.y.z, dst_f24->z.y.w));

    *((d_half8 *)(&dstPtr[dstIdx])) = dst_h24.x;
    dstIdx += increment;
    *((d_half8 *)(&dstPtr[dstIdx])) = dst_h24.y;
    dstIdx += increment;
    *((d_half8 *)(&dstPtr[dstIdx])) = dst_h24.z;
}

// WITH LAYOUT TOGGLE

// U8 stores with layout toggle PLN3 to PKD3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pkd3(uchar *dstPtr, uint dstIdx, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack(make_float4(dst_f24->x.x.x, dst_f24->y.x.x, dst_f24->z.x.x, dst_f24->x.x.y));
    dst.x.y = rpp_hip_pack(make_float4(dst_f24->y.x.y, dst_f24->z.x.y, dst_f24->x.x.z, dst_f24->y.x.z));
    dst.y.x = rpp_hip_pack(make_float4(dst_f24->z.x.z, dst_f24->x.x.w, dst_f24->y.x.w, dst_f24->z.x.w));
    dst.y.y = rpp_hip_pack(make_float4(dst_f24->x.y.x, dst_f24->y.y.x, dst_f24->z.y.x, dst_f24->x.y.y));
    dst.z.x = rpp_hip_pack(make_float4(dst_f24->y.y.y, dst_f24->z.y.y, dst_f24->x.y.z, dst_f24->y.y.z));
    dst.z.y = rpp_hip_pack(make_float4(dst_f24->z.y.z, dst_f24->x.y.w, dst_f24->y.y.w, dst_f24->z.y.w));

    *((d_uint6 *)(&dstPtr[dstIdx])) = dst;
}

// F32 stores with layout toggle PLN3 to PKD3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pkd3(float *dstPtr, uint dstIdx, d_float24 *dst_f24)
{
    d_float24 *dstPtr_f24;
    dstPtr_f24 = (d_float24 *)&dstPtr[dstIdx];

    dstPtr_f24->x.x.x = dst_f24->x.x.x;
    dstPtr_f24->x.x.y = dst_f24->y.x.x;
    dstPtr_f24->x.x.z = dst_f24->z.x.x;

    dstPtr_f24->x.x.w = dst_f24->x.x.y;
    dstPtr_f24->x.y.x = dst_f24->y.x.y;
    dstPtr_f24->x.y.y = dst_f24->z.x.y;

    dstPtr_f24->x.y.z = dst_f24->x.x.z;
    dstPtr_f24->x.y.w = dst_f24->y.x.z;
    dstPtr_f24->y.x.x = dst_f24->z.x.z;

    dstPtr_f24->y.x.y = dst_f24->x.x.w;
    dstPtr_f24->y.x.z = dst_f24->y.x.w;
    dstPtr_f24->y.x.w = dst_f24->z.x.w;

    dstPtr_f24->y.y.x = dst_f24->x.y.x;
    dstPtr_f24->y.y.y = dst_f24->y.y.x;
    dstPtr_f24->y.y.z = dst_f24->z.y.x;

    dstPtr_f24->y.y.w = dst_f24->x.y.y;
    dstPtr_f24->z.x.x = dst_f24->y.y.y;
    dstPtr_f24->z.x.y = dst_f24->z.y.y;

    dstPtr_f24->z.x.z = dst_f24->x.y.z;
    dstPtr_f24->z.x.w = dst_f24->y.y.z;
    dstPtr_f24->z.y.x = dst_f24->z.y.z;

    dstPtr_f24->z.y.y = dst_f24->x.y.w;
    dstPtr_f24->z.y.z = dst_f24->y.y.w;
    dstPtr_f24->z.y.w = dst_f24->z.y.w;
}

// I8 stores with layout toggle PLN3 to PKD3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pkd3(signed char *dstPtr, uint dstIdx, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = rpp_hip_pack_i8(make_float4(dst_f24->x.x.x, dst_f24->y.x.x, dst_f24->z.x.x, dst_f24->x.x.y));
    dst.x.y = rpp_hip_pack_i8(make_float4(dst_f24->y.x.y, dst_f24->z.x.y, dst_f24->x.x.z, dst_f24->y.x.z));
    dst.y.x = rpp_hip_pack_i8(make_float4(dst_f24->z.x.z, dst_f24->x.x.w, dst_f24->y.x.w, dst_f24->z.x.w));
    dst.y.y = rpp_hip_pack_i8(make_float4(dst_f24->x.y.x, dst_f24->y.y.x, dst_f24->z.y.x, dst_f24->x.y.y));
    dst.z.x = rpp_hip_pack_i8(make_float4(dst_f24->y.y.y, dst_f24->z.y.y, dst_f24->x.y.z, dst_f24->y.y.z));
    dst.z.y = rpp_hip_pack_i8(make_float4(dst_f24->z.y.z, dst_f24->x.y.w, dst_f24->y.y.w, dst_f24->z.y.w));

    *((d_uint6 *)(&dstPtr[dstIdx])) = dst;
}

// F16 stores with layout toggle PLN3 to PKD3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pkd3(half *dstPtr, uint dstIdx, d_float24 *dst_f24)
{
    d_half24 dst_h24;

    dst_h24.x.x.x = __float22half2_rn(make_float2(dst_f24->x.x.x, dst_f24->y.x.x));
    dst_h24.x.x.y = __float22half2_rn(make_float2(dst_f24->z.x.x, dst_f24->x.x.y));
    dst_h24.x.y.x = __float22half2_rn(make_float2(dst_f24->y.x.y, dst_f24->z.x.y));
    dst_h24.x.y.y = __float22half2_rn(make_float2(dst_f24->x.x.z, dst_f24->y.x.z));

    dst_h24.y.x.x = __float22half2_rn(make_float2(dst_f24->z.x.z, dst_f24->x.x.w));
    dst_h24.y.x.y = __float22half2_rn(make_float2(dst_f24->y.x.w, dst_f24->z.x.w));
    dst_h24.y.y.x = __float22half2_rn(make_float2(dst_f24->x.y.x, dst_f24->y.y.x));
    dst_h24.y.y.y = __float22half2_rn(make_float2(dst_f24->z.y.x, dst_f24->x.y.y));

    dst_h24.z.x.x = __float22half2_rn(make_float2(dst_f24->y.y.y, dst_f24->z.y.y));
    dst_h24.z.x.y = __float22half2_rn(make_float2(dst_f24->x.y.z, dst_f24->y.y.z));
    dst_h24.z.y.x = __float22half2_rn(make_float2(dst_f24->z.y.z, dst_f24->x.y.w));
    dst_h24.z.y.y = __float22half2_rn(make_float2(dst_f24->y.y.w, dst_f24->z.y.w));

    *((d_half24 *)(&dstPtr[dstIdx])) = dst_h24;
}

// -------------------- Set 6 - LDS Loads --------------------

// WITHOUT LAYOUT TOGGLE

// U8 lds loads without layout toggle (8 U8 pixels)

__device__ __forceinline__ void rpp_hip_lds_load8(uchar *srcPtr, int srcIdx, uchar *src_lds)
{
    *(uint2 *)src_lds = *(uint2 *)&srcPtr[srcIdx];
}

// F32 lds loads without layout toggle (8 F32 pixels)

__device__ __forceinline__ void rpp_hip_lds_load8(float *srcPtr, int srcIdx, uchar *src_lds)
{
    d_float8 *srcPtr_f8;
    srcPtr_f8 = (d_float8 *)&srcPtr[srcIdx];

    d_float8 src_f8;
    src_f8.x = rpp_hip_pixel_check(srcPtr_f8->x * (float4) 255.0);
    src_f8.y = rpp_hip_pixel_check(srcPtr_f8->y * (float4) 255.0);

    uint2 *srcPtr_lds;
    srcPtr_lds = (uint2 *)src_lds;
    srcPtr_lds->x = rpp_hip_pack(src_f8.x);
    srcPtr_lds->y = rpp_hip_pack(src_f8.y);
}

// I8 lds loads without layout toggle (8 I8 pixels)

__device__ __forceinline__ void rpp_hip_lds_load8(signed char *srcPtr, int srcIdx, uchar *src_lds)
{
    rpp_hip_convert8_i8_to_u8(&srcPtr[srcIdx], src_lds);
}

// F16 lds loads without layout toggle (8 F16 pixels)

__device__ __forceinline__ void rpp_hip_lds_load8(half *srcPtr, int srcIdx, uchar *src_lds)
{
    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
    rpp_hip_lds_load8((float *)&src_f8, 0, src_lds);
}

// WITH LAYOUT TOGGLE

// U8 lds loads with layout toggle PKD3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_lds_load24_pkd3_to_pln3(uchar *srcPtr, int srcIdx, uchar **src_lds)
{
    d_uchar24 *srcPtr_uchar24;
    srcPtr_uchar24 = (d_uchar24 *)&srcPtr[srcIdx];

    d_uchar8 *src_lds_c1, *src_lds_c2, *src_lds_c3;
    src_lds_c1 = (d_uchar8 *)src_lds[0];
    src_lds_c2 = (d_uchar8 *)src_lds[1];
    src_lds_c3 = (d_uchar8 *)src_lds[2];

    src_lds_c1->x.x = srcPtr_uchar24->x.x.x;
    src_lds_c1->x.y = srcPtr_uchar24->x.x.w;
    src_lds_c1->x.z = srcPtr_uchar24->x.y.z;
    src_lds_c1->x.w = srcPtr_uchar24->y.x.y;
    src_lds_c1->y.x = srcPtr_uchar24->y.y.x;
    src_lds_c1->y.y = srcPtr_uchar24->y.y.w;
    src_lds_c1->y.z = srcPtr_uchar24->z.x.z;
    src_lds_c1->y.w = srcPtr_uchar24->z.y.y;

    src_lds_c2->x.x = srcPtr_uchar24->x.x.y;
    src_lds_c2->x.y = srcPtr_uchar24->x.y.x;
    src_lds_c2->x.z = srcPtr_uchar24->x.y.w;
    src_lds_c2->x.w = srcPtr_uchar24->y.x.z;
    src_lds_c2->y.x = srcPtr_uchar24->y.y.y;
    src_lds_c2->y.y = srcPtr_uchar24->z.x.x;
    src_lds_c2->y.z = srcPtr_uchar24->z.x.w;
    src_lds_c2->y.w = srcPtr_uchar24->z.y.z;

    src_lds_c3->x.x = srcPtr_uchar24->x.x.z;
    src_lds_c3->x.y = srcPtr_uchar24->x.y.y;
    src_lds_c3->x.z = srcPtr_uchar24->y.x.x;
    src_lds_c3->x.w = srcPtr_uchar24->y.x.w;
    src_lds_c3->y.x = srcPtr_uchar24->y.y.z;
    src_lds_c3->y.y = srcPtr_uchar24->z.x.y;
    src_lds_c3->y.z = srcPtr_uchar24->z.y.x;
    src_lds_c3->y.w = srcPtr_uchar24->z.y.w;
}

// F32 lds loads with layout toggle PKD3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_lds_load24_pkd3_to_pln3(float *srcPtr, int srcIdx, uchar **src_lds)
{
    d_float24 *srcPtr_f24;
    srcPtr_f24 = (d_float24 *)&srcPtr[srcIdx];

    d_uint6 src_uchar24;
    src_uchar24.x.x = rpp_hip_pack(rpp_hip_pixel_check(srcPtr_f24->x.x * (float4) 255.0));
    src_uchar24.x.y = rpp_hip_pack(rpp_hip_pixel_check(srcPtr_f24->x.y * (float4) 255.0));
    src_uchar24.y.x = rpp_hip_pack(rpp_hip_pixel_check(srcPtr_f24->y.x * (float4) 255.0));
    src_uchar24.y.y = rpp_hip_pack(rpp_hip_pixel_check(srcPtr_f24->y.y * (float4) 255.0));
    src_uchar24.z.x = rpp_hip_pack(rpp_hip_pixel_check(srcPtr_f24->z.x * (float4) 255.0));
    src_uchar24.z.y = rpp_hip_pack(rpp_hip_pixel_check(srcPtr_f24->z.y * (float4) 255.0));

    d_uchar8 *src_lds_c1, *src_lds_c2, *src_lds_c3;
    src_lds_c1 = (d_uchar8 *)src_lds[0];
    src_lds_c2 = (d_uchar8 *)src_lds[1];
    src_lds_c3 = (d_uchar8 *)src_lds[2];

    d_uchar24 *srcPtr_uchar24;
    srcPtr_uchar24 = (d_uchar24 *)&src_uchar24;

    src_lds_c1->x.x = srcPtr_uchar24->x.x.x;
    src_lds_c1->x.y = srcPtr_uchar24->x.x.w;
    src_lds_c1->x.z = srcPtr_uchar24->x.y.z;
    src_lds_c1->x.w = srcPtr_uchar24->y.x.y;
    src_lds_c1->y.x = srcPtr_uchar24->y.y.x;
    src_lds_c1->y.y = srcPtr_uchar24->y.y.w;
    src_lds_c1->y.z = srcPtr_uchar24->z.x.z;
    src_lds_c1->y.w = srcPtr_uchar24->z.y.y;

    src_lds_c2->x.x = srcPtr_uchar24->x.x.y;
    src_lds_c2->x.y = srcPtr_uchar24->x.y.x;
    src_lds_c2->x.z = srcPtr_uchar24->x.y.w;
    src_lds_c2->x.w = srcPtr_uchar24->y.x.z;
    src_lds_c2->y.x = srcPtr_uchar24->y.y.y;
    src_lds_c2->y.y = srcPtr_uchar24->z.x.x;
    src_lds_c2->y.z = srcPtr_uchar24->z.x.w;
    src_lds_c2->y.w = srcPtr_uchar24->z.y.z;

    src_lds_c3->x.x = srcPtr_uchar24->x.x.z;
    src_lds_c3->x.y = srcPtr_uchar24->x.y.y;
    src_lds_c3->x.z = srcPtr_uchar24->y.x.x;
    src_lds_c3->x.w = srcPtr_uchar24->y.x.w;
    src_lds_c3->y.x = srcPtr_uchar24->y.y.z;
    src_lds_c3->y.y = srcPtr_uchar24->z.x.y;
    src_lds_c3->y.z = srcPtr_uchar24->z.y.x;
    src_lds_c3->y.w = srcPtr_uchar24->z.y.w;
}

// F16 lds loads with layout toggle PKD3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_lds_load24_pkd3_to_pln3(half *srcPtr, int srcIdx, uchar **src_lds)
{
    d_half24 *srcPtr_h24;
    srcPtr_h24 = (d_half24 *)&srcPtr[srcIdx];

    d_float24 src_f24;
    float2 src1_f2, src2_f2;

    src1_f2 = __half22float2(srcPtr_h24->x.x.x);
    src2_f2 = __half22float2(srcPtr_h24->x.x.y);
    src_f24.x.x = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);
    src1_f2 = __half22float2(srcPtr_h24->x.y.x);
    src2_f2 = __half22float2(srcPtr_h24->x.y.y);
    src_f24.x.y = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);
    src1_f2 = __half22float2(srcPtr_h24->y.x.x);
    src2_f2 = __half22float2(srcPtr_h24->y.x.y);
    src_f24.y.x = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);
    src1_f2 = __half22float2(srcPtr_h24->y.y.x);
    src2_f2 = __half22float2(srcPtr_h24->y.y.y);
    src_f24.y.y = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);
    src1_f2 = __half22float2(srcPtr_h24->z.x.x);
    src2_f2 = __half22float2(srcPtr_h24->z.x.y);
    src_f24.z.x = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);
    src1_f2 = __half22float2(srcPtr_h24->z.y.x);
    src2_f2 = __half22float2(srcPtr_h24->z.y.y);
    src_f24.z.y = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);

    rpp_hip_lds_load24_pkd3_to_pln3((float *)&src_f24, 0, src_lds);
}

// I8 lds loads with layout toggle PKD3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_lds_load24_pkd3_to_pln3(signed char *srcPtr, int srcIdx, uchar **src_lds)
{
    d_uchar24 src_uchar24;
    rpp_hip_convert24_i8_to_u8(&srcPtr[srcIdx], (uchar *)&src_uchar24);
    rpp_hip_lds_load24_pkd3_to_pln3((uchar *)&src_uchar24, 0, src_lds);
}

#endif //RPP_HIP_COMMON_H