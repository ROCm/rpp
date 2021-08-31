/*To be used by the library alone*/
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

typedef struct d_uint6
{
    uint2 x;
    uint2 y;
    uint2 z;
} d_uint6;

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

// Host functions

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

// Device functions

// Packing

__device__ __forceinline__ uint rpp_hip_pack(float4 src)
{
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

__device__ __forceinline__ int rpp_hip_pack_to_i8(float4 src)
{
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

// Un-Packing

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

__device__ __forceinline__ float rpp_hip_unpack0(int src)
{
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack1(int src)
{
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack2(int src)
{
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack3(int src)
{
    return (float)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float4 rpp_hip_unpack_from_i8(int src)
{
    return make_float4(rpp_hip_unpack0(src), rpp_hip_unpack1(src), rpp_hip_unpack2(src), rpp_hip_unpack3(src));
}

// U8 loads and stores without layout toggle (8 U8 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(uchar *srcPtr, uint srcIdx, d_float8 *src_f8)
{
    uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
    src_f8->x = rpp_hip_unpack(src.x);
    src_f8->y = rpp_hip_unpack(src.y);
}

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(uchar *dstPtr, uint dstIdx, d_float8 *dst_f8)
{
    uint2 dst;
    dst.x = rpp_hip_pack(dst_f8->x);
    dst.y = rpp_hip_pack(dst_f8->y);
    *((uint2 *)(&dstPtr[dstIdx])) = dst;
}

// F32 loads and stores without layout toggle (8 F32 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(float *srcPtr, uint srcIdx, d_float8 *src_f8)
{
    src_f8->x = *((float4 *)(&srcPtr[srcIdx]));
    src_f8->y = *((float4 *)(&srcPtr[srcIdx + 4]));
}

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(float *dstPtr, uint dstIdx, d_float8 *dst_f8)
{
    *((float4 *)(&dstPtr[dstIdx])) = dst_f8->x;
    *((float4 *)(&dstPtr[dstIdx + 4])) = dst_f8->y;
}

// I8 loads and stores without layout toggle (8 I8 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(signed char *srcPtr, uint srcIdx, d_float8 *src_f8)
{
    uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
    src_f8->x = rpp_hip_unpack(src.x);
    src_f8->y = rpp_hip_unpack(src.y);
}

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(signed char *dstPtr, uint dstIdx, d_float8 *dst_f8)
{
    int2 dst;
    dst.x = (int) rpp_hip_pack(dst_f8->x);
    dst.y = (int) rpp_hip_pack(dst_f8->y);
    *((int2 *)(&dstPtr[dstIdx])) = dst;
}

// F16 loads and stores without layout toggle (8 F16 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(half *srcPtr, uint srcIdx, d_float8 *src_f8)
{
    d_half8 src_h8;
    src_h8 = *((d_half8 *)(&srcPtr[srcIdx]));

    *(float2 *)&(src_f8->x) = __half22float2(src_h8.x.x);
    *((float2 *)&(src_f8->x) + 1) = __half22float2(src_h8.x.y);
    *(float2 *)&(src_f8->y) = __half22float2(src_h8.y.x);
    *((float2 *)&(src_f8->y) + 1) = __half22float2(src_h8.y.y);
}

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(half *dstPtr, uint dstIdx, d_float8 *dst_f8)
{
    d_half8 dst_h8;

    dst_h8.x.x = __float22half2_rn(*(float2 *)&(dst_f8->x));
    dst_h8.x.y = __float22half2_rn(*((float2 *)&(dst_f8->x) + 1));
    dst_h8.y.x = __float22half2_rn(*(float2 *)&(dst_f8->y));
    dst_h8.y.y = __float22half2_rn(*((float2 *)&(dst_f8->y) + 1));

    *((d_half8 *)(&dstPtr[dstIdx])) = dst_h8;
}

// U8 loads with layout toggle PKD3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(uchar *srcPtr, uint srcIdx, d_float24 *src_f24)
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

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(float *srcPtr, uint srcIdx, d_float24 *src_f24)
{
    src_f24->x.x.x = srcPtr[srcIdx];
    src_f24->x.x.y = srcPtr[srcIdx + 3];
    src_f24->x.x.z = srcPtr[srcIdx + 6];
    src_f24->x.x.w = srcPtr[srcIdx + 9];
    src_f24->x.y.x = srcPtr[srcIdx + 12];
    src_f24->x.y.y = srcPtr[srcIdx + 15];
    src_f24->x.y.z = srcPtr[srcIdx + 18];
    src_f24->x.y.w = srcPtr[srcIdx + 21];

    src_f24->y.x.x = srcPtr[srcIdx + 1];
    src_f24->y.x.y = srcPtr[srcIdx + 4];
    src_f24->y.x.z = srcPtr[srcIdx + 7];
    src_f24->y.x.w = srcPtr[srcIdx + 10];
    src_f24->y.y.x = srcPtr[srcIdx + 13];
    src_f24->y.y.y = srcPtr[srcIdx + 16];
    src_f24->y.y.z = srcPtr[srcIdx + 19];
    src_f24->y.y.w = srcPtr[srcIdx + 22];

    src_f24->z.x.x = srcPtr[srcIdx + 2];
    src_f24->z.x.y = srcPtr[srcIdx + 5];
    src_f24->z.x.z = srcPtr[srcIdx + 8];
    src_f24->z.x.w = srcPtr[srcIdx + 11];
    src_f24->z.y.x = srcPtr[srcIdx + 14];
    src_f24->z.y.y = srcPtr[srcIdx + 17];
    src_f24->z.y.z = srcPtr[srcIdx + 20];
    src_f24->z.y.w = srcPtr[srcIdx + 23];
}

// I8 loads with layout toggle PKD3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(signed char *srcPtr, uint srcIdx, d_float24 *src_f24)
{
    d_uint6 src = *((d_uint6 *)(&srcPtr[srcIdx]));

    src_f24->x.x = make_float4(rpp_hip_unpack0(src.x.x), rpp_hip_unpack3(src.x.x), rpp_hip_unpack2(src.x.y), rpp_hip_unpack1(src.y.x));
    src_f24->x.y = make_float4(rpp_hip_unpack0(src.y.y), rpp_hip_unpack3(src.y.y), rpp_hip_unpack2(src.z.x), rpp_hip_unpack1(src.z.y));

    src_f24->y.x = make_float4(rpp_hip_unpack1(src.x.x), rpp_hip_unpack0(src.x.y), rpp_hip_unpack3(src.x.y), rpp_hip_unpack2(src.y.x));
    src_f24->y.y = make_float4(rpp_hip_unpack1(src.y.y), rpp_hip_unpack0(src.z.x), rpp_hip_unpack3(src.z.x), rpp_hip_unpack2(src.z.y));

    src_f24->z.x = make_float4(rpp_hip_unpack2(src.x.x), rpp_hip_unpack1(src.x.y), rpp_hip_unpack0(src.y.x), rpp_hip_unpack3(src.y.x));
    src_f24->z.y = make_float4(rpp_hip_unpack2(src.y.y), rpp_hip_unpack1(src.z.x), rpp_hip_unpack0(src.z.y), rpp_hip_unpack3(src.z.y));
}

// F16 loads with layout toggle PKD3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(half *srcPtr, uint srcIdx, d_float24 *src_f24)
{
    src_f24->x.x.x = __half2float(srcPtr[srcIdx]);
    src_f24->x.x.y = __half2float(srcPtr[srcIdx + 3]);
    src_f24->x.x.z = __half2float(srcPtr[srcIdx + 6]);
    src_f24->x.x.w = __half2float(srcPtr[srcIdx + 9]);
    src_f24->x.y.x = __half2float(srcPtr[srcIdx + 12]);
    src_f24->x.y.y = __half2float(srcPtr[srcIdx + 15]);
    src_f24->x.y.z = __half2float(srcPtr[srcIdx + 18]);
    src_f24->x.y.w = __half2float(srcPtr[srcIdx + 21]);

    src_f24->y.x.x = __half2float(srcPtr[srcIdx + 1]);
    src_f24->y.x.y = __half2float(srcPtr[srcIdx + 4]);
    src_f24->y.x.z = __half2float(srcPtr[srcIdx + 7]);
    src_f24->y.x.w = __half2float(srcPtr[srcIdx + 10]);
    src_f24->y.y.x = __half2float(srcPtr[srcIdx + 13]);
    src_f24->y.y.y = __half2float(srcPtr[srcIdx + 16]);
    src_f24->y.y.z = __half2float(srcPtr[srcIdx + 19]);
    src_f24->y.y.w = __half2float(srcPtr[srcIdx + 22]);

    src_f24->z.x.x = __half2float(srcPtr[srcIdx + 2]);
    src_f24->z.x.y = __half2float(srcPtr[srcIdx + 5]);
    src_f24->z.x.z = __half2float(srcPtr[srcIdx + 8]);
    src_f24->z.x.w = __half2float(srcPtr[srcIdx + 11]);
    src_f24->z.y.x = __half2float(srcPtr[srcIdx + 14]);
    src_f24->z.y.y = __half2float(srcPtr[srcIdx + 17]);
    src_f24->z.y.z = __half2float(srcPtr[srcIdx + 20]);
    src_f24->z.y.w = __half2float(srcPtr[srcIdx + 23]);
}

// U8 loads with layout toggle PLN3 to PKD3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(uchar *srcPtr, uint srcIdx, uint increment, d_float24 *src_f24)
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

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(float *srcPtr, uint srcIdx, uint increment, d_float24 *src_f24)
{
    float *srcPtrR, *srcPtrG, *srcPtrB;
    srcPtrR = srcPtr + srcIdx;
    srcPtrG = srcPtrR + increment;
    srcPtrB = srcPtrG + increment;

    src_f24->x.x.x = *srcPtrR;
    src_f24->x.x.y = *srcPtrG;
    src_f24->x.x.z = *srcPtrB;

    src_f24->x.x.w = *(srcPtrR + 1);
    src_f24->x.y.x = *(srcPtrG + 1);
    src_f24->x.y.y = *(srcPtrB + 1);

    src_f24->x.y.z = *(srcPtrR + 2);
    src_f24->x.y.w = *(srcPtrG + 2);
    src_f24->y.x.x = *(srcPtrB + 2);

    src_f24->y.x.y = *(srcPtrR + 3);
    src_f24->y.x.z = *(srcPtrG + 3);
    src_f24->y.x.w = *(srcPtrB + 3);

    src_f24->y.y.x = *(srcPtrR + 4);
    src_f24->y.y.y = *(srcPtrG + 4);
    src_f24->y.y.z = *(srcPtrB + 4);

    src_f24->y.y.w = *(srcPtrR + 5);
    src_f24->z.x.x = *(srcPtrG + 5);
    src_f24->z.x.y = *(srcPtrB + 5);

    src_f24->z.x.z = *(srcPtrR + 6);
    src_f24->z.x.w = *(srcPtrG + 6);
    src_f24->z.y.x = *(srcPtrB + 6);

    src_f24->z.y.y = *(srcPtrR + 7);
    src_f24->z.y.z = *(srcPtrG + 7);
    src_f24->z.y.w = *(srcPtrB + 7);
}

// I8 loads with layout toggle PLN3 to PKD3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(signed char *srcPtr, uint srcIdx, uint increment, d_float24 *src_f24)
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

// F16 loads with layout toggle PLN3 to PKD3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(half *srcPtr, uint srcIdx, uint increment, d_float24 *src_f24)
{
    half *srcPtrR, *srcPtrG, *srcPtrB;
    srcPtrR = srcPtr + srcIdx;
    srcPtrG = srcPtrR + increment;
    srcPtrB = srcPtrG + increment;

    src_f24->x.x.x = __half2float(*srcPtrR);
    src_f24->x.x.y = __half2float(*srcPtrG);
    src_f24->x.x.z = __half2float(*srcPtrB);

    src_f24->x.x.w = __half2float(*(srcPtrR + 1));
    src_f24->x.y.x = __half2float(*(srcPtrG + 1));
    src_f24->x.y.y = __half2float(*(srcPtrB + 1));

    src_f24->x.y.z = __half2float(*(srcPtrR + 2));
    src_f24->x.y.w = __half2float(*(srcPtrG + 2));
    src_f24->y.x.x = __half2float(*(srcPtrB + 2));

    src_f24->y.x.y = __half2float(*(srcPtrR + 3));
    src_f24->y.x.z = __half2float(*(srcPtrG + 3));
    src_f24->y.x.w = __half2float(*(srcPtrB + 3));

    src_f24->y.y.x = __half2float(*(srcPtrR + 4));
    src_f24->y.y.y = __half2float(*(srcPtrG + 4));
    src_f24->y.y.z = __half2float(*(srcPtrB + 4));

    src_f24->y.y.w = __half2float(*(srcPtrR + 5));
    src_f24->z.x.x = __half2float(*(srcPtrG + 5));
    src_f24->z.x.y = __half2float(*(srcPtrB + 5));

    src_f24->z.x.z = __half2float(*(srcPtrR + 6));
    src_f24->z.x.w = __half2float(*(srcPtrG + 6));
    src_f24->z.y.x = __half2float(*(srcPtrB + 6));

    src_f24->z.y.y = __half2float(*(srcPtrR + 7));
    src_f24->z.y.z = __half2float(*(srcPtrG + 7));
    src_f24->z.y.w = __half2float(*(srcPtrB + 7));
}

// U8 stores without layout toggle (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_and_store24(uchar *dstPtr, uint dstIdx, d_float24 *dst_f24)
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

// F32 stores without layout toggle (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_and_store24(float *dstPtr, uint dstIdx, d_float24 *dst_f24)
{
    *((d_float24 *)(&dstPtr[dstIdx])) = *dst_f24;
}

// I8 stores without layout toggle (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_and_store24(signed char *dstPtr, uint dstIdx, d_float24 *dst_f24)
{
    d_uint6 dst;

    dst.x.x = (int) rpp_hip_pack(dst_f24->x.x);
    dst.x.y = (int) rpp_hip_pack(dst_f24->x.y);
    dst.y.x = (int) rpp_hip_pack(dst_f24->y.x);
    dst.y.y = (int) rpp_hip_pack(dst_f24->y.y);
    dst.z.x = (int) rpp_hip_pack(dst_f24->z.x);
    dst.z.y = (int) rpp_hip_pack(dst_f24->z.y);

    *((d_uint6 *)(&dstPtr[dstIdx])) = dst;
}

// F16 stores without layout toggle (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_and_store24(half *dstPtr, uint dstIdx, d_float24 *dst_f24)
{
    d_half24 dst_h24;

    dst_h24.x.x.x = __float22half2_rn(*(float2 *)&(dst_f24->x));
    dst_h24.x.x.y = __float22half2_rn(*((float2 *)&(dst_f24->x) + 1));
    dst_h24.x.y.x = __float22half2_rn(*((float2 *)&(dst_f24->x) + 2));
    dst_h24.x.y.y = __float22half2_rn(*((float2 *)&(dst_f24->x) + 3));

    dst_h24.y.x.x = __float22half2_rn(*((float2 *)&(dst_f24->x) + 4));
    dst_h24.y.x.y = __float22half2_rn(*((float2 *)&(dst_f24->x) + 5));
    dst_h24.y.y.x = __float22half2_rn(*((float2 *)&(dst_f24->x) + 6));
    dst_h24.y.y.y = __float22half2_rn(*((float2 *)&(dst_f24->x) + 7));

    dst_h24.z.x.x = __float22half2_rn(*((float2 *)&(dst_f24->x) + 8));
    dst_h24.z.x.y = __float22half2_rn(*((float2 *)&(dst_f24->x) + 9));
    dst_h24.z.y.x = __float22half2_rn(*((float2 *)&(dst_f24->x) + 10));
    dst_h24.z.y.y = __float22half2_rn(*((float2 *)&(dst_f24->x) + 11));

    *((d_half24 *)(&dstPtr[dstIdx])) = dst_h24;
}

// Other

__device__ __forceinline__ float4 rpp_hip_pixel_check(float4 src_f4)
{
    return make_float4(fminf(fmaxf(src_f4.x, 0), 255),
                       fminf(fmaxf(src_f4.y, 0), 255),
                       fminf(fmaxf(src_f4.z, 0), 255),
                       fminf(fmaxf(src_f4.w, 0), 255));
}

#endif //RPP_HIP_COMMON_H