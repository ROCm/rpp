/*To be used by the library alone*/
#ifndef RPP_HIP_COMMON_H
#define RPP_HIP_COMMON_H

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <rppdefs.h>
#include <vector>
#include <half.hpp>
using half_float::half;
typedef half Rpp16f;

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

__device__ __forceinline__ uint rpp_hip_pack(float4 src)
{
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

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

__device__ __forceinline__ int rpp_hip_pack_to_i8(float4 src)
{
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
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

__device__ void rpp_hip_load8_and_unpack_to_float8(uchar *srcPtr, uint srcIdx, float4 *srcX_f4, float4 *srcY_f4)
{
    uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
    *srcX_f4 = rpp_hip_unpack(src.x);
    *srcY_f4 = rpp_hip_unpack(src.y);
}

__device__ void rpp_hip_pack_float8_and_store8(uchar *dstPtr, uint dstIdx, float4 *dstX_f4, float4 *dstY_f4)
{
    uint2 dst;
    dst.x = rpp_hip_pack(*dstX_f4);
    dst.y = rpp_hip_pack(*dstY_f4);
    *((uint2 *)(&dstPtr[dstIdx])) = dst;
}

__device__ void rpp_hip_load8_and_unpack_to_float8(float *srcPtr, uint srcIdx, float4 *srcX_f4, float4 *srcY_f4)
{
    *srcX_f4 = *((float4 *)(&srcPtr[srcIdx]));
    *srcY_f4 = *((float4 *)(&srcPtr[srcIdx + 4]));
}

__device__ void rpp_hip_pack_float8_and_store8(float *dstPtr, uint dstIdx, float4 *dstX_f4, float4 *dstY_f4)
{
    *((float4 *)(&dstPtr[dstIdx])) = *dstX_f4;
    *((float4 *)(&dstPtr[dstIdx + 4])) = *dstY_f4;
}

__device__ void rpp_hip_load8_and_unpack_to_float8(signed char *srcPtr, uint srcIdx, float4 *srcX_f4, float4 *srcY_f4)
{
    int2 src = *((int2 *)(&srcPtr[srcIdx]));
    *srcX_f4 = rpp_hip_unpack_from_i8(src.x);
    *srcY_f4 = rpp_hip_unpack_from_i8(src.y);
}

__device__ void rpp_hip_pack_float8_and_store8(signed char *dstPtr, uint dstIdx, float4 *dstX_f4, float4 *dstY_f4)
{
    int2 dst;
    dst.x = rpp_hip_pack_to_i8(*dstX_f4);
    dst.y = rpp_hip_pack_to_i8(*dstY_f4);
    *((int2 *)(&dstPtr[dstIdx])) = dst;
}

#endif //RPP_HIP_COMMON_H