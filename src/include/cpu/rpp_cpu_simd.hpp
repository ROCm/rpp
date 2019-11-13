//
// Created by svcbuild on 11/12/19.
//

#ifndef AMD_RPP_RPP_CPU_SIMD_HPP
#define AMD_RPP_RPP_CPU_SIMD_HPP
#if ENABLE_SIMD_INTRINSICS
#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif
#define M256I(m256i_register) (*((_m256i_union*)&m256i_register))
typedef union {
    char               m256i_i8[32];
    short              m256i_i16[16];
    int                m256i_i32[8];
    long long          m256i_i64[4];
    __m128i            m256i_i128[2];
}_m256i_union;

#endif
#endif //AMD_RPP_RPP_CPU_SIMD_HPP
