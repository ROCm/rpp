#ifndef RPP_CPU_COMMON_ARITHMETIC_H
#define RPP_CPU_COMMON_ARITHMETIC_H

inline void compute_sum_16_host(__m256i *p, __m256i *pSum)
{
    pSum[0] = _mm256_add_epi32(_mm256_add_epi32(p[0], p[1]), pSum[0]); //add 16 values to 8
}

inline void compute_sum_48_host(__m256i *p, __m256i *pSumR, __m256i *pSumG, __m256i *pSumB)
{
    pSumR[0] = _mm256_add_epi32(_mm256_add_epi32(p[0], p[1]), pSumR[0]); //add 16R values and bring it down to 8
    pSumG[0] = _mm256_add_epi32(_mm256_add_epi32(p[2], p[3]), pSumG[0]); //add 16G values and bring it down to 8
    pSumB[0] = _mm256_add_epi32(_mm256_add_epi32(p[4], p[5]), pSumB[0]); //add 16B values and bring it down to 8
}

inline void compute_sum_8_host(__m256d *p, __m256d *pSum)
{
    pSum[0] = _mm256_add_pd(_mm256_add_pd(p[0], p[1]), pSum[0]); //add 8 values and bring it down to 4
}

inline void compute_sum_24_host(__m256d *p, __m256d *pSumR, __m256d *pSumG, __m256d *pSumB)
{
    pSumR[0] = _mm256_add_pd(_mm256_add_pd(p[0], p[1]), pSumR[0]); //add 8R values and bring it down to 4
    pSumG[0] = _mm256_add_pd(_mm256_add_pd(p[2], p[3]), pSumG[0]); //add 8G values and bring it down to 4
    pSumB[0] = _mm256_add_pd(_mm256_add_pd(p[4], p[5]), pSumB[0]); //add 8B values and bring it down to 4
}

#endif