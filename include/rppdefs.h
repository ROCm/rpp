/*
   MulticoreWare Inc.
*/

#ifndef RPPIDEFS_H
#define RPPIDEFS_H
#ifdef __cplusplus
extern "C" {
#endif

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

typedef void*              RppPtr_t;

typedef void*              RppHandle_t;


typedef enum
{
    RPP_SUCCESS = 0,
    RPP_ERROR   = 1,
} RppStatus;

typedef enum
{
    RPPI_HORIZONTAL_AXIS,
    RPPI_VERTICAL_AXIS,
    RPPI_BOTH_AXIS
} RppiAxis;

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

typedef struct {
    unsigned int width;
    unsigned int height;
    } RppiSize;

typedef struct
   {
       int x;
       int y;
       int width;
       int height;
   } RppiRect;

typedef enum{
    GAUSS3,
    GAUSS5,
    GAUSS3x1,
    GAUSS1x3,
    AVG3 = 10,
    AVG5
} RppiBlur;

typedef enum{
    ZEROPAD,
    NOPAD
} RppiPad;

typedef struct {
       Rpp32f rho;
       Rpp32f theta;
   } RppPointPolar;


#define RPP_MIN_8U      ( 0 )
#define RPP_MAX_8U      ( 255 )
#define RPP_MIN_16U     ( 0 )
#define RPP_MAX_16U     ( 65535 )

#ifdef __cplusplus
}
#endif
#endif /* RPPIDEFS_H */
