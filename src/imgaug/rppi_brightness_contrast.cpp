#include <rppdefs.h>
#include <rppi_brightness_illumination_functions.h>


//--------------------------- Device Calls -------------------------------------
template <typename T>
RppStatus hip_brightness_contrast (T* inputPtr, RppiSize imgDim, T* outputPtr,
                                Rpp32f alpha, Rpp32f beta, RppiChnFormat chnFormat);

RppStatus cl_brightness_contrast( Rpp8u* pSrc, unsigned int height, unsigned int width,
                                  Rpp8u* pDst, Rpp32f alpha, Rpp32f beta);
//------------------------------------------------------------------------------

RppStatus
rppi_brighten_1C8U_pln(Rpp8u *pSrc, RppiSize size, Rpp8u *pDst, Rpp32f alpha = 1, Rpp32f beta = 0)
{

#ifdef HIP_COMPILE
    hip_brightness_contrast<Rpp8u>(pSrc, size, pDst, alpha, beta, RPPI_CHN_PLANAR );
#elif defined (OCL_COMPILE)
    cl_brightness_contrast( pSrc, size.height, size.width, pDst, alpha, beta);
#endif

    return RPP_SUCCESS;
}


//------------------------------ Host Calls ------------------------------------
template <typename T>
RppStatus host_brightness_contrast(T* pSrc, RppiSize size, T* pDst,
                                Rpp32f alpha, Rpp32f beta, RppiChnFormat chnFormat);


//------------------------------------------------------------------------------

RppStatus
rppi_brighten_1C8U_pln_host(Rpp8u *pSrc, RppiSize size, Rpp8u *pDst, Rpp32f alpha = 1, Rpp32f beta = 0)
{

    host_brightness_contrast<Rpp8u>(pSrc, size, pDst, alpha, beta, RPPI_CHN_PLANAR );

#ifdef CV_COMPILE
    printf("OpenCV base not implemented");
    return RPP_ERROR;
#endif

    return RPP_SUCCESS;

}
