#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// Compute Inverse matrix (3x3)
__global__ void get_inverse_hip(d_float9 *matTensor, d_float9 *invMatTensor)
{
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    d_float9 *mat = &matTensor[id_z];
    d_float9 *invMat = &invMatTensor[id_z];
    float det = mat->f1[0] * (mat->f1[4] * mat->f1[8] - mat->f1[7] * mat->f1[5]) - mat->f1[1] * (mat->f1[3] * mat->f1[8] - mat->f1[5] * mat->f1[6]) + mat->f1[2] * (mat->f1[3] * mat->f1[7] - mat->f1[4] * mat->f1[6]);
    if(det != 0)
    {
        float invDet = 1 / det;
        invMat->f1[0] = (mat->f1[4] * mat->f1[8] - mat->f1[7] * mat->f1[5]) * invDet;
        invMat->f1[1] = (mat->f1[2] * mat->f1[7] - mat->f1[1] * mat->f1[8]) * invDet;
        invMat->f1[2] = (mat->f1[1] * mat->f1[5] - mat->f1[2] * mat->f1[4]) * invDet;
        invMat->f1[3] = (mat->f1[5] * mat->f1[6] - mat->f1[3] * mat->f1[8]) * invDet;
        invMat->f1[4] = (mat->f1[0] * mat->f1[8] - mat->f1[2] * mat->f1[6]) * invDet;
        invMat->f1[5] = (mat->f1[3] * mat->f1[2] - mat->f1[0] * mat->f1[5]) * invDet;
        invMat->f1[6] = (mat->f1[3] * mat->f1[7] - mat->f1[6] * mat->f1[4]) * invDet;
        invMat->f1[7] = (mat->f1[6] * mat->f1[1] - mat->f1[0] * mat->f1[7]) * invDet;
        invMat->f1[8] = (mat->f1[0] * mat->f1[4] - mat->f1[3] * mat->f1[1]) * invDet;
    }
}

__global__ void compute_remap_tables_hip_tensor(float *rowRemapTable,
                                                float *colRemapTable,
                                                d_float9 *cameraMatrixTensor,
                                                d_float9 *inverseMatrixTensor,
                                                d_float8 *distortionCoeffsTensor,
                                                d_float9 *newCameraMatrixTensor,
                                                uint2 remapTableStridesNH,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int height = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    int width = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    d_float9 cameraMatrix = cameraMatrixTensor[id_z];
    d_float9 newCameraMatrix = newCameraMatrixTensor[id_z];
    d_float9 inverseMatrix = inverseMatrixTensor[id_z];
    d_float8 distortionCoeffs = distortionCoeffsTensor[id_z];

    // Get radial and tangential distortion coefficients
    float rCoeff[6] = { distortionCoeffs.f1[0], distortionCoeffs.f1[1], distortionCoeffs.f1[4], distortionCoeffs.f1[5], distortionCoeffs.f1[6], distortionCoeffs.f1[7] };
    float tCoeff[2] = { distortionCoeffs.f1[2], distortionCoeffs.f1[3] };

    // Get the focal length and principal point of the camera
    float u0 = cameraMatrix.f1[2],  v0 = cameraMatrix.f1[5];
    float fx = cameraMatrix.f1[0],  fy = cameraMatrix.f1[4];

    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y + id_x;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y + id_x;
    float xCamera = id_y * inverseMatrix.f1[1] + inverseMatrix.f1[2] + id_x * inverseMatrix.f1[0];
    float yCamera = id_y * inverseMatrix.f1[4] + inverseMatrix.f1[5] + id_x * inverseMatrix.f1[3];
    float zCamera = id_y * inverseMatrix.f1[7] + inverseMatrix.f1[8] + id_x * inverseMatrix.f1[6];
    float z = 1./zCamera, x = xCamera * z, y = yCamera * z;
    float xSquare = x * x, ySquare = y * y;
    float r2 = xSquare + ySquare, xyMul2 = 2 * x * y;
    float kr = (1 + ((rCoeff[2] * r2 + rCoeff[1]) * r2 + rCoeff[0]) * r2) / (1 + ((rCoeff[5] * r2 + rCoeff[4]) * r2 + rCoeff[3]) *r2);
    float colLoc = fx * (x * kr + tCoeff[0] * xyMul2 + tCoeff[1] * (r2 + 2 * xSquare)) + u0;
    float rowLoc = fy * (y * kr + tCoeff[0] * (r2 + 2 * ySquare) + tCoeff[1] * xyMul2) + v0;
    *colRemapTableTemp = colLoc;
    *rowRemapTableTemp = rowLoc;
}

// -------------------- Set 3 - Kernel Executors --------------------

RppStatus hip_exec_lens_correction_tensor(RpptDescPtr dstDescPtr,
                                          Rpp32f *rowRemapTable,
                                          Rpp32f *colRemapTable,
                                          RpptDescPtr remapTableDescPtr,
                                          Rpp32f *cameraMatrix,
                                          Rpp32f *distanceCoeffs,
                                          Rpp32f *newCameraMatrix,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = dstDescPtr->w;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = dstDescPtr->n;

    float *inverseMatrix = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    hipLaunchKernelGGL(get_inverse_hip,
                       dim3(1, 1, ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(1, 1, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       reinterpret_cast<d_float9 *>(cameraMatrix),
                       reinterpret_cast<d_float9 *>(inverseMatrix));
    hipStreamSynchronize(handle.GetStream());
    hipLaunchKernelGGL(compute_remap_tables_hip_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       rowRemapTable,
                       colRemapTable,
                       reinterpret_cast<d_float9 *>(cameraMatrix),
                       reinterpret_cast<d_float9 *>(inverseMatrix),
                       reinterpret_cast<d_float8 *>(distanceCoeffs),
                       reinterpret_cast<d_float9 *>(newCameraMatrix),
                       make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                       roiTensorPtrSrc);
    return RPP_SUCCESS;
}