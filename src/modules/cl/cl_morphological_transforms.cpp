#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"
#define UNROLL 1

// This function takes a positive integer and rounds it up to
// the nearest multiple of another provided integer
unsigned int roundUp(unsigned int value, unsigned int multiple) {
	
  // Determine how far past the nearest multiple the value is
  unsigned int remainder = value % multiple;
  
  // Add the difference to make the value a multiple
  if(remainder != 0) {
          value += (multiple-remainder);
  }
  
  return value;
}

// /********************** Dilate ************************/
// RppStatus
// dilate_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
// {
//     int ctr=0;
//     cl_kernel theKernel;
//     cl_program theProgram;
    
//     if(chnFormat == RPPI_CHN_PACKED)
//     {
//         CreateProgramFromBinary(handle,"dilate.cl","dilate.cl.bin","dilate_pkd",theProgram,theKernel);
//         clRetainKernel(theKernel);
//     }
//     else
//     {
//         CreateProgramFromBinary(handle,"dilate.cl","dilate.cl.bin","dilate_local",theProgram,theKernel);
//         clRetainKernel(theKernel);
//     }
    
//     //---- Args Setter
//     clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
//     clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
//     clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
//     clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
//     clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
//     clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
        
//     size_t gDim3[3];
//     gDim3[0] = srcSize.width;
//     gDim3[1] = srcSize.height;
//     gDim3[2] = channel;

//     size_t lDim3[3];
//     lDim3[0] = 16;
//     lDim3[1] = 16;
//     lDim3[2] = 1;

//     cl_kernel_implementer (handle, gDim3, lDim3, theProgram, theKernel);
    
//     return RPP_SUCCESS;    
// }


/********************** Dilate ************************/
RppStatus
dilate_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};

        #if UNROLL
        handle.AddKernel("", "", "dilate.cl", "dilate_unroll", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                kernelSize
                                                                );
        #else
        std::cerr << "coming in pkd" << std::endl;
        handle.AddKernel("", "", "dilate.cl", "dilate_pkd", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                kernelSize
                                                                );
        #endif
    }
    else
    {
        std::vector<size_t> vld{16, 16, 1};
        std::vector<size_t> vgd{srcSize.width / 16 , srcSize.height ,channel};
        uint p1_stride =srcSize.width, p0_stride = srcSize.width, p0_offset = 0, p1_offset =0 ;
        #if UNROLL
        
        //void Dilate_kernel(uint width, uint height, __global uchar * p0_buf, uint p0_stride, uint p0_offset, __global uchar * p1_buf, uint p1_stride, uint p1_offset)
        //handle.AddKernel("", "", "dummy.cl", "Dilate_kernel", vld, vgd, "")(srcSize.width, srcSize.height, dstPtr,  p0_stride , p0_offset , srcPtr, p1_stride , p1_offset);
        handle.AddKernel("", "", "dilate.cl", "dilate_optimized_trail", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                kernelSize
                                                                );
        #else
        handle.AddKernel("", "", "dilate.cl", "dilate_pln", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                kernelSize
                                                                );
        #endif
    }
    return RPP_SUCCESS;    
}


RppStatus
dilate_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "dilate.cl", "dilate_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}



/********************** Erode ************************/
RppStatus
erode_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "erode.cl", "erode_pkd", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                kernelSize
                                                                );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "erode.cl", "erode_pln", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                kernelSize
                                                                );
    }
    return RPP_SUCCESS;    
}


RppStatus
erode_cl_batch (   cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "erode.cl", "erode_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}
