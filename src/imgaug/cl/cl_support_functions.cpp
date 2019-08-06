#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

RppStatus
data_object_copy_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    clEnqueueCopyBuffer(theQueue, srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
    
    return RPP_SUCCESS;    
}