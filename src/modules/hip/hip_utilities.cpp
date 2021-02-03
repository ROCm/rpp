#include "hip_declarations.hpp"

void max_size(Rpp32u *height, Rpp32u *width, unsigned int batch_size, unsigned int *max_height, unsigned int *max_width)
{
    int i;
    *max_height  = 0;
    *max_width =0;
    for (i=0; i<batch_size; i++){
        if(*max_height < height[i])
            *max_height = height[i];
        if(*max_width < width[i])
            *max_width = width[i];
    }
}

void get_kernel_name(std::string &kernel_name, const RPPTensorFunctionMetaData &tensor_info)
{
    switch (tensor_info._in_type)
    {
    case RPPTensorDataType::U8:
        switch (tensor_info._out_type)
        {
        case RPPTensorDataType::U8:
            break;
        case RPPTensorDataType::FP32:
            kernel_name = kernel_name + "_u8_fp32";
            break;
        case RPPTensorDataType::FP16:
            kernel_name = kernel_name + "_u8_fp16";
            break;
        case RPPTensorDataType::I8:
            kernel_name = kernel_name + "_u8_int8";
            break;
        default:
            break;
        }
        break;
    case RPPTensorDataType::FP32:
        kernel_name = kernel_name + "_fp32";
        break;
    case RPPTensorDataType::FP16:
        kernel_name = kernel_name + "_fp16";
        break;
    case RPPTensorDataType::I8:
        kernel_name = kernel_name + "_int8";
        break;
    default:
        break;
    }
}
