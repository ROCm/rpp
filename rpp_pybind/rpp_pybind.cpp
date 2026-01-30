// # Mukesh/rpp/rpp_pybind/rpp_pybind.cpp
/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#ifdef STATIC
#undef STATIC
#endif
#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/dlpack.h>
#include <dlpack/dlpack.h>

#include <rpp.h>
#include <rppdefs.h>
#include <rppt_tensor_color_augmentations.h>
#include <rppt_tensor_geometric_augmentations.h>
#include <rppt_tensor_effects_augmentations.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

// -----------------------------------------------------
// DLPack Helpers
// -----------------------------------------------------
struct TensorData {
    void* ptr;
    RpptDataType dtype;
    RpptLayout layout;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLDevice device;
};

TensorData get_tensor_data(const torch::Tensor& tensor) {
    if (!tensor.defined()) {
        throw std::runtime_error("Tensor is not defined");
    }
    
    if (!tensor.is_contiguous()) {
        throw std::runtime_error("Tensor must be contiguous");
    }

    TensorData data;
    data.ptr = tensor.data_ptr();

    if (data.ptr == nullptr) {
        throw std::runtime_error("Tensor data pointer is null");
    }
    
    // Map PyTorch dtype to RPP dtype
    if (tensor.dtype() == torch::kUInt8) {
        data.dtype = RpptDataType::U8;
    } else if (tensor.dtype() == torch::kFloat32) {
        data.dtype = RpptDataType::F32;
    } else if (tensor.dtype() == torch::kFloat16) {
        data.dtype = RpptDataType::F16;
    } else if (tensor.dtype() == torch::kInt8) {
        data.dtype = RpptDataType::I8;
    } else {
        throw std::runtime_error("Unsupported tensor dtype");
    }
    
    // Determine layout based on tensor shape (assuming NCHW by default)
    data.layout = RpptLayout::NCHW;
    
    // Get shape and strides
    for (int i = 0; i < tensor.dim(); i++) {
        data.shape.push_back(tensor.size(i));
        data.strides.push_back(tensor.stride(i));
    }
    
    // Set device
    if (tensor.is_cuda()) {
        data.device = {kDLROCM, 0};  // ROCm device
    } else {
        data.device = {kDLCPU, 0};
    }
    
    return data;
}

void setup_tensor_descriptor(RpptDesc& desc, const TensorData& data) {
    std::cout << "DEBUG: Setting up descriptor" << std::endl;
    std::cout << "DEBUG: shape = [" << data.shape[0] << "," << data.shape[1] 
              << "," << data.shape[2] << "," << data.shape[3] << "]" << std::endl;
    desc.dataType = data.dtype;
    desc.layout = data.layout;
    desc.numDims = data.shape.size();
    desc.offsetInBytes = 0;
    
    if (desc.numDims == 4) {
        desc.n = data.shape[0];
        desc.c = data.shape[1]; 
        desc.h = data.shape[2];
        desc.w = data.shape[3];
        std::cout << "DEBUG: n=" << desc.n << ", c=" << desc.c 
                  << ", h=" << desc.h << ", w=" << desc.w << std::endl;
        
        // Calculate strides in bytes
        size_t element_size = (data.dtype == RpptDataType::U8 || data.dtype == RpptDataType::I8) ? 1 :
                            (data.dtype == RpptDataType::F16) ? 2 : 4;
        
        desc.strides.nStride = data.strides[0] * element_size;
        desc.strides.cStride = data.strides[1] * element_size;
        desc.strides.hStride = data.strides[2] * element_size;
        desc.strides.wStride = data.strides[3] * element_size;
    }
}

// -----------------------------------------------------
// Core wrapper functions (10 augmentations)
// -----------------------------------------------------

// 1. Brightness (Color)
void brightness(const torch::Tensor& input_tensor,
               torch::Tensor& output_tensor,
               const std::vector<float>& alpha,
               const std::vector<float>& beta,
               uintptr_t handle,
               int backend) {
    std::cout << "DEBUG: C++ brightness() called" << std::endl;
    std::cout << "DEBUG: input tensor shape: " << input_tensor.sizes() << std::endl;
    std::cout << "DEBUG: handle = " << handle << std::endl;
    std::cout << "DEBUG: backend = " << backend << std::endl;

    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    std::cout << "DEBUG: get_tensor_data() completed" << std::endl;
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    std::cout << "Batchsize: " << batch_size;
    for (int i = 0; i < batch_size; i++) {
        int width = static_cast<int>(input_data.shape[3]);   // 56
        int height = static_cast<int>(input_data.shape[2]);  // 50
        
        roi[i].xywhROI.xy.x = 0;
        roi[i].xywhROI.xy.y = 0; 
        roi[i].xywhROI.roiWidth = width;   // 56
        roi[i].xywhROI.roiHeight = height; // 50
        
        std::cout << "DEBUG: Set ROI[" << i << "] = {0, 0, " 
                << width << ", " << height << "}" << std::endl;
    }

    std::cout << "DEBUG: Data pointers - input=" << input_data.ptr 
            << ", output=" << output_data.ptr << std::endl;
    std::cout << "DEBUG: Strides - input n=" << src_desc.strides.nStride 
            << ", c=" << src_desc.strides.cStride << std::endl;
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
    }
    // rppHandle_t rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    // Add critical debugging before RPP call
    std::cout << "DEBUG: Complete descriptor info:" << std::endl;
    std::cout << "  src dataType=" << static_cast<int>(src_desc.dataType) << " (0=U8, 2=F32)" << std::endl;
    std::cout << "  src layout=" << static_cast<int>(src_desc.layout) << std::endl; 
    std::cout << "  src offsetInBytes=" << src_desc.offsetInBytes << std::endl;
    std::cout << "  dst dataType=" << static_cast<int>(dst_desc.dataType) << std::endl;
    
    // Validate parameters before call
    std::cout << "DEBUG: Parameter validation:" << std::endl;
    std::cout << "  alpha size=" << alpha.size() << ", values=[" << alpha[0] << "]" << std::endl;
    std::cout << "  beta size=" << beta.size() << ", values=[" << beta[0] << "]" << std::endl;
    std::cout << "  batch_size=" << batch_size << std::endl;
    
    // Check if data type conversion is needed
    if (src_desc.dataType != RpptDataType::U8) {
        std::cout << "WARNING: Input tensor is not U8! RPP brightness expects U8 input." << std::endl;
        std::cout << "  Current input type: " << static_cast<int>(src_desc.dataType) << std::endl;
    }
    
    std::cout << "DEBUG: About to call rppt_brightness()" << std::endl;
    rppt_brightness(input_data.ptr, &src_desc,
                   output_data.ptr, &dst_desc,
                   const_cast<float*>(alpha.data()),
                   const_cast<float*>(beta.data()),
                   roi.data(), RpptRoiType::XYWH,
                   rpp_handle, static_cast<RppBackend>(backend));
    std::cout << "DEBUG: rppt_brightness() completed" << std::endl;
        
    
    // In brightness() function, improve the HIP cleanup section:
    // if (backend == 1) {  // HIP backend
    //     std::cout << "DEBUG: HIP backend cleanup..." << std::endl;
        
    //     // ADD THESE CRITICAL FIXES:
    //     hipError_t hipErr = hipDeviceSynchronize();
    //     if (hipErr != hipSuccess) {
    //         std::cout << "ERROR: GPU sync failed: " << hipGetErrorString(hipErr) << std::endl;
    //     }
        
    //     // Ensure all GPU operations complete before returning
    //     hipStreamSynchronize(0);  // Sync default stream
    // }
        
    std::cout << "DEBUG: brightness function complete" << std::endl;
}

// // 2. Gamma Correction (Color)
// void gamma_correction(const torch::Tensor& input_tensor,
//                      torch::Tensor& output_tensor,
//                      const std::vector<float>& gamma,
//                      uintptr_t handle,
//                      int backend) {
//     auto input_data = get_tensor_data(input_tensor);
//     auto output_data = get_tensor_data(output_tensor);
    
//     RpptDesc src_desc, dst_desc;
//     memset(&src_desc, 0, sizeof(RpptDesc)); 
//     memset(&dst_desc, 0, sizeof(RpptDesc));
//     setup_tensor_descriptor(src_desc, input_data);
//     setup_tensor_descriptor(dst_desc, output_data);
//     auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
//     int batch_size = input_data.shape[0];
//     std::vector<RpptROI> roi(batch_size);
//     for (int i = 0; i < batch_size; i++) {
//         int width = static_cast<int>(input_data.shape[3]);
//         int height = static_cast<int>(input_data.shape[2]);
        
//         roi[i].xywhROI.xy.x = 0;
//         roi[i].xywhROI.xy.y = 0; 
//         roi[i].xywhROI.roiWidth = width;
//         roi[i].xywhROI.roiHeight = height;
//     }
    
//     // rppHandle_t rpp_handle = reinterpret_cast<rppHandle_t>(handle);

//     rppt_gamma_correction(input_data.ptr, &src_desc,
//                          output_data.ptr, &dst_desc,
//                          const_cast<float*>(gamma.data()),
//                          roi.data(), RpptRoiType::XYWH,
//                          rpp_handle, static_cast<RppBackend>(backend));
//     if (backend == 1) {
//         std::cout << "DEBUG: HIP backend cleanup..." << std::endl;
//     }
// }

// 2. Gamma Correction (Color)
void gamma_correction(const torch::Tensor& input_tensor,
                     torch::Tensor& output_tensor,
                     const std::vector<float>& gamma,
                     uintptr_t handle,
                     int backend) {
    std::cout << "DEBUG: C++ gamma_correction() called" << std::endl;
    std::cout << "DEBUG: input tensor shape: " << input_tensor.sizes() << std::endl;
    std::cout << "DEBUG: handle = " << handle << std::endl;
    std::cout << "DEBUG: backend = " << backend << std::endl;

    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    std::cout << "DEBUG: get_tensor_data() completed" << std::endl;
    
    RpptDesc src_desc, dst_desc;
    memset(&src_desc, 0, sizeof(RpptDesc)); 
    memset(&dst_desc, 0, sizeof(RpptDesc));
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for (int i = 0; i < batch_size; i++) {
        int width = static_cast<int>(input_data.shape[3]);
        int height = static_cast<int>(input_data.shape[2]);
        
        roi[i].xywhROI.xy.x = 0;
        roi[i].xywhROI.xy.y = 0; 
        roi[i].xywhROI.roiWidth = width;
        roi[i].xywhROI.roiHeight = height;
        
        std::cout << "DEBUG: Set ROI[" << i << "] = {0, 0, " 
                << width << ", " << height << "}" << std::endl;
    }

    std::cout << "DEBUG: Data pointers - input=" << input_data.ptr 
            << ", output=" << output_data.ptr << std::endl;
    std::cout << "DEBUG: Strides - input n=" << src_desc.strides.nStride 
            << ", c=" << src_desc.strides.cStride << std::endl;
    
    // Add critical debugging before RPP call
    std::cout << "DEBUG: Complete descriptor info:" << std::endl;
    std::cout << "  src dataType=" << static_cast<int>(src_desc.dataType) << " (0=U8, 2=F32)" << std::endl;
    std::cout << "  src layout=" << static_cast<int>(src_desc.layout) << std::endl; 
    std::cout << "  src offsetInBytes=" << src_desc.offsetInBytes << std::endl;
    std::cout << "  dst dataType=" << static_cast<int>(dst_desc.dataType) << std::endl;
    
    // Validate parameters before call
    std::cout << "DEBUG: Parameter validation:" << std::endl;
    std::cout << "  gamma size=" << gamma.size() << ", values=[" << gamma[0] << "]" << std::endl;
    std::cout << "  batch_size=" << batch_size << std::endl;
    
    // Check if data type conversion is needed
    if (src_desc.dataType != RpptDataType::U8) {
        std::cout << "WARNING: Input tensor is not U8! RPP gamma_correction expects U8 input." << std::endl;
        std::cout << "  Current input type: " << static_cast<int>(src_desc.dataType) << std::endl;
    }
    
    std::cout << "DEBUG: About to call rppt_gamma_correction()" << std::endl;
    rppt_gamma_correction(input_data.ptr, &src_desc,
                         output_data.ptr, &dst_desc,
                         const_cast<float*>(gamma.data()),
                         roi.data(), RpptRoiType::XYWH,
                         rpp_handle, static_cast<RppBackend>(backend));
    std::cout << "DEBUG: rppt_gamma_correction() completed" << std::endl;
    
    // Add GPU-specific cleanup for HIP backend
    if (backend == 1) {  // HIP backend
        std::cout << "DEBUG: HIP backend cleanup..." << std::endl;
        // Ensure GPU operations complete before cleanup
        // Add potential GPU sync or cleanup calls here
    }
    
    std::cout << "DEBUG: gamma_correction function complete" << std::endl;
}

// 3. Contrast (Color)
void contrast(const torch::Tensor& input_tensor,
             torch::Tensor& output_tensor,
             const std::vector<float>& contrast_factor,
             const std::vector<float>& contrast_center,
             rppHandle_t handle,
             int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
    }
    
    rppt_contrast(input_data.ptr, &src_desc,
                 output_data.ptr, &dst_desc,
                 const_cast<float*>(contrast_factor.data()),
                 const_cast<float*>(contrast_center.data()),
                 roi.data(), RpptRoiType::XYWH,
                 handle, static_cast<RppBackend>(backend));
}

// 4. Hue (Color)
void hue(const torch::Tensor& input_tensor,
        torch::Tensor& output_tensor,
        const std::vector<float>& hue_shift,
        rppHandle_t handle,
        int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
    }
    
    rppt_hue(input_data.ptr, &src_desc,
            output_data.ptr, &dst_desc,
            const_cast<float*>(hue_shift.data()),
            roi.data(), RpptRoiType::XYWH,
            handle, static_cast<RppBackend>(backend));
}

// 5. Flip (Geometric)
void flip(const torch::Tensor& input_tensor,
         torch::Tensor& output_tensor,
         const std::vector<int>& horizontal,
         const std::vector<int>& vertical,
         rppHandle_t handle,
         int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    std::vector<Rpp32u> h_tensor(batch_size);
    std::vector<Rpp32u> v_tensor(batch_size);
    
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
        h_tensor[i] = horizontal[i];
        v_tensor[i] = vertical[i];
    }
    
    rppt_flip(input_data.ptr, &src_desc,
             output_data.ptr, &dst_desc,
             h_tensor.data(), v_tensor.data(),
             roi.data(), RpptRoiType::XYWH,
             handle, static_cast<RppBackend>(backend));
}

// 6. Resize (Geometric)
void resize(const torch::Tensor& input_tensor,
           torch::Tensor& output_tensor,
           const std::vector<int>& dst_width,
           const std::vector<int>& dst_height,
           rppHandle_t handle,
           int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    std::vector<RpptImagePatch> dst_sizes(batch_size);
    
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
        dst_sizes[i].width = dst_width[i];
        dst_sizes[i].height = dst_height[i];
    }
    
    rppt_resize(input_data.ptr, &src_desc,
               output_data.ptr, &dst_desc,
               dst_sizes.data(),
               RpptInterpolationType::BILINEAR,
               roi.data(), RpptRoiType::XYWH,
               handle, static_cast<RppBackend>(backend));
}

// 7. Rotate (Geometric)
void rotate(const torch::Tensor& input_tensor,
           torch::Tensor& output_tensor,
           const std::vector<float>& angle,
           rppHandle_t handle,
           int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
    }
    
    rppt_rotate(input_data.ptr, &src_desc,
               output_data.ptr, &dst_desc,
               const_cast<float*>(angle.data()),
               RpptInterpolationType::BILINEAR,
               roi.data(), RpptRoiType::XYWH,
               handle, static_cast<RppBackend>(backend));
}

// 8. Crop (Geometric)
void crop(const torch::Tensor& input_tensor,
         torch::Tensor& output_tensor,
         const std::vector<int>& x1,
         const std::vector<int>& y1,
         const std::vector<int>& crop_width,
         const std::vector<int>& crop_height,
         rppHandle_t handle,
         int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {x1[i], 
                         y1[i], 
                         crop_width[i], 
                         crop_height[i]};
    }
    
    rppt_crop(input_data.ptr, &src_desc,
             output_data.ptr, &dst_desc,
             roi.data(), RpptRoiType::XYWH,
             handle, static_cast<RppBackend>(backend));
}

// 9. Vignette (Effects)
void vignette(const torch::Tensor& input_tensor,
             torch::Tensor& output_tensor,
             const std::vector<float>& intensity,
             rppHandle_t handle,
             int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
    }
    
    rppt_vignette(input_data.ptr, &src_desc,
                 output_data.ptr, &dst_desc,
                 const_cast<float*>(intensity.data()),
                 roi.data(), RpptRoiType::XYWH,
                 handle, static_cast<RppBackend>(backend));
}

// 10. Pixelate (Effects)
void pixelate(const torch::Tensor& input_tensor,
             torch::Tensor& output_tensor,
             const torch::Tensor& scratch_tensor,
             float pixelation_pct,
             rppHandle_t handle,
             int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    auto scratch_data = get_tensor_data(scratch_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
    }
    
    rppt_pixelate(input_data.ptr, &src_desc,
                 output_data.ptr, &dst_desc,
                 scratch_data.ptr,
                 pixelation_pct,
                 roi.data(), RpptRoiType::XYWH,
                 handle, static_cast<RppBackend>(backend));
}

// -----------------------------------------------------
// Python module definition
// -----------------------------------------------------
PYBIND11_MODULE(_rpp_pybind, m) {
    m.doc() = "PyRPP - Python bindings for AMD ROCm Performance Primitives";
    py::print("Initializing RPP Python bindings...");

    // Version info
    m.attr("__version__") = "1.0.0";
    
    // Types module
    auto types_module = m.def_submodule("types", "RPP type definitions");
    
    py::enum_<RppBackend>(types_module, "RppBackend")
        .value("HOST", RPP_HOST_BACKEND)
        .value("HIP", RPP_HIP_BACKEND);
    
    py::enum_<RppStatus>(types_module, "RppStatus")
        .value("SUCCESS", RPP_SUCCESS)
        .value("ERROR", RPP_ERROR);
    
    py::enum_<RpptDataType>(types_module, "RpptDataType")
        .value("U8", RpptDataType::U8)
        .value("F16", RpptDataType::F16)
        .value("F32", RpptDataType::F32)
        .value("I8", RpptDataType::I8)
        .value("I16", RpptDataType::I16);
    
    py::enum_<RpptLayout>(types_module, "RpptLayout")
        .value("NCHW", RpptLayout::NCHW)
        .value("NHWC", RpptLayout::NHWC)
        .value("NCDHW", RpptLayout::NCDHW)
        .value("NDHWC", RpptLayout::NDHWC)
        .value("NHW", RpptLayout::NHW)
        .value("NFT", RpptLayout::NFT)
        .value("NTF", RpptLayout::NTF);

    // Handle management
    // reinterpret_cast<rppHandle_t>(static_cast<uintptr_t>(handle))
    m.def("rppCreate", [](int batch_size, int backend) {
        rppHandle_t handle = nullptr;
        std::cout << "DEBUG: Creating handle with batch=" << batch_size 
              << ", backend=" << backend << std::endl;
        rppCreate(&handle,
                  static_cast<size_t>(batch_size),
                  0,
                  nullptr,
                  static_cast<RppBackend>(backend));

        return reinterpret_cast<uintptr_t>(handle);
    }, "Create RPP handle", py::arg("batch_size"), py::arg("backend"));
    
    m.def("rppDestroy", [](uintptr_t handle, int backend) {
        rppDestroy(reinterpret_cast<rppHandle_t>(handle), static_cast<RppBackend>(backend));
    }, "Destroy RPP handle", py::arg("handle"), py::arg("backend"));

    // // Bind the 10 augmentation functions
    // m.def("brightness", [](const torch::Tensor& input, torch::Tensor& output,
    //     const std::vector<float>& alpha, const std::vector<float>& beta,
    //     py::int_ handle, int backend){
    //         brightness(input, output, alpha, beta, reinterpret_cast<rppHandle_t>(static_cast<uintptr_t>(handle)), backend);
    //     },
    //     "Brightness augmentation",
    //     py::arg("input"), py::arg("output"),
    //     py::arg("alpha"), py::arg("beta"),
    //     py::arg("handle"), py::arg("backend")
    // );

    // // 1. gamma_correction
    // m.def("gamma_correction", [](const torch::Tensor& input,
    //     torch::Tensor& output,
    //     const std::vector<float>& gamma,
    //     py::int_ handle,
    //     int backend) {
    //         gamma_correction(
    //             input,
    //             output,
    //             gamma,
    //             reinterpret_cast<rppHandle_t>(static_cast<uintptr_t>(handle)),
    //             backend
    //         );
    //     }, "Gamma correction",
    //     py::arg("input"), py::arg("output"),
    //     py::arg("gamma"),
    //     py::arg("handle"), py::arg("backend"));

    m.def("brightness", &brightness, "Brightness augmentation",
          py::arg("input"), py::arg("output"), 
          py::arg("alpha"), py::arg("beta"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("gamma_correction", &gamma_correction, "Gamma correction",
          py::arg("input"), py::arg("output"),
          py::arg("gamma"), 
          py::arg("handle"), py::arg("backend"));
    
    // m.def("contrast", &contrast, "Contrast augmentation",
    //       py::arg("input"), py::arg("output"),
    //       py::arg("contrast_factor"), py::arg("contrast_center"),
    //       py::arg("handle"), py::arg("backend"));
    
    // m.def("hue", &hue, "Hue augmentation",
    //       py::arg("input"), py::arg("output"),
    //       py::arg("hue_shift"),
    //       py::arg("handle"), py::arg("backend"));
    
    // m.def("flip", &flip, "Flip augmentation",
    //       py::arg("input"), py::arg("output"),
    //       py::arg("horizontal"), py::arg("vertical"),
    //       py::arg("handle"), py::arg("backend"));
    
    // m.def("resize", &resize, "Resize augmentation",
    //       py::arg("input"), py::arg("output"),
    //       py::arg("dst_width"), py::arg("dst_height"),
    //       py::arg("handle"), py::arg("backend"));
    
    // m.def("rotate", &rotate, "Rotate augmentation",
    //       py::arg("input"), py::arg("output"),
    //       py::arg("angle"),
    //       py::arg("handle"), py::arg("backend"));
    
    // m.def("crop", &crop, "Crop augmentation",
    //       py::arg("input"), py::arg("output"),
    //       py::arg("x1"), py::arg("y1"), py::arg("crop_width"), py::arg("crop_height"),
    //       py::arg("handle"), py::arg("backend"));
    
    // m.def("vignette", &vignette, "Vignette effect",
    //       py::arg("input"), py::arg("output"),
    //       py::arg("intensity"),
    //       py::arg("handle"), py::arg("backend"));
    
    // m.def("pixelate", &pixelate, "Pixelate effect",
    //       py::arg("input"), py::arg("output"), py::arg("scratch"),
    //       py::arg("pixelation_pct"),
    //       py::arg("handle"), py::arg("backend"));
}
