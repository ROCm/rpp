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
    if(!tensor.defined()) {
        throw std::runtime_error("Tensor is not defined");
    }
    
    if(!tensor.is_contiguous()) {
        throw std::runtime_error("Tensor must be contiguous");
    }

    TensorData data;
    data.ptr = tensor.data_ptr();

    if(data.ptr == nullptr) {
        throw std::runtime_error("Tensor data pointer is null");
    }
    
    // Map PyTorch dtype to RPP dtype
    if(tensor.dtype() == torch::kUInt8) {
        data.dtype = RpptDataType::U8;
    } else if(tensor.dtype() == torch::kFloat32) {
        data.dtype = RpptDataType::F32;
    } else if(tensor.dtype() == torch::kFloat16) {
        data.dtype = RpptDataType::F16;
    } else if(tensor.dtype() == torch::kInt8) {
        data.dtype = RpptDataType::I8;
    } else {
        throw std::runtime_error("Unsupported tensor dtype");
    }
    
    // Determine layout based on tensor shape (assuming NCHW by default)
    data.layout = RpptLayout::NCHW;
    
    // Get shape and strides
    for(int i = 0; i < tensor.dim(); i++) {
        data.shape.push_back(tensor.size(i));
        data.strides.push_back(tensor.stride(i));
    }
    
    // Set device
    if(tensor.is_cuda()) {
        data.device = {kDLROCM, 0};  // ROCm device
    } else {
        data.device = {kDLCPU, 0};
    }
    
    return data;
}

RpptLayout detect_layout_from_tensor(const torch::Tensor& tensor) {
    // Detect layout based on tensor shape
    // PKD3 (Packed) = NHWC -> last dim is channels (3 for RGB, 1 for grayscale)
    // PLN3 (Planar) = NCHW -> 2nd dim is channels
    if (tensor.dim() == 4) {
        if (tensor.size(3) == 3 || tensor.size(3) == 1) {
            // Last dim is channels -> NHWC (PKD3/PKD1)
            return RpptLayout::NHWC;
        } else if (tensor.size(1) == 3 || tensor.size(1) == 1) {
            // Second dim is channels -> NCHW (PLN3/PLN1)
            return RpptLayout::NCHW;
        }
    }
    // Default to NCHW
    return RpptLayout::NCHW;
}

void setup_tensor_descriptor(RpptDesc& desc, const TensorData& data) {
    desc.dataType = data.dtype;
    desc.layout = data.layout;
    desc.numDims = data.shape.size();
    desc.offsetInBytes = 0;
    
    // if(desc.numDims == 4) {
    //     desc.n = data.shape[0];
    //     desc.c = data.shape[1]; 
    //     desc.h = data.shape[2];
    //     desc.w = data.shape[3];
        
    //     desc.strides.nStride = data.strides[0];
    //     desc.strides.cStride = data.strides[1];
    //     desc.strides.hStride = data.strides[2];
    //     desc.strides.wStride = data.strides[3];
    // }
    if (desc.layout == RpptLayout::NCHW) {
        desc.n = data.shape[0];
        desc.c = data.shape[1]; 
        desc.h = data.shape[2];
        desc.w = data.shape[3];
        desc.strides.nStride = data.strides[0];
        desc.strides.cStride = data.strides[1];
        desc.strides.hStride = data.strides[2];
        desc.strides.wStride = data.strides[3];
    } 
    else if (desc.layout == RpptLayout::NHWC) {
        desc.n = data.shape[0];
        desc.h = data.shape[1];
        desc.w = data.shape[2];
        desc.c = data.shape[3];
            
        desc.strides.nStride = data.strides[0];
        desc.strides.hStride = data.strides[1];
        desc.strides.wStride = data.strides[2];
        desc.strides.cStride = data.strides[3];
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
               const std::vector<int>& roi_widths,
               const std::vector<int>& roi_heights,
               uintptr_t handle,
               int backend) {

    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);

    // Detect layouts from tensor shapes
    input_data.layout = detect_layout_from_tensor(input_tensor);
    output_data.layout = detect_layout_from_tensor(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);

    for(int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, roi_widths[i], roi_heights[i]};
    }

    // Initialize pointers to CPU data by default
    float* alpha_ptr = const_cast<float*>(alpha.data());
    float* beta_ptr = const_cast<float*>(beta.data());
    RpptROI* roi_ptr = roi.data();
    float* alpha_gpu_ptr = nullptr;
    float* beta_gpu_ptr = nullptr;
    RpptROI *roi_gpu_ptr = nullptr;

    // Check if backend is HIP (GPU)
    if(backend == 1) {
        // Allocate GPU memory
        size_t alpha_size = alpha.size() * sizeof(float);
        size_t beta_size = beta.size() * sizeof(float);
        size_t roi_size = batch_size * sizeof(RpptROI);
        
        hipMalloc(&alpha_gpu_ptr, alpha_size);
        hipMalloc(&beta_gpu_ptr, beta_size);
        hipMalloc(&roi_gpu_ptr, roi_size);
        
        // Copy CPU data to GPU
        hipMemcpy(alpha_gpu_ptr, alpha_ptr, alpha_size, hipMemcpyHostToDevice);
        hipMemcpy(beta_gpu_ptr, beta_ptr, beta_size, hipMemcpyHostToDevice);
        hipMemcpy(roi_gpu_ptr, roi.data(), batch_size * sizeof(RpptROI), hipMemcpyHostToDevice);
        
        // Use GPU pointers for the function call
        alpha_ptr = alpha_gpu_ptr;
        beta_ptr = beta_gpu_ptr;
        roi_ptr = roi_gpu_ptr;
    }
    
    rppt_brightness(input_data.ptr, &src_desc,
                   output_data.ptr, &dst_desc,
                   alpha_ptr,
                   beta_ptr,
                   roi_ptr, RpptRoiType::XYWH,
                   rpp_handle, static_cast<RppBackend>(backend));
    
    if(backend == 1)
    {
        hipFree(alpha_gpu_ptr);
        hipFree(beta_gpu_ptr);
        hipFree(roi_gpu_ptr);
    }
        
}

// 2. Gamma Correction (Color)
void gamma_correction(const torch::Tensor& input_tensor,
                     torch::Tensor& output_tensor,
                     const std::vector<float>& gamma,
                     const std::vector<int>& roi_widths,
                     const std::vector<int>& roi_heights,
                     uintptr_t handle,
                     int backend) {

    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);

    // Detect layouts from tensor shapes
    input_data.layout = detect_layout_from_tensor(input_tensor);
    output_data.layout = detect_layout_from_tensor(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for(int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, roi_widths[i], roi_heights[i]};
    }

    // Initialize pointers to CPU data by default
    float* gamma_ptr = const_cast<float*>(gamma.data());
    RpptROI* roi_ptr = roi.data();
    float* gamma_gpu_ptr = nullptr;
    RpptROI *roi_gpu_ptr = nullptr;

    // Check if backend is HIP (GPU)
    if(backend == 1) {
        // Allocate GPU memory
        size_t gamma_size = gamma.size() * sizeof(float);
        size_t roi_size = batch_size * sizeof(RpptROI);
        
        hipMalloc(&gamma_gpu_ptr, gamma_size);
        hipMalloc(&roi_gpu_ptr, roi_size);
        
        // Copy CPU data to GPU
        hipMemcpy(gamma_gpu_ptr, gamma_ptr, gamma_size, hipMemcpyHostToDevice);
        hipMemcpy(roi_gpu_ptr, roi.data(), batch_size * sizeof(RpptROI), hipMemcpyHostToDevice);
        
        // Use GPU pointers for the function call
        gamma_ptr = gamma_gpu_ptr;
        roi_ptr = roi_gpu_ptr;
    }
    
    rppt_gamma_correction(input_data.ptr, &src_desc,
                         output_data.ptr, &dst_desc,
                         gamma_ptr,
                         roi_ptr, RpptRoiType::XYWH,
                         rpp_handle, static_cast<RppBackend>(backend));


    if(backend == 1) {
        hipFree(gamma_gpu_ptr);
        hipFree(roi_gpu_ptr);
    }
}

// 3. Contrast (Color)
void contrast(const torch::Tensor& input_tensor,
             torch::Tensor& output_tensor,
             const std::vector<float>& contrast_factor,
             const std::vector<float>& contrast_center,
             const std::vector<int>& roi_widths,
             const std::vector<int>& roi_heights,
             uintptr_t handle,
             int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);

    // Detect layouts from tensor shapes
    input_data.layout = detect_layout_from_tensor(input_tensor);
    output_data.layout = detect_layout_from_tensor(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for(int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, roi_widths[i], roi_heights[i]};
    }

    // Initialize pointers to CPU data by default
    float* contrast_factor_ptr = const_cast<float*>(contrast_factor.data());
    float* contrast_center_ptr = const_cast<float*>(contrast_center.data());
    RpptROI* roi_ptr = roi.data();

    float* contrast_factor_gpu = nullptr;
    float* contrast_center_gpu = nullptr;
    RpptROI* roi_gpu = nullptr;

    if(backend == 1) {  // HIP backend
        // Allocate GPU memory
        size_t contrast_factor_size = contrast_factor.size() * sizeof(float);
        size_t contrast_center_size = contrast_center.size() * sizeof(float);
        size_t roi_size = batch_size * sizeof(RpptROI);
        
        hipMalloc(&contrast_factor_gpu, contrast_factor_size);
        hipMalloc(&contrast_center_gpu, contrast_center_size);
        hipMalloc(&roi_gpu, roi_size);
        
        // Copy CPU data to GPU
        hipMemcpy(contrast_factor_gpu, contrast_factor_ptr, contrast_factor_size, hipMemcpyHostToDevice);
        hipMemcpy(contrast_center_gpu, contrast_center_ptr, contrast_center_size, hipMemcpyHostToDevice);
        hipMemcpy(roi_gpu, roi.data(), batch_size * sizeof(RpptROI), hipMemcpyHostToDevice);
        
        // Use GPU pointers for the function call
        contrast_factor_ptr = contrast_factor_gpu;
        contrast_center_ptr = contrast_center_gpu;
        roi_ptr = roi_gpu;
    }
    
    rppt_contrast(input_data.ptr, &src_desc,
                 output_data.ptr, &dst_desc,
                 contrast_factor_ptr,
                 contrast_center_ptr,
                 roi_ptr, RpptRoiType::XYWH,
                 rpp_handle, static_cast<RppBackend>(backend));
    
    if(backend == 1) {
        hipFree(contrast_factor_gpu);
        hipFree(contrast_center_gpu);
        hipFree(roi_gpu);
    }
}

// 4. Hue (Color)
void hue(const torch::Tensor& input_tensor,
        torch::Tensor& output_tensor,
        const std::vector<float>& hue_shift,
        const std::vector<int>& roi_widths,
        const std::vector<int>& roi_heights,
        uintptr_t handle,
        int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);

    // Detect layouts from tensor shapes
    input_data.layout = detect_layout_from_tensor(input_tensor);
    output_data.layout = detect_layout_from_tensor(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for(int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, roi_widths[i], roi_heights[i]};
    }

    // Initialize pointers to CPU data by default
    float* hue_shift_ptr = const_cast<float*>(hue_shift.data());
    RpptROI* roi_ptr = roi.data();
    
    float* hue_shift_gpu = nullptr;
    RpptROI* roi_gpu = nullptr;

    if(backend == 1) {  // HIP backend
        // Allocate GPU memory
        size_t hue_shift_size = hue_shift.size() * sizeof(float);
        size_t roi_size = batch_size * sizeof(RpptROI);
        
        hipMalloc(&hue_shift_gpu, hue_shift_size);
        hipMalloc(&roi_gpu, roi_size);
        
        // Copy CPU data to GPU
        hipMemcpy(hue_shift_gpu, hue_shift_ptr, hue_shift_size, hipMemcpyHostToDevice);
        hipMemcpy(roi_gpu, roi.data(), roi_size, hipMemcpyHostToDevice);
        
        // Use GPU pointers
        hue_shift_ptr = hue_shift_gpu;
        roi_ptr = roi_gpu;
    }

    rppt_hue(input_data.ptr, &src_desc,
            output_data.ptr, &dst_desc,
            hue_shift_ptr,   
            roi_ptr, RpptRoiType::XYWH,
            rpp_handle, static_cast<RppBackend>(backend));
    
    if(backend == 1) {  // HIP backend cleanup
        hipFree(hue_shift_gpu);
        hipFree(roi_gpu);
    }
}

// 5. Flip (Geometric)
void flip(const torch::Tensor& input_tensor,
         torch::Tensor& output_tensor,
         const std::vector<int>& horizontal,
         const std::vector<int>& vertical,
         const std::vector<int>& roi_widths,
         const std::vector<int>& roi_heights,
         uintptr_t handle,
         int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);

    // Detect layouts from tensor shapes
    input_data.layout = detect_layout_from_tensor(input_tensor);
    output_data.layout = detect_layout_from_tensor(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    std::vector<Rpp32u> h_tensor(batch_size);
    std::vector<Rpp32u> v_tensor(batch_size);

    for(int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, roi_widths[i], roi_heights[i]};
        h_tensor[i] = horizontal[i];
        v_tensor[i] = vertical[i];
    }

    // Initialize pointers to CPU data by default
    Rpp32u* h_tensor_ptr = h_tensor.data();
    Rpp32u* v_tensor_ptr = v_tensor.data();
    RpptROI* roi_ptr = roi.data();

    Rpp32u* h_tensor_gpu = nullptr;
    Rpp32u* v_tensor_gpu = nullptr;
    RpptROI* roi_gpu = nullptr;

    if(backend == 1) {  // HIP backend
        // Allocate GPU memory
        size_t h_tensor_size = batch_size * sizeof(Rpp32u);
        size_t v_tensor_size = batch_size * sizeof(Rpp32u);
        size_t roi_size = batch_size * sizeof(RpptROI);
        
        hipMalloc(&h_tensor_gpu, h_tensor_size);
        hipMalloc(&v_tensor_gpu, v_tensor_size);
        hipMalloc(&roi_gpu, roi_size);
        
        // Copy CPU data to GPU
        hipMemcpy(h_tensor_gpu, h_tensor_ptr, h_tensor_size, hipMemcpyHostToDevice);
        hipMemcpy(v_tensor_gpu, v_tensor_ptr, v_tensor_size, hipMemcpyHostToDevice);
        hipMemcpy(roi_gpu, roi.data(), roi_size, hipMemcpyHostToDevice);
        
        // Use GPU pointers
        h_tensor_ptr = h_tensor_gpu;
        v_tensor_ptr = v_tensor_gpu;
        roi_ptr = roi_gpu;
    }
    
    rppt_flip(input_data.ptr, &src_desc,
             output_data.ptr, &dst_desc,
             h_tensor_ptr, v_tensor_ptr,
             roi_ptr, RpptRoiType::XYWH,
             rpp_handle, static_cast<RppBackend>(backend));
    
    if(backend == 1) {  // HIP backend cleanup
        hipFree(h_tensor_gpu);
        hipFree(v_tensor_gpu);
        hipFree(roi_gpu);
    }
}

// 6. Resize (Geometric)
void resize(const torch::Tensor& input_tensor,
           torch::Tensor& output_tensor,
           const std::vector<int>& dst_width,
           const std::vector<int>& dst_height,
           const std::vector<int>& roi_widths,
           const std::vector<int>& roi_heights,
           uintptr_t handle,
           int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);

    // Detect layouts from tensor shapes
    input_data.layout = detect_layout_from_tensor(input_tensor);
    output_data.layout = detect_layout_from_tensor(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    std::vector<RpptImagePatch> dst_sizes(batch_size);

    for(int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, roi_widths[i], roi_heights[i]};
        dst_sizes[i].width = dst_width[i];
        dst_sizes[i].height = dst_height[i];
    }

    // Initialize pointers to CPU data by default
    RpptImagePatch* dst_sizes_ptr = dst_sizes.data();
    RpptROI* roi_ptr = roi.data();

    RpptImagePatch* dst_sizes_gpu = nullptr;
    RpptROI* roi_gpu = nullptr;

    if(backend == 1) {  // HIP backend
        // Allocate GPU memory
        size_t dst_sizes_size = batch_size * sizeof(RpptImagePatch);
        size_t roi_size = batch_size * sizeof(RpptROI);
        
        hipMalloc(&dst_sizes_gpu, dst_sizes_size);
        hipMalloc(&roi_gpu, roi_size);
        
        // Copy CPU data to GPU
        hipMemcpy(dst_sizes_gpu, dst_sizes_ptr, dst_sizes_size, hipMemcpyHostToDevice);
        hipMemcpy(roi_gpu, roi.data(), roi_size, hipMemcpyHostToDevice);
        
        // Use GPU pointers
        dst_sizes_ptr = dst_sizes_gpu;
        roi_ptr = roi_gpu;
    }
    
    rppt_resize(input_data.ptr, &src_desc,
               output_data.ptr, &dst_desc,
               dst_sizes_ptr,
               RpptInterpolationType::BILINEAR,
               roi_ptr, RpptRoiType::XYWH,
               rpp_handle, static_cast<RppBackend>(backend));
    
    if(backend == 1) {  // HIP backend cleanup
        hipFree(dst_sizes_gpu);
        hipFree(roi_gpu);
    }
}

// 7. Rotate (Geometric)
void rotate(const torch::Tensor& input_tensor,
           torch::Tensor& output_tensor,
           const std::vector<float>& angle,
           const std::vector<int>& roi_widths,
           const std::vector<int>& roi_heights,
           uintptr_t handle,
           int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);

    // Detect layouts from tensor shapes
    input_data.layout = detect_layout_from_tensor(input_tensor);
    output_data.layout = detect_layout_from_tensor(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for(int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, roi_widths[i], roi_heights[i]};
    }

    // Initialize pointers to CPU data by default
    float* angle_ptr = const_cast<float*>(angle.data());
    RpptROI* roi_ptr = roi.data();

    float* angle_gpu = nullptr;
    RpptROI* roi_gpu = nullptr;

    if(backend == 1) {  // HIP backend
        // Allocate GPU memory
        size_t angle_size = angle.size() * sizeof(float);
        size_t roi_size = batch_size * sizeof(RpptROI);
        
        hipMalloc(&angle_gpu, angle_size);
        hipMalloc(&roi_gpu, roi_size);
        
        // Copy CPU data to GPU
        hipMemcpy(angle_gpu, angle_ptr, angle_size, hipMemcpyHostToDevice);
        hipMemcpy(roi_gpu, roi.data(), roi_size, hipMemcpyHostToDevice);
        
        // Use GPU pointers
        angle_ptr = angle_gpu;
        roi_ptr = roi_gpu;
    }
    
    rppt_rotate(input_data.ptr, &src_desc,
               output_data.ptr, &dst_desc,
               angle_ptr,
               RpptInterpolationType::BILINEAR,
               roi_ptr, RpptRoiType::XYWH,
               rpp_handle, static_cast<RppBackend>(backend));
    
    if(backend == 1) {  // HIP backend cleanup
        hipFree(angle_gpu);
        hipFree(roi_gpu);
    }
}

// 8. Crop (Geometric)
void crop(const torch::Tensor& input_tensor,
         torch::Tensor& output_tensor,
         const std::vector<int>& x1,
         const std::vector<int>& y1,
         const std::vector<int>& crop_width,
         const std::vector<int>& crop_height,
         uintptr_t handle,
         int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);

    // Detect layouts from tensor shapes
    input_data.layout = detect_layout_from_tensor(input_tensor);
    output_data.layout = detect_layout_from_tensor(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    
    for(int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {x1[i], 
                         y1[i], 
                         crop_width[i], 
                         crop_height[i]};
    }

    // Initialize pointers to CPU data by default
    RpptROI* roi_ptr = roi.data();

    RpptROI* roi_gpu = nullptr;

    if(backend == 1) {  // HIP backend
        // Allocate GPU memory
        size_t roi_size = batch_size * sizeof(RpptROI);
        
        hipMalloc(&roi_gpu, roi_size);
        
        // Copy CPU data to GPU
        hipMemcpy(roi_gpu, roi.data(), roi_size, hipMemcpyHostToDevice);
        
        // Use GPU pointers
        roi_ptr = roi_gpu;
    }
    
    rppt_crop(input_data.ptr, &src_desc,
             output_data.ptr, &dst_desc,
             roi_ptr, RpptRoiType::XYWH,
             rpp_handle, static_cast<RppBackend>(backend));
    
    if(backend == 1) {  // HIP backend cleanup
        hipFree(roi_gpu);
    }
}

// 9. Vignette (Effects)
void vignette(const torch::Tensor& input_tensor,
             torch::Tensor& output_tensor,
             const std::vector<float>& intensity,
             const std::vector<int>& roi_widths,
             const std::vector<int>& roi_heights,
             uintptr_t handle,
             int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);

    // Detect layouts from tensor shapes
    input_data.layout = detect_layout_from_tensor(input_tensor);
    output_data.layout = detect_layout_from_tensor(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for(int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, roi_widths[i], roi_heights[i]};
    }

    // Initialize pointers to CPU data by default
    float* intensity_ptr = const_cast<float*>(intensity.data());
    RpptROI* roi_ptr = roi.data();

    float* intensity_gpu = nullptr;
    RpptROI* roi_gpu = nullptr;

    if(backend == 1) {  // HIP backend
        // Allocate GPU memory
        size_t intensity_size = intensity.size() * sizeof(float);
        size_t roi_size = batch_size * sizeof(RpptROI);
        
        hipMalloc(&intensity_gpu, intensity_size);
        hipMalloc(&roi_gpu, roi_size);
        
        // Copy CPU data to GPU
        hipMemcpy(intensity_gpu, intensity_ptr, intensity_size, hipMemcpyHostToDevice);
        hipMemcpy(roi_gpu, roi.data(), roi_size, hipMemcpyHostToDevice);
        
        // Use GPU pointers
        intensity_ptr = intensity_gpu;
        roi_ptr = roi_gpu;
    }
    
    rppt_vignette(input_data.ptr, &src_desc,
                 output_data.ptr, &dst_desc,
                 intensity_ptr,
                 roi_ptr, RpptRoiType::XYWH,
                 rpp_handle, static_cast<RppBackend>(backend));
    
    if(backend == 1) {  // HIP backend cleanup
        hipFree(intensity_gpu);
        hipFree(roi_gpu);
    }
}

// 10. Pixelate (Effects)
void pixelate(const torch::Tensor& input_tensor,
             torch::Tensor& output_tensor,
             const torch::Tensor& scratch_tensor,
             float pixelation_pct,
             const std::vector<int>& roi_widths,
             const std::vector<int>& roi_heights,
             uintptr_t handle,
             int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    auto scratch_data = get_tensor_data(scratch_tensor);

    // Detect layouts from tensor shapes
    input_data.layout = detect_layout_from_tensor(input_tensor);
    output_data.layout = detect_layout_from_tensor(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];

    std::vector<RpptROI> roi(batch_size);
    for(int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, roi_widths[i], roi_heights[i]};
    }

    // Initialize pointers to CPU data by default
    RpptROI* roi_ptr = roi.data();

    RpptROI* roi_gpu = nullptr;

    if(backend == 1) {  // HIP backend
        // Allocate GPU memory
        size_t roi_size = batch_size * sizeof(RpptROI);
        
        hipMalloc(&roi_gpu, roi_size);
        
        // Copy CPU data to GPU
        hipMemcpy(roi_gpu, roi.data(), roi_size, hipMemcpyHostToDevice);
        
        // Use GPU pointers
        roi_ptr = roi_gpu;
    }
    
    rppt_pixelate(input_data.ptr, &src_desc,
                 output_data.ptr, &dst_desc,
                 scratch_data.ptr,
                 pixelation_pct,
                 roi_ptr, RpptRoiType::XYWH,
                 rpp_handle, static_cast<RppBackend>(backend));
    
    if(backend == 1) {  // HIP backend cleanup
        hipFree(roi_gpu);
    }
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

    // Bind the 10 augmentation functions
    m.def("brightness", &brightness, "Brightness augmentation",
          py::arg("input"), py::arg("output"), 
          py::arg("alpha"), py::arg("beta"),
          py::arg("roi_widths"), py::arg("roi_heights"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("gamma_correction", &gamma_correction, "Gamma correction",
          py::arg("input"), py::arg("output"),
          py::arg("gamma"),
          py::arg("roi_widths"), py::arg("roi_heights"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("contrast", &contrast, "Contrast augmentation",
          py::arg("input"), py::arg("output"),
          py::arg("contrast_factor"), py::arg("contrast_center"),
          py::arg("roi_widths"), py::arg("roi_heights"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("hue", &hue, "Hue augmentation",
          py::arg("input"), py::arg("output"),
          py::arg("hue_shift"),
          py::arg("roi_widths"), py::arg("roi_heights"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("flip", &flip, "Flip augmentation",
          py::arg("input"), py::arg("output"),
          py::arg("horizontal"), py::arg("vertical"),
          py::arg("roi_widths"), py::arg("roi_heights"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("resize", &resize, "Resize augmentation",
          py::arg("input"), py::arg("output"),
          py::arg("dst_width"), py::arg("dst_height"),
          py::arg("roi_widths"), py::arg("roi_heights"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("rotate", &rotate, "Rotate augmentation",
          py::arg("input"), py::arg("output"),
          py::arg("angle"),
          py::arg("roi_widths"), py::arg("roi_heights"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("crop", &crop, "Crop augmentation",
          py::arg("input"), py::arg("output"),
          py::arg("x1"), py::arg("y1"), 
          py::arg("crop_width"), py::arg("crop_height"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("vignette", &vignette, "Vignette effect",
          py::arg("input"), py::arg("output"),
          py::arg("intensity"),
          py::arg("roi_widths"), py::arg("roi_heights"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("pixelate", &pixelate, "Pixelate effect",
          py::arg("input"), py::arg("output"), py::arg("scratch"),
          py::arg("pixelation_pct"),
          py::arg("roi_widths"), py::arg("roi_heights"),
          py::arg("handle"), py::arg("backend"));
}
