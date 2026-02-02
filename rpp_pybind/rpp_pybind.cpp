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

    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    // std::cout << "Batchsize: " << batch_size;
    // for (int i = 0; i < batch_size; i++) {
    //     int width = static_cast<int>(input_data.shape[3]);   
    //     int height = static_cast<int>(input_data.shape[2]); 
        
    //     roi[i].xywhROI.xy.x = 0;
    //     roi[i].xywhROI.xy.y = 0; 
    //     roi[i].xywhROI.roiWidth = width;   
    //     roi[i].xywhROI.roiHeight = height; 
        
    //     std::cout << "DEBUG: Set ROI[" << i << "] = {0, 0, " 
    //             << width << ", " << height << "}" << std::endl;
    // }

    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
    }

    // Initialize pointers to CPU data by default
    float* alpha_ptr = const_cast<float*>(alpha.data());
    float* beta_ptr = const_cast<float*>(beta.data());
    RpptROI* roi_ptr = roi.data();
    float* alpha_gpu_ptr = nullptr;
    float* beta_gpu_ptr = nullptr;
    RpptROI *roi_gpu_ptr = nullptr;

    // Check if backend is HIP (GPU)
    if (backend == 1) {
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
    
    if (backend == 1)
    {
        hipFree(alpha_gpu_ptr);
        hipFree(beta_gpu_ptr);
        hipFree(roi_gpu_ptr);
    }
        
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

    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
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
    }

    // Initialize pointers to CPU data by default
    float* gamma_ptr = const_cast<float*>(gamma.data());
    RpptROI* roi_ptr = roi.data();

#ifdef __HIP_PLATFORM_AMD__
    float *gamma_gpu = nullptr;
    RpptROI *roi_gpu = nullptr;
   
    if (backend == 1) {  // HIP backend - allocate on GPU
        std::cout << "DEBUG: HIP backend - allocating parameters on GPU..." << std::endl;
       
        // Allocate GPU memory for gamma
        hipError_t err = hipMalloc(&gamma_gpu, batch_size * sizeof(float));
        if (err != hipSuccess) {
            std::cout << "ERROR: Failed to allocate gamma on GPU: " << hipGetErrorString(err) << std::endl;
            throw std::runtime_error("GPU memory allocation failed for gamma");
        }
       
        // Allocate GPU memory for ROI
        err = hipMalloc(&roi_gpu, batch_size * sizeof(RpptROI));
        if (err != hipSuccess) {
            hipFree(gamma_gpu);
            std::cout << "ERROR: Failed to allocate ROI on GPU: " << hipGetErrorString(err) << std::endl;
            throw std::runtime_error("GPU memory allocation failed for ROI");
        }
       
        // Copy data to GPU
        hipMemcpy(gamma_gpu, gamma.data(), batch_size * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(roi_gpu, roi.data(), batch_size * sizeof(RpptROI), hipMemcpyHostToDevice);
       
        // Use GPU pointers
        gamma_ptr = gamma_gpu;
        roi_ptr = roi_gpu;
       
        std::cout << "DEBUG: GPU parameter allocation complete" << std::endl;
        std::cout << "  gamma_gpu=" << static_cast<void*>(gamma_gpu) << std::endl;
        std::cout << "  roi_gpu=" << static_cast<void*>(roi_gpu) << std::endl;
    }
#endif
    
    rppt_gamma_correction(input_data.ptr, &src_desc,
                         output_data.ptr, &dst_desc,
                         gamma_ptr,
                         roi_ptr, RpptRoiType::XYWH,
                         rpp_handle, static_cast<RppBackend>(backend));

#ifdef __HIP_PLATFORM_AMD__
    if (backend == 1) {  // HIP backend - cleanup GPU memory
        std::cout << "DEBUG: HIP backend cleanup..." << std::endl;
        
        // Synchronize GPU operations before cleanup
        hipError_t hipErr = hipDeviceSynchronize();
        if (hipErr != hipSuccess) {
            std::cout << "ERROR: GPU sync failed: " << hipGetErrorString(hipErr) << std::endl;
        }
        
        // Free GPU memory
        if (gamma_gpu) {
            hipFree(gamma_gpu);
            std::cout << "DEBUG: Freed gamma_gpu memory" << std::endl;
        }
        if (roi_gpu) {
            hipFree(roi_gpu);
            std::cout << "DEBUG: Freed roi_gpu memory" << std::endl;
        }
    }
#endif

}

// 3. Contrast (Color)
void contrast(const torch::Tensor& input_tensor,
             torch::Tensor& output_tensor,
             const std::vector<float>& contrast_factor,
             const std::vector<float>& contrast_center,
             uintptr_t handle,
             int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
    }

    // Initialize pointers to CPU data by default
    float* contrast_factor_ptr = const_cast<float*>(contrast_factor.data());
    float* contrast_center_ptr = const_cast<float*>(contrast_center.data());
    RpptROI* roi_ptr = roi.data();

    float* contrast_factor_gpu = nullptr;
    float* contrast_center_gpu = nullptr;
    RpptROI* roi_gpu = nullptr;

    if (backend == 1) {  // HIP backend
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
        hipMemcpy(roi_gpu, roi.data(), roi_size, hipMemcpyHostToDevice);
        
        // Use GPU pointers
        contrast_factor_ptr = contrast_factor_gpu;
        contrast_center_ptr = contrast_center_gpu;
        roi_ptr = roi_gpu;
    }
    
    rppt_contrast(input_data.ptr, &src_desc,
                 output_data.ptr, &dst_desc,
                 const_cast<float*>(contrast_factor.data()),
                 const_cast<float*>(contrast_center.data()),
                 roi.data(), RpptRoiType::XYWH,
                 rpp_handle, static_cast<RppBackend>(backend));
    
    if (backend == 1) {  // HIP backend cleanup
        hipFree(contrast_factor_gpu);
        hipFree(contrast_center_gpu);
        hipFree(roi_gpu);
    }
}

// 4. Hue (Color)
void hue(const torch::Tensor& input_tensor,
        torch::Tensor& output_tensor,
        const std::vector<float>& hue_shift,
        uintptr_t handle,
        int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
    }

    // Initialize pointers to CPU data by default
    float* hue_shift_ptr = const_cast<float*>(hue_shift.data());
    RpptROI* roi_ptr = roi.data();
    
    float* hue_shift_gpu = nullptr;
    RpptROI* roi_gpu = nullptr;

    if (backend == 1) {  // HIP backend
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
            const_cast<float*>(hue_shift.data()),
            roi.data(), RpptRoiType::XYWH,
            rpp_handle, static_cast<RppBackend>(backend));
    
    if (backend == 1) {  // HIP backend cleanup
        hipFree(hue_shift_gpu);
        hipFree(roi_gpu);
    }
}

// 5. Flip (Geometric)
void flip(const torch::Tensor& input_tensor,
         torch::Tensor& output_tensor,
         const std::vector<int>& horizontal,
         const std::vector<int>& vertical,
         uintptr_t handle,
         int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    std::vector<Rpp32u> h_tensor(batch_size);
    std::vector<Rpp32u> v_tensor(batch_size);
    
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
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

    if (backend == 1) {  // HIP backend
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
             h_tensor.data(), v_tensor.data(),
             roi.data(), RpptRoiType::XYWH,
             rpp_handle, static_cast<RppBackend>(backend));
    
    if (backend == 1) {  // HIP backend cleanup
        hipFree(h_tensor_gpu);
        hipFree(v_tensor_gpu);
        hipFree(roi_gpu);
    }
    std::cout << "DEBUG: Flip HIP backend cleanup..." << std::endl;
}

// 6. Resize (Geometric)
void resize(const torch::Tensor& input_tensor,
           torch::Tensor& output_tensor,
           const std::vector<int>& dst_width,
           const std::vector<int>& dst_height,
           uintptr_t handle,
           int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
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
               rpp_handle, static_cast<RppBackend>(backend));
}

// 7. Rotate (Geometric)
void rotate(const torch::Tensor& input_tensor,
           torch::Tensor& output_tensor,
           const std::vector<float>& angle,
           uintptr_t handle,
           int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
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
               rpp_handle, static_cast<RppBackend>(backend));
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
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
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
             rpp_handle, static_cast<RppBackend>(backend));
}

// 9. Vignette (Effects)
void vignette(const torch::Tensor& input_tensor,
             torch::Tensor& output_tensor,
             const std::vector<float>& intensity,
             uintptr_t handle,
             int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
    int batch_size = input_data.shape[0];
    std::vector<RpptROI> roi(batch_size);
    for (int i = 0; i < batch_size; i++) {
        roi[i].xywhROI = {0, 0, static_cast<int>(input_data.shape[3]), static_cast<int>(input_data.shape[2])};
    }
    
    rppt_vignette(input_data.ptr, &src_desc,
                 output_data.ptr, &dst_desc,
                 const_cast<float*>(intensity.data()),
                 roi.data(), RpptRoiType::XYWH,
                 rpp_handle, static_cast<RppBackend>(backend));
}

// 10. Pixelate (Effects)
void pixelate(const torch::Tensor& input_tensor,
             torch::Tensor& output_tensor,
             const torch::Tensor& scratch_tensor,
             float pixelation_pct,
             uintptr_t handle,
             int backend) {
    auto input_data = get_tensor_data(input_tensor);
    auto output_data = get_tensor_data(output_tensor);
    auto scratch_data = get_tensor_data(scratch_tensor);
    
    RpptDesc src_desc, dst_desc;
    setup_tensor_descriptor(src_desc, input_data);
    setup_tensor_descriptor(dst_desc, output_data);
    auto rpp_handle = reinterpret_cast<rppHandle_t>(handle);
    
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
                 rpp_handle, static_cast<RppBackend>(backend));
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

    // Bind the 10 augmentation functions
    m.def("brightness", &brightness, "Brightness augmentation",
          py::arg("input"), py::arg("output"), 
          py::arg("alpha"), py::arg("beta"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("gamma_correction", &gamma_correction, "Gamma correction",
          py::arg("input"), py::arg("output"),
          py::arg("gamma"), 
          py::arg("handle"), py::arg("backend"));
    
    m.def("contrast", &contrast, "Contrast augmentation",
          py::arg("input"), py::arg("output"),
          py::arg("contrast_factor"), py::arg("contrast_center"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("hue", &hue, "Hue augmentation",
          py::arg("input"), py::arg("output"),
          py::arg("hue_shift"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("flip", &flip, "Flip augmentation",
          py::arg("input"), py::arg("output"),
          py::arg("horizontal"), py::arg("vertical"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("resize", &resize, "Resize augmentation",
          py::arg("input"), py::arg("output"),
          py::arg("dst_width"), py::arg("dst_height"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("rotate", &rotate, "Rotate augmentation",
          py::arg("input"), py::arg("output"),
          py::arg("angle"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("crop", &crop, "Crop augmentation",
          py::arg("input"), py::arg("output"),
          py::arg("x1"), py::arg("y1"), py::arg("crop_width"), py::arg("crop_height"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("vignette", &vignette, "Vignette effect",
          py::arg("input"), py::arg("output"),
          py::arg("intensity"),
          py::arg("handle"), py::arg("backend"));
    
    m.def("pixelate", &pixelate, "Pixelate effect",
          py::arg("input"), py::arg("output"), py::arg("scratch"),
          py::arg("pixelation_pct"),
          py::arg("handle"), py::arg("backend"));
}
