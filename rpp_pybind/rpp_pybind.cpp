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
    TensorData data;
    data.ptr = tensor.data_ptr();
    
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
    desc.dataType = data.dtype;
    desc.layout = data.layout;
    desc.numDims = data.shape.size();
    
    if (desc.numDims == 4) {
        desc.n = data.shape[0];
        desc.c = data.shape[1]; 
        desc.h = data.shape[2];
        desc.w = data.shape[3];
        
        // Calculate strides in bytes
        size_t element_size = (data.dtype == RpptDataType::U8 || data.dtype == RpptDataType::I8) ? 1 :
                            (data.dtype == RpptDataType::F16) ? 2 : 4;
        
        desc.strides.nStride = data.strides[0] * element_size;
        desc.strides.cStride = data.strides[1] * element_size;
        desc.strides.hStride = data.strides[2] * element_size;
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
    
    rppt_brightness(input_data.ptr, &src_desc,
                   output_data.ptr, &dst_desc,
                   const_cast<float*>(alpha.data()),
                   const_cast<float*>(beta.data()),
                   roi.data(), RpptRoiType::XYWH,
                   handle, static_cast<RppBackend>(backend));
}

// 2. Gamma Correction (Color)
void gamma_correction(const torch::Tensor& input_tensor,
                     torch::Tensor& output_tensor,
                     const std::vector<float>& gamma,
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
    
    rppt_gamma_correction(input_data.ptr, &src_desc,
                         output_data.ptr, &dst_desc,
                         const_cast<float*>(gamma.data()),
                         roi.data(), RpptRoiType::XYWH,
                         handle, static_cast<RppBackend>(backend));
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
PYBIND11_MODULE(rpp_pybind, m) {
    m.doc() = "PyRPP - Python bindings for AMD ROCm Performance Primitives";

    // Version info
    m.attr("__version__") = "1.0.0";
    
    // Enums wrapped properly as py::enum_
    py::enum_<RppBackend>(m, "RppBackendInternal")
        .value("HOST", RPP_HOST_BACKEND)
        .value("HIP", RPP_HIP_BACKEND);
    
    py::enum_<RppStatus>(m, "RppStatusInternal")
        .value("SUCCESS", RPP_SUCCESS)
        .value("ERROR", RPP_ERROR);
    
    py::enum_<RpptDataType>(m, "RpptDataTypeInternal")
        .value("U8", RpptDataType::U8)
        .value("F16", RpptDataType::F16)
        .value("F32", RpptDataType::F32)
        .value("I8", RpptDataType::I8)
        .value("I16", RpptDataType::I16);
    
    py::enum_<RpptLayout>(m, "RpptLayoutInternal")
        .value("NCHW", RpptLayout::NCHW)
        .value("NHWC", RpptLayout::NHWC)
        .value("NCDHW", RpptLayout::NCDHW)
        .value("NDHWC", RpptLayout::NDHWC)
        .value("NHW", RpptLayout::NHW)
        .value("NFT", RpptLayout::NFT)
        .value("NTF", RpptLayout::NTF);
    
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
    m.def("rppCreate", [](int batch_size, int backend) {
        rppHandle_t handle;
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
