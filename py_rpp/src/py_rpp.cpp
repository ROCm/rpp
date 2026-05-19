// _py_rpp — pybind11 extension module for RPP (ROCm Performance Primitives)
//
// This file is the thin C++ entry point.  All business logic lives in ops/*.cpp.
// The ergonomic Python API (scalar broadcasting, layout detection, auto-backend)
// lives in py_rpp/fn.py; this module is the low-level layer that fn.py calls.

#include "ops/color.hpp"
#include "ops/geometric.hpp"
#include "ops/filter.hpp"
#include "ops/morphological.hpp"
#include "ops/bitwise.hpp"
#include "ops/statistical.hpp"
#include "ops/effects.hpp"

namespace py = pybind11;

// ─── Handle lifecycle (power-user API) ───────────────────────────────────────
// Returns an opaque handle as uintptr_t so Python can hold and reuse it.
// Normal callers should use fn.py; these are for advanced use cases.
static uintptr_t rpp_create(size_t batch_size, const std::string& backend) {
    rppHandle_t h;
    Backend bk = parse_backend(backend);
    uint32_t threads = (bk == Backend::CPU) ? 4 : 0;
    check_rpp_create(
        rppCreate(&h, batch_size, threads, nullptr, to_rpp_backend(bk)),
        "rppCreate");
    return reinterpret_cast<uintptr_t>(h);
}

static void rpp_destroy(uintptr_t handle, const std::string& backend) {
    if (!handle) throw std::invalid_argument("rpp_destroy: null handle");
    rppDestroy(reinterpret_cast<rppHandle_t>(handle), to_rpp_backend(parse_backend(backend)));
}

// ─── Module ──────────────────────────────────────────────────────────────────
PYBIND11_MODULE(_py_rpp, m) {
    m.doc() = R"doc(
_py_rpp — Low-level Python bindings for RPP (ROCm Performance Primitives).

Supports both HIP (GPU) and CPU backends.  All functions accept C-contiguous
numpy uint8 arrays in NHWC layout (N, H, W, C).

For the ergonomic API with scalar broadcasting, layout auto-detection, and
auto-backend selection, use the high-level ``py_rpp.fn`` module instead.
)doc";

    // ── types submodule ───────────────────────────────────────────────────────
    auto types = m.def_submodule("types", "RPP type enumerations");

    py::enum_<RppBackend>(types, "RppBackend")
        .value("HOST", RPP_HOST_BACKEND)
        .value("HIP",  RPP_HIP_BACKEND)
        .export_values();

    py::enum_<RpptLayout>(types, "RpptLayout")
        .value("NCHW", RpptLayout::NCHW)
        .value("NHWC", RpptLayout::NHWC)
        .export_values();

    py::enum_<RpptDataType>(types, "RpptDataType")
        .value("U8",  RpptDataType::U8)
        .value("F32", RpptDataType::F32)
        .value("F16", RpptDataType::F16)
        .value("I8",  RpptDataType::I8)
        .export_values();

    py::enum_<RpptInterpolationType>(types, "RpptInterpolationType")
        .value("NEAREST_NEIGHBOR", RpptInterpolationType::NEAREST_NEIGHBOR)
        .value("BILINEAR",         RpptInterpolationType::BILINEAR)
        .value("BICUBIC",          RpptInterpolationType::BICUBIC)
        .export_values();

    // ── handle lifecycle (advanced use) ──────────────────────────────────────
    m.def("rpp_create",  &rpp_create,
          py::arg("batch_size"), py::arg("backend") = "hip",
          "Create an RPP handle. Returns opaque uintptr_t. Use rpp_destroy when done.");

    m.def("rpp_destroy", &rpp_destroy,
          py::arg("handle"), py::arg("backend") = "hip",
          "Destroy an RPP handle created by rpp_create.");

    // ── color augmentations ───────────────────────────────────────────────────
    m.def("brightness", &ops::brightness,
          py::arg("src"), py::arg("alpha"), py::arg("beta"),
          py::arg("backend") = "hip",
          "clamp(alpha * src + beta, 0, 255).  alpha/beta: float32 (N,).");

    m.def("gamma_correction", &ops::gamma_correction,
          py::arg("src"), py::arg("gamma"),
          py::arg("backend") = "hip",
          "Per-image gamma correction: src ^ gamma.");

    m.def("hue", &ops::hue,
          py::arg("src"), py::arg("hue_shift"),
          py::arg("backend") = "hip",
          "Hue rotation in degrees [-180, 180] per image (3-channel only).");

    m.def("saturation", &ops::saturation,
          py::arg("src"), py::arg("factor"),
          py::arg("backend") = "hip",
          "Saturation scaling per image (1.0 = identity).");

    m.def("contrast", &ops::contrast,
          py::arg("src"), py::arg("factor"), py::arg("center"),
          py::arg("backend") = "hip",
          "clamp(factor * (src - center) + center, 0, 255).");

    m.def("exposure", &ops::exposure,
          py::arg("src"), py::arg("factor"),
          py::arg("backend") = "hip",
          "Exposure shift per image.");

    m.def("blend", &ops::blend,
          py::arg("src1"), py::arg("src2"), py::arg("alpha"),
          py::arg("backend") = "hip",
          "dst = alpha * src1 + (1 - alpha) * src2.");

    m.def("color_twist", &ops::color_twist,
          py::arg("src"),
          py::arg("brightness"), py::arg("contrast"),
          py::arg("hue_shift"),  py::arg("saturation"),
          py::arg("backend") = "hip",
          "Combined brightness + contrast + hue + saturation adjustment.");

    m.def("histogram_equalize", &ops::histogram_equalize,
          py::arg("src"), py::arg("backend") = "hip",
          "Per-channel histogram equalization.");

    // ── geometric augmentations ───────────────────────────────────────────────
    m.def("flip", &ops::flip,
          py::arg("src"), py::arg("horizontal"), py::arg("vertical"),
          py::arg("backend") = "hip",
          "flip: horizontal/vertical are uint32 (N,) arrays of 0 or 1.");

    m.def("crop", &ops::crop,
          py::arg("src"), py::arg("x"), py::arg("y"),
          py::arg("out_h"), py::arg("out_w"),
          py::arg("backend") = "hip",
          "Crop a uniform (out_h, out_w) region starting at per-image (x, y).");

    m.def("rotate", &ops::rotate,
          py::arg("src"), py::arg("angle"),
          py::arg("backend") = "hip",
          "Rotate by angle (degrees) per image, bilinear interpolation.");

    m.def("resize", &ops::resize,
          py::arg("src"), py::arg("out_h"), py::arg("out_w"),
          py::arg("backend") = "hip",
          "Resize all images to (out_h, out_w), bilinear interpolation.");

    // ── filter augmentations ──────────────────────────────────────────────────
    m.def("box_filter", &ops::box_filter,
          py::arg("src"), py::arg("kernel_size") = 3,
          py::arg("backend") = "hip",
          "Uniform (mean) box filter.  kernel_size must be odd.");

    m.def("gaussian_filter", &ops::gaussian_filter,
          py::arg("src"), py::arg("std_dev"), py::arg("kernel_size") = 3,
          py::arg("backend") = "hip",
          "Gaussian filter with per-image std_dev.  kernel_size must be odd.");

    m.def("median_filter", &ops::median_filter,
          py::arg("src"), py::arg("kernel_size") = 3,
          py::arg("backend") = "hip",
          "Median filter (edge-preserving).  kernel_size must be odd.");

    // ── morphological operations ──────────────────────────────────────────────
    m.def("erode", &ops::erode,
          py::arg("src"), py::arg("kernel_size") = 3,
          py::arg("backend") = "hip",
          "Morphological erosion (min filter).  kernel_size must be odd.");

    m.def("dilate", &ops::dilate,
          py::arg("src"), py::arg("kernel_size") = 3,
          py::arg("backend") = "hip",
          "Morphological dilation (max filter).  kernel_size must be odd.");

    // ── bitwise operations ────────────────────────────────────────────────────
    m.def("bitwise_and", &ops::bitwise_and,
          py::arg("src1"), py::arg("src2"), py::arg("backend") = "hip");
    m.def("bitwise_or",  &ops::bitwise_or,
          py::arg("src1"), py::arg("src2"), py::arg("backend") = "hip");
    m.def("bitwise_xor", &ops::bitwise_xor,
          py::arg("src1"), py::arg("src2"), py::arg("backend") = "hip");
    m.def("bitwise_not", &ops::bitwise_not,
          py::arg("src"), py::arg("backend") = "hip");

    // ── statistical operations ────────────────────────────────────────────────
    m.def("tensor_mean", &ops::tensor_mean,
          py::arg("src"), py::arg("backend") = "hip",
          "Per-image per-channel mean.  Returns float32 (N, C+1).");

    m.def("tensor_min",  &ops::tensor_min,
          py::arg("src"), py::arg("backend") = "hip",
          "Per-image per-channel min.  Returns uint8 (N, C+1).");

    m.def("tensor_max",  &ops::tensor_max,
          py::arg("src"), py::arg("backend") = "hip",
          "Per-image per-channel max.  Returns uint8 (N, C+1).");

    m.def("threshold", &ops::threshold,
          py::arg("src"), py::arg("min_val"), py::arg("max_val"),
          py::arg("backend") = "hip",
          "Binary mask: 255 where min_val <= pixel <= max_val, else 0.");

    // ── effects / augmentations ───────────────────────────────────────────────
    m.def("gaussian_noise", &ops::gaussian_noise,
          py::arg("src"), py::arg("mean"), py::arg("std_dev"),
          py::arg("seed") = 0, py::arg("backend") = "hip",
          "Additive Gaussian noise with per-image mean and std_dev.");

    m.def("salt_and_pepper_noise", &ops::salt_and_pepper_noise,
          py::arg("src"),
          py::arg("noise_prob"), py::arg("salt_prob"),
          py::arg("salt_value"), py::arg("pepper_value"),
          py::arg("seed") = 0, py::arg("backend") = "hip",
          "Salt-and-pepper noise with per-image parameters.");

    m.def("vignette", &ops::vignette,
          py::arg("src"), py::arg("intensity"),
          py::arg("backend") = "hip",
          "Vignette effect: darkens image edges.  intensity per image [0, 1].");

    m.def("pixelate", &ops::pixelate,
          py::arg("src"), py::arg("pixelation_percentage") = 50.0f,
          py::arg("backend") = "hip",
          "Pixelate effect.  pixelation_percentage in [0, 100].");
}
