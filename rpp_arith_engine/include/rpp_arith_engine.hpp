#pragma once

/*
 * rppArithmeticEngine — a high-level GPU tensor engine wrapping RPP.
 *
 * Manages its own HIP device buffer and RPP handle.
 * Tensors are F32, NHWC layout.
 *
 * Arithmetic operators  — scalar and tensor-tensor (returns new engine)
 *   eng * 2.0f      →  rppt_fused_multiply_add_scalar(alpha=2, beta=0)
 *   eng + 0.5f      →  rppt_fused_multiply_add_scalar(alpha=1, beta=0.5)
 *   engA + engB     →  rppt_tensor_add_tensor
 *   engA - engB     →  rppt_tensor_subtract_tensor
 *   engA * engB     →  rppt_tensor_multiply_tensor
 *   engA / engB     →  rppt_tensor_divide_tensor
 *
 * Color augmentations   — in-place, chainable
 *   .brightness(alpha, beta)
 *   .gammaCorrection(gamma)
 *   .colorTwist(brightness, contrast, hue, saturation)
 *   .histogramEqualization()
 *   .exposure(factor)
 *   .contrast(factor, center)
 *   .hue(shift)
 *   .saturation(factor)
 */

#include <hip/hip_runtime.h>
#include <rpp/rpp.h>
#include <rpp/rppdefs.h>
#include <rpp/rppt_tensor_arithmetic_operations.h>
#include <rpp/rppt_tensor_color_augmentations.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

class rppArithmeticEngine
{
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // Allocates a GPU float32 tensor of shape (N, H, W, C) and creates an RPP handle.
    rppArithmeticEngine(uint32_t n, uint32_t h, uint32_t w, uint32_t c);
    ~rppArithmeticEngine();

    // Move-only (no copies — each instance owns its device buffer)
    rppArithmeticEngine(const rppArithmeticEngine&)            = delete;
    rppArithmeticEngine& operator=(const rppArithmeticEngine&) = delete;
    rppArithmeticEngine(rppArithmeticEngine&&) noexcept;
    rppArithmeticEngine& operator=(rppArithmeticEngine&&) noexcept;

    // ── Host I/O ──────────────────────────────────────────────────────────────

    rppArithmeticEngine& upload(const float* hostPtr);   // host → device
    void                 download(float* hostPtr) const; // device → host

    uint32_t n() const { return N_; }
    uint32_t h() const { return H_; }
    uint32_t w() const { return W_; }
    uint32_t c() const { return C_; }
    size_t   numel() const { return (size_t)N_ * H_ * W_ * C_; }

    // ── Scalar arithmetic — returns a new engine ──────────────────────────────
    //   Backed by rppt_fused_multiply_add_scalar(mul, add) on all batch images.

    rppArithmeticEngine operator+(float scalar) const;  // x + s
    rppArithmeticEngine operator-(float scalar) const;  // x - s
    rppArithmeticEngine operator*(float scalar) const;  // x * s
    rppArithmeticEngine operator/(float scalar) const;  // x / s

    // ── Tensor-tensor arithmetic — same shape required, returns a new engine ──

    rppArithmeticEngine operator+(const rppArithmeticEngine& other) const;
    rppArithmeticEngine operator-(const rppArithmeticEngine& other) const;
    rppArithmeticEngine operator*(const rppArithmeticEngine& other) const;
    rppArithmeticEngine operator/(const rppArithmeticEngine& other) const;

    // ── Color augmentations — in-place, return *this for chaining ────────────

    // output = clamp(alpha * input + beta)   [alpha per image, beta per image]
    rppArithmeticEngine& brightness(const std::vector<float>& alpha,
                                    const std::vector<float>& beta);

    // output = input ^ gamma                 [gamma per image]
    rppArithmeticEngine& gammaCorrection(const std::vector<float>& gamma);

    // Combined brightness + contrast + hue + saturation  [per image]
    rppArithmeticEngine& colorTwist(const std::vector<float>& brightness,
                                    const std::vector<float>& contrast,
                                    const std::vector<float>& hue,
                                    const std::vector<float>& saturation);

    // Per-channel histogram equalization
    rppArithmeticEngine& histogramEqualization();

    // Exposure adjustment  [factor per image]
    rppArithmeticEngine& exposure(const std::vector<float>& factor);

    // Contrast adjustment  [factor and center per image]
    rppArithmeticEngine& contrast(const std::vector<float>& factor,
                                  const std::vector<float>& center);

    // Hue shift           [degrees per image]
    rppArithmeticEngine& hue(const std::vector<float>& shift);

    // Saturation scaling  [factor per image]
    rppArithmeticEngine& saturation(const std::vector<float>& factor);

private:
    uint32_t    N_{0}, H_{0}, W_{0}, C_{0};
    float*      devBuf_{nullptr};   // device float32 buffer
    rppHandle_t handle_{nullptr};
    RpptDesc    desc_{};
    RpptGenericDesc genericDesc_{};
    std::vector<RpptROI>   rois_;
    std::vector<uint32_t>  roiTensor_; // flat ROI for generic ops (per-image XYZWHD)

    rppArithmeticEngine scalarFma(float alpha, float beta) const;

    void buildDescs();
    void buildRois();

    // Upload a host vector to a freshly allocated device buffer, return ptr.
    float* uploadParam(const std::vector<float>& v) const;
    void   freeParam(float* p) const;

    void checkShape(const rppArithmeticEngine& other) const;

    static void hipCheck(hipError_t e, const char* what);
    static void rppCheck(RppStatus  s, const char* what);
};
