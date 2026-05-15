#include "rpp_arith_engine.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

// ─── Error helpers ────────────────────────────────────────────────────────────

void rppArithmeticEngine::hipCheck(hipError_t e, const char* what)
{
    if (e != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

void rppArithmeticEngine::rppCheck(RppStatus s, const char* what)
{
    if (s != RPP_SUCCESS)
        throw std::runtime_error(std::string(what) +
                                 " failed (RppStatus=" + std::to_string(s) + ")");
}

// ─── Descriptor builders ──────────────────────────────────────────────────────

void rppArithmeticEngine::buildDescs()
{
    // Standard NHWC descriptor (color augmentation ops)
    desc_ = {};
    desc_.numDims       = 4;
    desc_.offsetInBytes = 0;
    desc_.dataType      = RpptDataType::F32;
    desc_.layout        = RpptLayout::NHWC;
    desc_.n = N_; desc_.h = H_; desc_.w = W_; desc_.c = C_;
    desc_.strides.nStride = C_ * W_ * H_;
    desc_.strides.hStride = C_ * W_;
    desc_.strides.wStride = C_;
    desc_.strides.cStride = 1;

    // Generic descriptor (arithmetic ops)
    genericDesc_ = {};
    genericDesc_.numDims       = 4;
    genericDesc_.offsetInBytes = 0;
    genericDesc_.dataType      = RpptDataType::F32;
    genericDesc_.layout        = RpptLayout::NHWC;
    genericDesc_.dims[0]    = N_; genericDesc_.dims[1]    = H_;
    genericDesc_.dims[2]    = W_; genericDesc_.dims[3]    = C_;
    genericDesc_.strides[0] = C_ * W_ * H_; genericDesc_.strides[1] = C_ * W_;
    genericDesc_.strides[2] = C_;           genericDesc_.strides[3] = 1;
}

void rppArithmeticEngine::buildRois()
{
    // Full-image XYWH ROIs for color ops
    rois_.resize(N_);
    for (uint32_t i = 0; i < N_; ++i) {
        rois_[i].xywhROI.xy.x      = 0;
        rois_[i].xywhROI.xy.y      = 0;
        rois_[i].xywhROI.roiWidth  = static_cast<int>(W_);
        rois_[i].xywhROI.roiHeight = static_cast<int>(H_);
    }

    // Flat uint32 ROI tensor for generic arithmetic ops:
    // layout per image: [x, y, z, w, h, d] — for 2-D images z=0, d=1
    roiTensor_.assign(N_ * 6, 0);
    for (uint32_t i = 0; i < N_; ++i) {
        uint32_t* r = roiTensor_.data() + i * 6;
        r[0] = 0; r[1] = 0; r[2] = 0;          // x, y, z
        r[3] = W_; r[4] = H_; r[5] = 1;         // width, height, depth
    }
}

// ─── Param helpers ────────────────────────────────────────────────────────────

float* rppArithmeticEngine::uploadParam(const std::vector<float>& v) const
{
    float* d = nullptr;
    hipCheck(hipMalloc(&d, v.size() * sizeof(float)), "hipMalloc param");
    hipCheck(hipMemcpy(d, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice),
             "hipMemcpy param H2D");
    return d;
}

void rppArithmeticEngine::freeParam(float* p) const
{
    if (p) (void)hipFree(p);
}

void rppArithmeticEngine::checkShape(const rppArithmeticEngine& o) const
{
    if (N_ != o.N_ || H_ != o.H_ || W_ != o.W_ || C_ != o.C_)
        throw std::invalid_argument(
            "rppArithmeticEngine: tensor shapes do not match for binary op");
}

// ─── Construction / Destruction ───────────────────────────────────────────────

rppArithmeticEngine::rppArithmeticEngine(uint32_t n, uint32_t h, uint32_t w, uint32_t c)
    : N_(n), H_(h), W_(w), C_(c)
{
    hipCheck(hipMalloc(&devBuf_, numel() * sizeof(float)), "hipMalloc tensor");

    rppStatus_t st = rppCreate(&handle_, n, 0, nullptr, RPP_HIP_BACKEND);
    if (st != rppStatusSuccess)
        throw std::runtime_error("rppCreate failed: " + std::to_string(st));

    buildDescs();
    buildRois();
}

rppArithmeticEngine::~rppArithmeticEngine()
{
    if (devBuf_)  (void)hipFree(devBuf_);
    if (handle_)  rppDestroy(handle_, RPP_HIP_BACKEND);
}

rppArithmeticEngine::rppArithmeticEngine(rppArithmeticEngine&& o) noexcept
    : N_(o.N_), H_(o.H_), W_(o.W_), C_(o.C_),
      devBuf_(o.devBuf_), handle_(o.handle_),
      desc_(o.desc_), genericDesc_(o.genericDesc_),
      rois_(std::move(o.rois_)), roiTensor_(std::move(o.roiTensor_))
{
    o.devBuf_ = nullptr;
    o.handle_ = nullptr;
}

rppArithmeticEngine& rppArithmeticEngine::operator=(rppArithmeticEngine&& o) noexcept
{
    if (this != &o) {
        if (devBuf_) (void)hipFree(devBuf_);
        if (handle_) rppDestroy(handle_, RPP_HIP_BACKEND);
        N_ = o.N_; H_ = o.H_; W_ = o.W_; C_ = o.C_;
        devBuf_      = o.devBuf_;
        handle_      = o.handle_;
        desc_        = o.desc_;
        genericDesc_ = o.genericDesc_;
        rois_        = std::move(o.rois_);
        roiTensor_   = std::move(o.roiTensor_);
        o.devBuf_    = nullptr;
        o.handle_    = nullptr;
    }
    return *this;
}

// ─── Host I/O ─────────────────────────────────────────────────────────────────

rppArithmeticEngine& rppArithmeticEngine::upload(const float* hostPtr)
{
    hipCheck(hipMemcpy(devBuf_, hostPtr, numel() * sizeof(float), hipMemcpyHostToDevice),
             "upload H2D");
    return *this;
}

void rppArithmeticEngine::download(float* hostPtr) const
{
    hipCheck(hipMemcpy(hostPtr, devBuf_, numel() * sizeof(float), hipMemcpyDeviceToHost),
             "download D2H");
}

// ─── Scalar arithmetic ────────────────────────────────────────────────────────
// rppt_brightness does output = clamp(alpha*input + beta) for NHWC F32.
// All scalar ops map to fma(alpha, beta):
//   x + s  →  brightness(alpha=1, beta=s)
//   x - s  →  brightness(alpha=1, beta=-s)
//   x * s  →  brightness(alpha=s, beta=0)
//   x / s  →  brightness(alpha=1/s, beta=0)

rppArithmeticEngine rppArithmeticEngine::scalarFma(float alpha, float beta) const
{
    rppArithmeticEngine out(N_, H_, W_, C_);
    // copy src → out on device, then apply brightness in-place on out
    hipCheck(hipMemcpy(out.devBuf_, devBuf_, numel()*sizeof(float),
                       hipMemcpyDeviceToDevice), "D2D copy for scalar op");

    // rppt_brightness for F32 applies beta * (1/255) internally,
    // so scale beta to [0,255] space to get the actual additive offset.
    std::vector<float> aVec(N_, alpha), bVec(N_, beta * 255.0f);
    float* aD   = out.uploadParam(aVec);
    float* bD   = out.uploadParam(bVec);
    float* roiD = nullptr;
    hipCheck(hipMalloc(&roiD, rois_.size() * sizeof(RpptROI)), "hipMalloc roi");
    hipCheck(hipMemcpy(roiD, rois_.data(), rois_.size()*sizeof(RpptROI),
                       hipMemcpyHostToDevice), "roi H2D");
    rppCheck(rppt_brightness(out.devBuf_, &out.desc_, out.devBuf_, &out.desc_,
                             aD, bD,
                             reinterpret_cast<RpptROIPtr>(roiD),
                             RpptRoiType::XYWH, out.handle_, RPP_HIP_BACKEND),
             "rppt_brightness (scalar op)");
    hipCheck(hipDeviceSynchronize(), "sync scalar op");
    out.freeParam(aD); out.freeParam(bD); (void)hipFree(roiD);
    return out;
}

rppArithmeticEngine rppArithmeticEngine::operator+(float s) const { return scalarFma(1.0f,      s); }
rppArithmeticEngine rppArithmeticEngine::operator-(float s) const { return scalarFma(1.0f,     -s); }
rppArithmeticEngine rppArithmeticEngine::operator*(float s) const { return scalarFma(s,       0.0f); }
rppArithmeticEngine rppArithmeticEngine::operator/(float s) const
{
    if (s == 0.0f) throw std::invalid_argument("rppArithmeticEngine: division by zero");
    return scalarFma(1.0f / s, 0.0f);
}

// ─── Tensor-tensor arithmetic ─────────────────────────────────────────────────

rppArithmeticEngine rppArithmeticEngine::operator+(const rppArithmeticEngine& o) const
{
    checkShape(o);
    rppArithmeticEngine out(N_, H_, W_, C_);
    uint32_t* roi1D = nullptr, *roi2D = nullptr;
    hipCheck(hipMalloc(&roi1D, roiTensor_.size() * sizeof(uint32_t)), "hipMalloc roi1");
    hipCheck(hipMalloc(&roi2D, roiTensor_.size() * sizeof(uint32_t)), "hipMalloc roi2");
    hipCheck(hipMemcpy(roi1D, roiTensor_.data(),
                       roiTensor_.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "roi1 H2D");
    hipCheck(hipMemcpy(roi2D, roiTensor_.data(),
                       roiTensor_.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "roi2 H2D");
    rppCheck(rppt_tensor_add_tensor(
                 devBuf_, o.devBuf_,
                 const_cast<RpptGenericDescPtr>(&genericDesc_), const_cast<RpptGenericDescPtr>(&o.genericDesc_),
                 out.devBuf_, &out.genericDesc_,
                 RpptBroadcastMode::RPP_BROADCAST_DISABLE,
                 roi1D, roi2D, handle_, RPP_HIP_BACKEND),
             "rppt_tensor_add_tensor");
    hipCheck(hipDeviceSynchronize(), "sync tensor+");
    (void)hipFree(roi1D); (void)hipFree(roi2D);
    return out;
}

rppArithmeticEngine rppArithmeticEngine::operator-(const rppArithmeticEngine& o) const
{
    checkShape(o);
    rppArithmeticEngine out(N_, H_, W_, C_);
    uint32_t* roi1D = nullptr, *roi2D = nullptr;
    hipCheck(hipMalloc(&roi1D, roiTensor_.size() * sizeof(uint32_t)), "hipMalloc roi1");
    hipCheck(hipMalloc(&roi2D, roiTensor_.size() * sizeof(uint32_t)), "hipMalloc roi2");
    hipCheck(hipMemcpy(roi1D, roiTensor_.data(),
                       roiTensor_.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "roi1 H2D");
    hipCheck(hipMemcpy(roi2D, roiTensor_.data(),
                       roiTensor_.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "roi2 H2D");
    rppCheck(rppt_tensor_subtract_tensor(
                 devBuf_, o.devBuf_,
                 const_cast<RpptGenericDescPtr>(&genericDesc_), const_cast<RpptGenericDescPtr>(&o.genericDesc_),
                 out.devBuf_, &out.genericDesc_,
                 RpptBroadcastMode::RPP_BROADCAST_DISABLE,
                 roi1D, roi2D, handle_, RPP_HIP_BACKEND),
             "rppt_tensor_subtract_tensor");
    hipCheck(hipDeviceSynchronize(), "sync tensor-");
    (void)hipFree(roi1D); (void)hipFree(roi2D);
    return out;
}

rppArithmeticEngine rppArithmeticEngine::operator*(const rppArithmeticEngine& o) const
{
    checkShape(o);
    rppArithmeticEngine out(N_, H_, W_, C_);
    uint32_t* roi1D = nullptr, *roi2D = nullptr;
    hipCheck(hipMalloc(&roi1D, roiTensor_.size() * sizeof(uint32_t)), "hipMalloc roi1");
    hipCheck(hipMalloc(&roi2D, roiTensor_.size() * sizeof(uint32_t)), "hipMalloc roi2");
    hipCheck(hipMemcpy(roi1D, roiTensor_.data(),
                       roiTensor_.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "roi1 H2D");
    hipCheck(hipMemcpy(roi2D, roiTensor_.data(),
                       roiTensor_.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "roi2 H2D");
    rppCheck(rppt_tensor_multiply_tensor(
                 devBuf_, o.devBuf_,
                 const_cast<RpptGenericDescPtr>(&genericDesc_), const_cast<RpptGenericDescPtr>(&o.genericDesc_),
                 out.devBuf_, &out.genericDesc_,
                 RpptBroadcastMode::RPP_BROADCAST_DISABLE,
                 roi1D, roi2D, handle_, RPP_HIP_BACKEND),
             "rppt_tensor_multiply_tensor");
    hipCheck(hipDeviceSynchronize(), "sync tensor*");
    (void)hipFree(roi1D); (void)hipFree(roi2D);
    return out;
}

rppArithmeticEngine rppArithmeticEngine::operator/(const rppArithmeticEngine& o) const
{
    checkShape(o);
    rppArithmeticEngine out(N_, H_, W_, C_);
    uint32_t* roi1D = nullptr, *roi2D = nullptr;
    hipCheck(hipMalloc(&roi1D, roiTensor_.size() * sizeof(uint32_t)), "hipMalloc roi1");
    hipCheck(hipMalloc(&roi2D, roiTensor_.size() * sizeof(uint32_t)), "hipMalloc roi2");
    hipCheck(hipMemcpy(roi1D, roiTensor_.data(),
                       roiTensor_.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "roi1 H2D");
    hipCheck(hipMemcpy(roi2D, roiTensor_.data(),
                       roiTensor_.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "roi2 H2D");
    rppCheck(rppt_tensor_divide_tensor(
                 devBuf_, o.devBuf_,
                 const_cast<RpptGenericDescPtr>(&genericDesc_), const_cast<RpptGenericDescPtr>(&o.genericDesc_),
                 out.devBuf_, &out.genericDesc_,
                 RpptBroadcastMode::RPP_BROADCAST_DISABLE,
                 roi1D, roi2D, handle_, RPP_HIP_BACKEND),
             "rppt_tensor_divide_tensor");
    hipCheck(hipDeviceSynchronize(), "sync tensor/");
    (void)hipFree(roi1D); (void)hipFree(roi2D);
    return out;
}

// ─── Color augmentations (in-place) ──────────────────────────────────────────

rppArithmeticEngine& rppArithmeticEngine::brightness(const std::vector<float>& alpha,
                                                      const std::vector<float>& beta)
{
    float* aD = uploadParam(alpha);
    float* bD = uploadParam(beta);
    float* roiD = nullptr;
    hipCheck(hipMalloc(&roiD, rois_.size() * sizeof(RpptROI)), "hipMalloc roi");
    hipCheck(hipMemcpy(roiD, rois_.data(), rois_.size() * sizeof(RpptROI),
                       hipMemcpyHostToDevice), "roi H2D");
    rppCheck(rppt_brightness(devBuf_, &desc_, devBuf_, &desc_,
                             aD, bD,
                             reinterpret_cast<RpptROIPtr>(roiD),
                             RpptRoiType::XYWH, handle_, RPP_HIP_BACKEND),
             "rppt_brightness");
    hipCheck(hipDeviceSynchronize(), "sync brightness");
    freeParam(aD); freeParam(bD); (void)hipFree(roiD);
    return *this;
}

rppArithmeticEngine& rppArithmeticEngine::gammaCorrection(const std::vector<float>& gamma)
{
    float* gD = uploadParam(gamma);
    float* roiD = nullptr;
    hipCheck(hipMalloc(&roiD, rois_.size() * sizeof(RpptROI)), "hipMalloc roi");
    hipCheck(hipMemcpy(roiD, rois_.data(), rois_.size() * sizeof(RpptROI),
                       hipMemcpyHostToDevice), "roi H2D");
    rppCheck(rppt_gamma_correction(devBuf_, &desc_, devBuf_, &desc_,
                                   gD,
                                   reinterpret_cast<RpptROIPtr>(roiD),
                                   RpptRoiType::XYWH, handle_, RPP_HIP_BACKEND),
             "rppt_gamma_correction");
    hipCheck(hipDeviceSynchronize(), "sync gammaCorrection");
    freeParam(gD); (void)hipFree(roiD);
    return *this;
}

rppArithmeticEngine& rppArithmeticEngine::colorTwist(const std::vector<float>& bright,
                                                      const std::vector<float>& contrast,
                                                      const std::vector<float>& hue_,
                                                      const std::vector<float>& sat)
{
    float* bD  = uploadParam(bright);
    float* cD  = uploadParam(contrast);
    float* hD  = uploadParam(hue_);
    float* sD  = uploadParam(sat);
    float* roiD = nullptr;
    hipCheck(hipMalloc(&roiD, rois_.size() * sizeof(RpptROI)), "hipMalloc roi");
    hipCheck(hipMemcpy(roiD, rois_.data(), rois_.size() * sizeof(RpptROI),
                       hipMemcpyHostToDevice), "roi H2D");
    rppCheck(rppt_color_twist(devBuf_, &desc_, devBuf_, &desc_,
                              bD, cD, hD, sD,
                              reinterpret_cast<RpptROIPtr>(roiD),
                              RpptRoiType::XYWH, handle_, RPP_HIP_BACKEND),
             "rppt_color_twist");
    hipCheck(hipDeviceSynchronize(), "sync colorTwist");
    freeParam(bD); freeParam(cD); freeParam(hD); freeParam(sD); (void)hipFree(roiD);
    return *this;
}

rppArithmeticEngine& rppArithmeticEngine::histogramEqualization()
{
    float* roiD = nullptr;
    hipCheck(hipMalloc(&roiD, rois_.size() * sizeof(RpptROI)), "hipMalloc roi");
    hipCheck(hipMemcpy(roiD, rois_.data(), rois_.size() * sizeof(RpptROI),
                       hipMemcpyHostToDevice), "roi H2D");
    rppCheck(rppt_histogram_equalize(devBuf_, &desc_, devBuf_, &desc_,
                                     reinterpret_cast<RpptROIPtr>(roiD),
                                     RpptRoiType::XYWH, handle_, RPP_HIP_BACKEND),
             "rppt_histogram_equalize");
    hipCheck(hipDeviceSynchronize(), "sync histogramEqualization");
    (void)hipFree(roiD);
    return *this;
}

rppArithmeticEngine& rppArithmeticEngine::exposure(const std::vector<float>& factor)
{
    float* fD = uploadParam(factor);
    float* roiD = nullptr;
    hipCheck(hipMalloc(&roiD, rois_.size() * sizeof(RpptROI)), "hipMalloc roi");
    hipCheck(hipMemcpy(roiD, rois_.data(), rois_.size() * sizeof(RpptROI),
                       hipMemcpyHostToDevice), "roi H2D");
    rppCheck(rppt_exposure(devBuf_, &desc_, devBuf_, &desc_,
                           fD,
                           reinterpret_cast<RpptROIPtr>(roiD),
                           RpptRoiType::XYWH, handle_, RPP_HIP_BACKEND),
             "rppt_exposure");
    hipCheck(hipDeviceSynchronize(), "sync exposure");
    freeParam(fD); (void)hipFree(roiD);
    return *this;
}

rppArithmeticEngine& rppArithmeticEngine::contrast(const std::vector<float>& factor,
                                                    const std::vector<float>& center)
{
    float* fD = uploadParam(factor);
    float* cD = uploadParam(center);
    float* roiD = nullptr;
    hipCheck(hipMalloc(&roiD, rois_.size() * sizeof(RpptROI)), "hipMalloc roi");
    hipCheck(hipMemcpy(roiD, rois_.data(), rois_.size() * sizeof(RpptROI),
                       hipMemcpyHostToDevice), "roi H2D");
    rppCheck(rppt_contrast(devBuf_, &desc_, devBuf_, &desc_,
                           fD, cD,
                           reinterpret_cast<RpptROIPtr>(roiD),
                           RpptRoiType::XYWH, handle_, RPP_HIP_BACKEND),
             "rppt_contrast");
    hipCheck(hipDeviceSynchronize(), "sync contrast");
    freeParam(fD); freeParam(cD); (void)hipFree(roiD);
    return *this;
}

rppArithmeticEngine& rppArithmeticEngine::hue(const std::vector<float>& shift)
{
    float* hD = uploadParam(shift);
    float* roiD = nullptr;
    hipCheck(hipMalloc(&roiD, rois_.size() * sizeof(RpptROI)), "hipMalloc roi");
    hipCheck(hipMemcpy(roiD, rois_.data(), rois_.size() * sizeof(RpptROI),
                       hipMemcpyHostToDevice), "roi H2D");
    rppCheck(rppt_hue(devBuf_, &desc_, devBuf_, &desc_,
                      hD,
                      reinterpret_cast<RpptROIPtr>(roiD),
                      RpptRoiType::XYWH, handle_, RPP_HIP_BACKEND),
             "rppt_hue");
    hipCheck(hipDeviceSynchronize(), "sync hue");
    freeParam(hD); (void)hipFree(roiD);
    return *this;
}

rppArithmeticEngine& rppArithmeticEngine::saturation(const std::vector<float>& factor)
{
    float* sD = uploadParam(factor);
    float* roiD = nullptr;
    hipCheck(hipMalloc(&roiD, rois_.size() * sizeof(RpptROI)), "hipMalloc roi");
    hipCheck(hipMemcpy(roiD, rois_.data(), rois_.size() * sizeof(RpptROI),
                       hipMemcpyHostToDevice), "roi H2D");
    rppCheck(rppt_saturation(devBuf_, &desc_, devBuf_, &desc_,
                             sD,
                             reinterpret_cast<RpptROIPtr>(roiD),
                             RpptRoiType::XYWH, handle_, RPP_HIP_BACKEND),
             "rppt_saturation");
    hipCheck(hipDeviceSynchronize(), "sync saturation");
    freeParam(sD); (void)hipFree(roiD);
    return *this;
}
