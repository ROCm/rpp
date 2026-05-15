/*
 * test_arith_engine.cpp — smoke tests for rppArithmeticEngine
 *
 * Build & run:
 *   source set_rocm_env.sh
 *   cd rpp_arith_engine/build && cmake .. && make -j$(nproc)
 *   ./test_arith_engine
 */

#include "rpp_arith_engine.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ─── Helpers ──────────────────────────────────────────────────────────────────

static int g_pass = 0, g_fail = 0;

static void check(bool cond, const char* msg)
{
    if (cond) {
        std::printf("  PASS  %s\n", msg);
        ++g_pass;
    } else {
        std::printf("  FAIL  %s\n", msg);
        ++g_fail;
    }
}

static bool nearlyEqual(float a, float b, float tol = 1e-4f)
{
    return std::fabs(a - b) <= tol;
}

// Fill a host buffer: every element = val
static std::vector<float> filled(size_t n, float val)
{
    return std::vector<float>(n, val);
}

// ─── Tests ────────────────────────────────────────────────────────────────────

static void test_scalar_add()
{
    std::printf("\n[test_scalar_add]\n");
    // 1 image, 4x4, 3 channels; all pixels = 0.5
    rppArithmeticEngine eng(1, 4, 4, 3);
    auto src = filled(1 * 4 * 4 * 3, 0.5f);
    eng.upload(src.data());

    auto result = eng + 0.3f;   // expect 0.8

    std::vector<float> out(result.numel());
    result.download(out.data());
    check(nearlyEqual(out[0], 0.8f), "scalar add: 0.5 + 0.3 = 0.8");
}

static void test_scalar_subtract()
{
    std::printf("\n[test_scalar_subtract]\n");
    rppArithmeticEngine eng(1, 4, 4, 3);
    eng.upload(filled(1 * 4 * 4 * 3, 1.0f).data());

    auto result = eng - 0.4f;   // expect 0.6

    std::vector<float> out(result.numel());
    result.download(out.data());
    check(nearlyEqual(out[0], 0.6f), "scalar sub: 1.0 - 0.4 = 0.6");
}

static void test_scalar_multiply()
{
    std::printf("\n[test_scalar_multiply]\n");
    rppArithmeticEngine eng(1, 4, 4, 3);
    eng.upload(filled(1 * 4 * 4 * 3, 0.5f).data());

    auto result = eng * 2.0f;   // expect 1.0

    std::vector<float> out(result.numel());
    result.download(out.data());
    check(nearlyEqual(out[0], 1.0f), "scalar mul: 0.5 * 2.0 = 1.0");
}

static void test_scalar_divide()
{
    std::printf("\n[test_scalar_divide]\n");
    rppArithmeticEngine eng(1, 4, 4, 3);
    eng.upload(filled(1 * 4 * 4 * 3, 0.8f).data());

    auto result = eng / 4.0f;   // expect 0.2

    std::vector<float> out(result.numel());
    result.download(out.data());
    check(nearlyEqual(out[0], 0.2f), "scalar div: 0.8 / 4.0 = 0.2");
}

static void test_tensor_add()
{
    std::printf("\n[test_tensor_add]\n");
    rppArithmeticEngine a(2, 4, 4, 3);
    rppArithmeticEngine b(2, 4, 4, 3);
    a.upload(filled(2 * 4 * 4 * 3, 0.3f).data());
    b.upload(filled(2 * 4 * 4 * 3, 0.5f).data());

    auto result = a + b;   // expect 0.8

    std::vector<float> out(result.numel());
    result.download(out.data());
    check(nearlyEqual(out[0], 0.8f), "tensor add: 0.3 + 0.5 = 0.8");
    check(result.n() == 2, "tensor add: batch size preserved");
}

static void test_tensor_subtract()
{
    std::printf("\n[test_tensor_subtract]\n");
    rppArithmeticEngine a(1, 4, 4, 3);
    rppArithmeticEngine b(1, 4, 4, 3);
    a.upload(filled(1 * 4 * 4 * 3, 0.9f).data());
    b.upload(filled(1 * 4 * 4 * 3, 0.4f).data());

    auto result = a - b;   // expect 0.5

    std::vector<float> out(result.numel());
    result.download(out.data());
    check(nearlyEqual(out[0], 0.5f), "tensor sub: 0.9 - 0.4 = 0.5");
}

static void test_chained_color_augmentations()
{
    std::printf("\n[test_chained_color_augmentations]\n");
    // Use 0.0 so gammaLUT[0] = 0.0^gamma = 0.0 exactly (no LUT quantization)
    rppArithmeticEngine eng(1, 8, 8, 3);
    eng.upload(filled(1 * 8 * 8 * 3, 0.0f).data());

    // gammaCorrection: gammaLUT[0] = 0.0 → identity for 0.0 input
    // brightness: alpha=1, beta=0 → identity
    eng.gammaCorrection({1.0f})
       .brightness({1.0f}, {0.0f});

    std::vector<float> out(eng.numel());
    eng.download(out.data());
    check(nearlyEqual(out[0], 0.0f), "chained aug (identity chain on 0.0): value unchanged");

    // Separately verify gammaCorrection quantization is within LUT resolution (1/255)
    rppArithmeticEngine eng2(1, 8, 8, 3);
    eng2.upload(filled(1 * 8 * 8 * 3, 0.5f).data());
    eng2.gammaCorrection({1.0f});
    std::vector<float> out2(eng2.numel());
    eng2.download(out2.data());
    check(nearlyEqual(out2[0], 0.5f, 1.0f/255.0f),
          "gammaCorrection(gamma=1) on 0.5: within LUT resolution (1/255)");
}

static void test_brightness_augmentation()
{
    std::printf("\n[test_brightness_augmentation]\n");
    rppArithmeticEngine eng(1, 4, 4, 3);
    eng.upload(filled(1 * 4 * 4 * 3, 0.4f).data());

    // alpha=2, beta=0 → 0.4 * 2 = 0.8
    eng.brightness({2.0f}, {0.0f});

    std::vector<float> out(eng.numel());
    eng.download(out.data());
    check(nearlyEqual(out[0], 0.8f, 1e-3f), "brightness: 0.4 * alpha=2 + beta=0 = 0.8");
}

static void test_move_semantics()
{
    std::printf("\n[test_move_semantics]\n");
    rppArithmeticEngine a(1, 4, 4, 3);
    a.upload(filled(1 * 4 * 4 * 3, 0.7f).data());

    rppArithmeticEngine b = std::move(a);  // move-construct
    std::vector<float> out(b.numel());
    b.download(out.data());
    check(nearlyEqual(out[0], 0.7f), "move-construct: data preserved");
    check(b.n() == 1 && b.h() == 4, "move-construct: shape preserved");
}

static void test_shape_mismatch_throws()
{
    std::printf("\n[test_shape_mismatch_throws]\n");
    rppArithmeticEngine a(1, 4, 4, 3);
    rppArithmeticEngine b(2, 4, 4, 3);  // different N
    bool threw = false;
    try { auto r = a + b; }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "shape mismatch throws std::invalid_argument");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main()
{
    std::printf("=================================================\n");
    std::printf("  rppArithmeticEngine — smoke tests\n");
    std::printf("=================================================\n");

    try {
        test_scalar_add();
        test_scalar_subtract();
        test_scalar_multiply();
        test_scalar_divide();
        test_tensor_add();
        test_tensor_subtract();
        test_chained_color_augmentations();
        test_brightness_augmentation();
        test_move_semantics();
        test_shape_mismatch_throws();
    } catch (const std::exception& e) {
        std::printf("\nFATAL: uncaught exception: %s\n", e.what());
        return 1;
    }

    std::printf("\n=================================================\n");
    std::printf("  Results: %d passed, %d failed\n", g_pass, g_fail);
    std::printf("=================================================\n");
    return g_fail > 0 ? 1 : 0;
}
