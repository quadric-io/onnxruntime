#include <cmath>
#include <random>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

struct QLinearConvTransposeParams {
  int64_t N = 1;
  int64_t C_in = 2;
  int64_t H = 4;
  int64_t W = 4;
  int64_t C_out = 3;
  // Spatial attributes as in the ONNX node: {H, W}, and pads as {top, left, bottom, right}.
  std::vector<int64_t> kernel_shape{3, 3};
  std::vector<int64_t> strides{2, 2};
  std::vector<int64_t> pads{1, 1, 1, 1};
  std::vector<int64_t> dilations{1, 1};
  std::vector<int64_t> output_padding{0, 0};
  int64_t group = 1;
  int8_t x_zero_point = 0;
  int8_t w_zero_point = 0;
  int8_t y_zero_point = 0;
  bool has_bias = false;
  // Pass the optional bias as an empty input name rather than leaving it off.
  bool empty_bias_input = false;
};

// Same arithmetic as MlasRequantizeOutput: scale, clamp to the output range, round half to even.
int8_t Requantize(int32_t sum, float scale, int8_t zero_point) {
  float f = static_cast<float>(sum) * scale;
  f = std::min(f, 127.f - zero_point);
  f = std::max(f, -128.f - zero_point);
  return static_cast<int8_t>(std::nearbyint(f) + zero_point);
}

// Builds random int8 inputs, computes the expected output with a naive int32 ConvTranspose over the
// zero-point-adjusted values, and checks the kernel against it.
void RunQLinearConvTransposeTest(const QLinearConvTransposeParams& p) {
  const float x_scale = 0.03f;
  const float w_scale = 0.02f;
  const float y_scale = 0.2f;
  const int64_t C_in_per_group = p.C_in / p.group;
  const int64_t C_out_per_group = p.C_out / p.group;
  const int64_t kH = p.kernel_shape[0];
  const int64_t kW = p.kernel_shape[1];
  const int64_t H_out = (p.H - 1) * p.strides[0] - p.pads[0] - p.pads[2] + p.dilations[0] * (kH - 1) + 1 +
                        p.output_padding[0];
  const int64_t W_out = (p.W - 1) * p.strides[1] - p.pads[1] - p.pads[3] + p.dilations[1] * (kW - 1) + 1 +
                        p.output_padding[1];

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> x_dist(-128, 127);
  std::uniform_int_distribution<int> w_dist(-127, 127);
  std::uniform_int_distribution<int32_t> b_dist(-2000, 2000);

  std::vector<int8_t> X(p.N * p.C_in * p.H * p.W);
  std::vector<int8_t> W(p.C_in * C_out_per_group * kH * kW);
  std::vector<int32_t> B(p.has_bias ? p.C_out : 0);
  for (auto& x : X) x = static_cast<int8_t>(x_dist(rng));
  for (auto& w : W) w = static_cast<int8_t>(w_dist(rng));
  for (auto& b : B) b = b_dist(rng);

  // Scatter each input pixel through the kernel into the output it contributes to.
  std::vector<int32_t> acc(p.N * p.C_out * H_out * W_out, 0);
  for (int64_t n = 0; n < p.N; ++n) {
    for (int64_t ci = 0; ci < p.C_in; ++ci) {
      const int64_t group_id = ci / C_in_per_group;
      for (int64_t ih = 0; ih < p.H; ++ih) {
        for (int64_t iw = 0; iw < p.W; ++iw) {
          const int32_t x = X[((n * p.C_in + ci) * p.H + ih) * p.W + iw] - p.x_zero_point;
          for (int64_t co = 0; co < C_out_per_group; ++co) {
            for (int64_t kh = 0; kh < kH; ++kh) {
              for (int64_t kw = 0; kw < kW; ++kw) {
                const int64_t oh = ih * p.strides[0] - p.pads[0] + kh * p.dilations[0];
                const int64_t ow = iw * p.strides[1] - p.pads[1] + kw * p.dilations[1];
                if (oh < 0 || oh >= H_out || ow < 0 || ow >= W_out) {
                  continue;
                }
                const int32_t w = W[((ci * C_out_per_group + co) * kH + kh) * kW + kw] - p.w_zero_point;
                const int64_t y_channel = group_id * C_out_per_group + co;
                acc[((n * p.C_out + y_channel) * H_out + oh) * W_out + ow] += x * w;
              }
            }
          }
        }
      }
    }
  }

  const float scale = x_scale * w_scale / y_scale;
  const int64_t output_size = H_out * W_out;
  std::vector<int8_t> Y(acc.size());
  for (size_t i = 0; i < acc.size(); ++i) {
    const int32_t bias = p.has_bias ? B[(i / output_size) % p.C_out] : 0;
    Y[i] = Requantize(acc[i] + bias, scale, p.y_zero_point);
  }

  OpTester test("QLinearConvTranspose", 1, kMSDomain);
  test.AddAttribute("kernel_shape", p.kernel_shape);
  test.AddAttribute("strides", p.strides);
  test.AddAttribute("pads", p.pads);
  test.AddAttribute("dilations", p.dilations);
  test.AddAttribute("output_padding", p.output_padding);
  test.AddAttribute("group", p.group);

  test.AddInput<int8_t>("x", {p.N, p.C_in, p.H, p.W}, X);
  test.AddInput<float>("x_scale", {}, {x_scale}, true);
  test.AddInput<int8_t>("x_zero_point", {}, {p.x_zero_point}, true);
  test.AddInput<int8_t>("w", {p.C_in, C_out_per_group, kH, kW}, W, true);
  test.AddInput<float>("w_scale", {}, {w_scale}, true);
  test.AddInput<int8_t>("w_zero_point", {}, {p.w_zero_point}, true);
  test.AddInput<float>("y_scale", {}, {y_scale}, true);
  test.AddInput<int8_t>("y_zero_point", {}, {p.y_zero_point}, true);
  if (p.has_bias) {
    test.AddInput<int32_t>("b", {p.C_out}, B, true);
  } else if (p.empty_bias_input) {
    test.AddOptionalInputEdge<int32_t>();
  }
  test.AddOutput<int8_t>("y", {p.N, p.C_out, H_out, W_out}, Y);
  test.Run();
}

}  // namespace

TEST(QLinearConvTransposeTest, Symmetric) {
  RunQLinearConvTransposeTest({});
}

// Asymmetric activation with symmetric weight, as produced by asymmetric activation quantization.
TEST(QLinearConvTransposeTest, ActivationZeroPoint) {
  QLinearConvTransposeParams p;
  p.x_zero_point = -40;
  p.y_zero_point = 25;
  RunQLinearConvTransposeTest(p);
}

TEST(QLinearConvTransposeTest, WeightZeroPoint) {
  QLinearConvTransposeParams p;
  p.w_zero_point = 7;
  RunQLinearConvTransposeTest(p);
}

TEST(QLinearConvTransposeTest, ZeroPointsWithBias) {
  QLinearConvTransposeParams p;
  p.x_zero_point = -128;
  p.w_zero_point = -9;
  p.y_zero_point = 17;
  p.has_bias = true;
  RunQLinearConvTransposeTest(p);
}

TEST(QLinearConvTransposeTest, EmptyBiasInput) {
  QLinearConvTransposeParams p;
  p.x_zero_point = -40;
  p.y_zero_point = 25;
  p.empty_bias_input = true;
  RunQLinearConvTransposeTest(p);
}

TEST(QLinearConvTransposeTest, Group) {
  QLinearConvTransposeParams p;
  p.C_in = 4;
  p.C_out = 4;
  p.group = 2;
  RunQLinearConvTransposeTest(p);
}

TEST(QLinearConvTransposeTest, Batch) {
  QLinearConvTransposeParams p;
  p.N = 2;
  RunQLinearConvTransposeTest(p);
}

// Different values per spatial axis, so swapped H/W arguments to Col2im would fail.
TEST(QLinearConvTransposeTest, NonSquareKernelAsymmetricPadsOutputPadding) {
  QLinearConvTransposeParams p;
  p.H = 4;
  p.W = 5;
  p.kernel_shape = {3, 2};
  p.strides = {2, 3};
  p.pads = {1, 0, 0, 1};
  p.dilations = {1, 2};
  p.output_padding = {1, 2};
  p.x_zero_point = -40;
  p.w_zero_point = 3;
  p.y_zero_point = 25;
  p.has_bias = true;
  RunQLinearConvTransposeTest(p);
}

TEST(QLinearConvTransposeTest, BatchGroupDilationZeroPointsWithBias) {
  QLinearConvTransposeParams p;
  p.N = 2;
  p.C_in = 4;
  p.C_out = 6;
  p.group = 2;
  p.strides = {1, 1};
  p.pads = {0, 0, 0, 0};
  p.dilations = {2, 2};
  p.x_zero_point = 64;
  p.w_zero_point = 3;
  p.y_zero_point = -30;
  p.has_bias = true;
  RunQLinearConvTransposeTest(p);
}

}  // namespace test
}  // namespace onnxruntime
