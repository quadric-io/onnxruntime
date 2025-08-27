#include "test/providers/compare_provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(QuantizeLinearFixedPointTest, BasicTest) {
  OpTester tester("QuantizeLinearFixedPoint", 1, kQuadricDomain);
  tester.AddInput<int32_t>("X", {5}, {-2013265920, 187904816, 268435456, 456340288, 1946157056});
  tester.AddInput<int8_t>("x_frac_bits", {}, {27});
  tester.AddInput<float>("scale", {}, {0.01865844801068306});
  tester.AddInput<int8_t>("zero_point", {}, {-14});
  tester.AddOutput<int8_t>("Y", {5}, {-128, 61, 93, 127, 127});
  tester.Run();
}

TEST(QuantizeLinearFixedPointTest, FourDimTest) {
  OpTester tester("QuantizeLinearFixedPoint", 1, kQuadricDomain);

  // 4D tensor shape: (2, 2, 2, 2) -> Requires 16 elements
  tester.AddInput<int32_t>("X", {1, 1, 2, 3}, {-2013265920, 187904816, 268435456, 456340288, 1946157056, 2080374784});

  tester.AddInput<int8_t>("x_frac_bits", {}, {27});
  tester.AddInput<float>("scale", {}, {0.01865844801068306});
  tester.AddInput<int8_t>("zero_point", {}, {-14});

  // Expected output with 16 elements, maintaining the same shape
  tester.AddOutput<int8_t>("Y", {1, 1, 2, 3}, {-128, 61, 93, 127, 127, 127});

  tester.Run();
}

TEST(QuantizeLinearFixedPointTest, ExactMidpointBeforeRound) {
  OpTester tester("QuantizeLinearFixedPoint", 1, kQuadricDomain);
  tester.AddInput<int32_t>("X", {1}, {116449696});
  tester.AddInput<int8_t>("x_frac_bits", {}, {27});
  tester.AddInput<float>("scale", {}, {0x1.31b3340000000p-6f});
  tester.AddInput<int8_t>("zero_point", {}, {-14});
  tester.AddOutput<int8_t>("Y", {1}, {33});
  tester.Run();
}

}  // namespace test
}  // namespace onnxruntime
