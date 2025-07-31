#include "test/providers/compare_provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(DequantizeLinearFixedPointTest, BasicTest) {
  OpTester tester("DequantizeLinearFixedPoint", 1, kQuadricDomain);
  tester.AddInput<int8_t>("X", {5}, {-128, 1, 2, 3, 127});
  tester.AddInput<int8_t>("y_frac_bits", {}, {27});
  tester.AddInput<float>("scale", {}, {0.10242629051208496});
  tester.AddInput<int8_t>("zero_point", {}, {5});
  tester.AddOutput<int32_t>("Y", {5}, {-1828407392, -54989696, -41242272, -27494848, 1677185728});
  tester.Run();
}
}  // namespace test
}  // namespace onnxruntime
