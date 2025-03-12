#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include <cmath>  // For log2()
#include <limits> // For int8_t min/max
#include <iostream>


namespace onnxruntime {
namespace contrib {

class DequantizeLinearFixedPoint final : public OpKernel {
 public:
  explicit DequantizeLinearFixedPoint(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* ctx) const override;
};

// Register kernel
ONNX_OPERATOR_KERNEL_EX(
    DequantizeLinearFixedPoint,
    kQuadricDomain,  // Ensure this is defined in contrib_ops.h
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>())  // Input tensor
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())  // Scale
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>()) // Zero-point
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()), // Output
    DequantizeLinearFixedPoint);

// Helper: Compute min/max range from scale & zero-point
std::pair<float, float> GetDequantizedRange(float scale, int8_t zero_point) {
    int8_t int8_min = std::numeric_limits<int8_t>::min();
    int8_t int8_max = std::numeric_limits<int8_t>::max();
    return { (int8_min - zero_point) * scale, (int8_max - zero_point) * scale };
}

// Helper: Compute number of fractional bits needed
int ComputeFracBits(float min_val, float max_val) {
    float range_min = std::fabs(min_val);
    float range_max = std::fabs(max_val);

    float largest = std::max(range_min, range_max);

    if (largest < 1.0f)
        return 31;  // Max precision for small values

    int int_bits = static_cast<int>(std::ceil(std::log2(largest + 1)));
    return 31 - int_bits;
}

// Compute function
Status DequantizeLinearFixedPoint::Compute(OpKernelContext* ctx) const {
  // Get input tensors
  const auto* X = ctx->Input<Tensor>(0);
  const auto* scale = ctx->Input<Tensor>(1);
  const auto* zero_point = ctx->Input<Tensor>(2);

  // Validate inputs
  ORT_ENFORCE(X != nullptr, "Input X is null");
  ORT_ENFORCE(scale != nullptr, "Scale is null");
  ORT_ENFORCE(zero_point != nullptr, "Zero point is null");

  // Retrieve input data
  const int8_t* x_data = X->Data<int8_t>();
  float s = *(scale->Data<float>());
  int8_t zp = *(zero_point->Data<int8_t>());

  std::cout << "Scale: " << s << ", Zero Point: " << static_cast<int>(zp) << std::endl;

  // Compute range of output
    auto [min_val, max_val] = GetDequantizedRange(s, zp);

//   // Compute fractional bits required
  int result_frac_bits = ComputeFracBits(min_val, max_val);
    std::cout << "Result frac bits: " << result_frac_bits << std::endl;
//   int result_frac_bits = 27;

  // Convert scale to Q-format fixed-point representation
  int scale_frac_bits = 31;  // Arbitrary choice, could be dynamic
  //   int scale_qfp = static_cast<int>(s * (1 << scale_frac_bits));
  int64_t scale_qfp_64 = static_cast<int64_t>(s * (1LL << scale_frac_bits));
  int scale_qfp = static_cast<int>(scale_qfp_64);  // Final cast after verifying range

  std::cout << "Scale QFP: " << scale_qfp << std::endl;
  int shift = scale_frac_bits - result_frac_bits;  // Bit shift adjustment

  // Create output tensor (same shape as input)
  auto* Y = ctx->Output(0, X->Shape());
  int32_t* y_data = Y->MutableData<int32_t>();

  // Get tensor size
  size_t tensor_size = X->Shape().Size();

  // Perform dequantization using integer arithmetic
  for (size_t i = 0; i < tensor_size; ++i) {
    int temp = static_cast<int32_t>(x_data[i]) - static_cast<int32_t>(zp);
    int64_t product = static_cast<int64_t>(temp) * scale_qfp;

    if (shift > 0) {
        y_data[i] = product >> shift;
    } else {
        y_data[i] = product << -shift;
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
