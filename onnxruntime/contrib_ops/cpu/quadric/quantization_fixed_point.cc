#include "core/framework/op_kernel.h"
#include "core/common/common.h"

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
    kQuadricDomain,  // Ensure this domain is defined
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>())  // Input tensor
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())  // Scale
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>()) // Zero-point
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()), // Output
    DequantizeLinearFixedPoint);

// Compute function
Status DequantizeLinearFixedPoint::Compute(OpKernelContext* ctx) const {
  // Get input tensors
  const auto* X = ctx->Input<Tensor>(0);
  const auto* scale = ctx->Input<Tensor>(1);
  const auto* zero_point = ctx->Input<Tensor>(2);

  // Ensure inputs exist
  ORT_ENFORCE(X != nullptr, "Input X is null");
  ORT_ENFORCE(scale != nullptr, "Scale is null");
  ORT_ENFORCE(zero_point != nullptr, "Zero point is null");

  // Get input data pointers
  const int8_t* x_data = X->Data<int8_t>();
  float s = *(scale->Data<float>());
  int8_t zp = *(zero_point->Data<int8_t>());

  // Create output tensor (same shape as input)
  auto* Y = ctx->Output(0, X->Shape());
  int32_t* y_data = Y->MutableData<int32_t>();

  // Get tensor size
  size_t tensor_size = X->Shape().Size();

  // Compute fixed-point dequantization
  int frac_bits = 16;  // Define fixed-point fraction bits
  int scale_qfp = static_cast<int>(s * (1 << frac_bits));  // Convert scale to Q-format
  int shift = frac_bits - 16;  // Adjust shift

  for (size_t i = 0; i < tensor_size; ++i) {
    int temp = static_cast<int32_t>(x_data[i]) - static_cast<int32_t>(zp);
    int64_t product = static_cast<int64_t>(temp) * scale_qfp;
    y_data[i] = shift >= 0 ? (product >> shift) : (product << -shift);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
