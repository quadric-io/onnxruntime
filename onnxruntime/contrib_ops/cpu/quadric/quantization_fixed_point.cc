#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include <cmath>  // For log2()
#include <limits> // For int8_t min/max
#include <iostream>
#include <iomanip>  // For std::setprecision
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

// --- DequantizeLinearFixedPoint

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

// Compute required fractional bits given a range
int ComputeFracBits(float min_val, float max_val) {
  constexpr int kMaxFracBits = 31;
  float largest = std::max(std::fabs(min_val), std::fabs(max_val));
  return (largest < 1.0f) ? kMaxFracBits : (kMaxFracBits - static_cast<int>(std::ceil(std::log2(largest + 1))));
}

int FixedPointMultiply(int32_t a, int32_t b, int shift) {
  int64_t product = static_cast<int64_t>(a) * b;
  return (shift > 0) ? (product >> shift) : (product << -shift);
}

Status DequantizeLinearFixedPoint::Compute(OpKernelContext* ctx) const {
  // Retrieve input tensors
  const auto* X = ctx->Input<Tensor>(0);
  const auto* scale = ctx->Input<Tensor>(1);
  const auto* zero_point = ctx->Input<Tensor>(2);

  // Validate inputs
  ORT_ENFORCE(X, "Input tensor 'X' is null.");
  ORT_ENFORCE(scale, "Scale tensor is null.");
  ORT_ENFORCE(zero_point, "Zero-point tensor is null.");

  // Extract values
  const int8_t* x_data = X->Data<int8_t>();
  float s = *(scale->Data<float>());
  int8_t zp = *(zero_point->Data<int8_t>());

  // Compute range and fractional bits
  auto [min_val, max_val] = GetDequantizedRange(s, zp);
  int result_frac_bits = ComputeFracBits(min_val, max_val);

  // Convert scale to fixed-point
  std::vector<double> ScaleValueVec = {s};
  auto p = dataToQfp(ScaleValueVec, -1, 32, false);  // Avoid creating a vector
  int scale_frac_bits = p.second;
  int32_t scale_qfp = static_cast<int32_t>(p.first[0]);

  int shift = scale_frac_bits - result_frac_bits;

  // Allocate output tensor
  auto* Y = ctx->Output(0, X->Shape());
  int32_t* y_data = Y->MutableData<int32_t>();
  size_t tensor_size = X->Shape().Size();

  for (size_t i = 0; i < tensor_size; ++i) {
    y_data[i] = FixedPointMultiply(x_data[i] - zp, scale_qfp, shift);
  }

  return Status::OK();
}

// --- QuantizeLinearFixedPoint
class QuantizeLinearFixedPoint final : public OpKernel {
  public:
   explicit QuantizeLinearFixedPoint(const OpKernelInfo& info) : OpKernel(info) {}
   Status Compute(OpKernelContext* ctx) const override;
 };

 // Register Kernel
 ONNX_OPERATOR_KERNEL_EX(
     QuantizeLinearFixedPoint,
     kQuadricDomain,
     1,
     kCpuExecutionProvider,
     KernelDefBuilder()
         .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())  // Input tensor
         .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())  // x_frac_bits
         .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>())   // Scale
         .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())  // Zero-point
         .TypeConstraint("T4", DataTypeImpl::GetTensorType<int8_t>()), // Output
     QuantizeLinearFixedPoint);

 // Compute function
 Status QuantizeLinearFixedPoint::Compute(OpKernelContext* ctx) const {
   // Get input tensors
   const auto* X = ctx->Input<Tensor>(0);
   const auto* x_frac_bits = ctx->Input<Tensor>(1);
   const auto* scale = ctx->Input<Tensor>(2);
   const auto* zero_point = ctx->Input<Tensor>(3);

   // Validate inputs
   ORT_ENFORCE(X != nullptr, "Input X is null");
   ORT_ENFORCE(x_frac_bits != nullptr, "x_frac_bits is null");
   ORT_ENFORCE(scale != nullptr, "Scale is null");
   ORT_ENFORCE(zero_point != nullptr, "Zero point is null");

   // Retrieve input data
   const int32_t* x_data = X->Data<int32_t>();
   int8_t x_frac_bits_val = *(x_frac_bits->Data<int8_t>());
   double s = *(scale->Data<float>());
   int8_t zp = *(zero_point->Data<int8_t>());

   double scale_inv = 1.0 / s;
    std::vector<double> ScaleValueVec = {scale_inv};
    auto p = dataToQfp(ScaleValueVec, -1, 32, false); // Returns std::make_pair(qfp, fracBits)
    int64_t scale_inv_qfp = p.first[0];
    int scale_inv_frac_bits = p.second;

   int post_mac_int_bits = 29;
   int post_mac_frac_bits = 31 - post_mac_int_bits;

   int result_frac_bits = post_mac_frac_bits;
   int shift = scale_inv_frac_bits + x_frac_bits_val - result_frac_bits;
   if (shift > 31){
        shift = 31;
        result_frac_bits = scale_inv_frac_bits + x_frac_bits_val - 31;
   }

   auto* Y = ctx->Output(0, X->Shape());
   int8_t* y_data = Y->MutableData<int8_t>();
   size_t tensor_size = X->Shape().Size();
   for (size_t i = 0; i < tensor_size; ++i) {
    int32_t product = FixedPointMultiply(x_data[i], scale_inv_qfp, shift);
     // Rounding
     int32_t product_round = fxRoundPosInf(static_cast<int32_t>(product), static_cast<uint8_t>(result_frac_bits));

     // Clip and apply zero-point
     y_data[i] = static_cast<int8_t>(std::min(std::max(product_round + zp, static_cast<int32_t>(std::numeric_limits<int8_t>::min())), static_cast<int32_t>(std::numeric_limits<int8_t>::max())));
   }
   return Status::OK();
 }

}  // namespace contrib
}  // namespace onnxruntime
