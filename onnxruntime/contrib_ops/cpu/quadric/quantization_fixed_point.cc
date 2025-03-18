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
std::pair<float, float> getDequantizedRange(float scale, int8_t zeroPoint) {
    constexpr int8_t int8Min = std::numeric_limits<int8_t>::min();
    constexpr int8_t int8Max = std::numeric_limits<int8_t>::max();
    return { (int8Min - zeroPoint) * scale, (int8Max - zeroPoint) * scale };
}

// Compute required fractional bits given a range
int computeFracBits(float minVal, float maxVal) {
  constexpr int maxFracBits = 31;
  float largest = std::max(std::fabs(minVal), std::fabs(maxVal));
  return (largest < 1.0f) ? maxFracBits : (maxFracBits - static_cast<int>(std::ceil(std::log2(largest + 1))));
}

// Helper: Fixed-point multiplication with shifting
int fixedPointMultiply(int32_t a, int32_t b, int shift) {
  int64_t product = static_cast<int64_t>(a) * b;
  return (shift > 0) ? (product >> shift) : (product << -shift);
}

Status DequantizeLinearFixedPoint::Compute(OpKernelContext* ctx) const {
  // Retrieve input tensors
  const auto* X = ctx->Input<Tensor>(0);
  const auto* scale = ctx->Input<Tensor>(1);
  const auto* zeroPoint = ctx->Input<Tensor>(2);

  // Validate inputs
  ORT_ENFORCE(X, "Input tensor 'X' is null.");
  ORT_ENFORCE(scale, "Scale tensor is null.");
  ORT_ENFORCE(zeroPoint, "Zero-point tensor is null.");

  // Extract values
  const int8_t* xData = X->Data<int8_t>();
  float s = *(scale->Data<float>());
  int8_t zp = *(zeroPoint->Data<int8_t>());

  // Compute range and fractional bits
  auto [minVal, maxVal] = getDequantizedRange(s, zp);
  int resultFracBits = computeFracBits(minVal, maxVal);

  // Convert scale to fixed-point
  std::vector<double> scaleValueVec = {s};
  auto p = dataToQfp(scaleValueVec, -1, 32, false);
  int scaleFracBits = p.second;
  int32_t scaleQfp = static_cast<int32_t>(p.first[0]);

  int shift = scaleFracBits - resultFracBits;

  // Allocate output tensor
  auto* Y = ctx->Output(0, X->Shape());
  int32_t* yData = Y->MutableData<int32_t>();
  size_t tensorSize = X->Shape().Size();

  for (size_t i = 0; i < tensorSize; ++i) {
    yData[i] = fixedPointMultiply(xData[i] - zp, scaleQfp, shift);
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
         .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())  // xFracBits
         .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>())   // Scale
         .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())  // Zero-point
         .TypeConstraint("T4", DataTypeImpl::GetTensorType<int8_t>()), // Output
     QuantizeLinearFixedPoint);

 // Compute function
 Status QuantizeLinearFixedPoint::Compute(OpKernelContext* ctx) const {
  // Get input tensors
  const auto* X = ctx->Input<Tensor>(0);
  const auto* xFracBitsTensor = ctx->Input<Tensor>(1);
  const auto* scale = ctx->Input<Tensor>(2);
  const auto* zeroPoint = ctx->Input<Tensor>(3);

  // Validate inputs
  ORT_ENFORCE(X != nullptr, "Input X is null");
  ORT_ENFORCE(xFracBitsTensor != nullptr, "xFracBits is null");
  ORT_ENFORCE(scale != nullptr, "Scale is null");
  ORT_ENFORCE(zeroPoint != nullptr, "Zero point is null");

  // Retrieve input data
  const int32_t* x_data = X->Data<int32_t>();
  int8_t xFracBits = *(xFracBitsTensor->Data<int8_t>());
  double s = *(scale->Data<float>());
  int8_t zp = *(zeroPoint->Data<int8_t>());

  double scaleInv = 1.0 / s;
   std::vector<double> ScaleValueVec = {scaleInv};
   auto p = dataToQfp(ScaleValueVec, -1, 32, false); // Returns std::make_pair(qfp, fracBits)
   int64_t scaleInvQfp = p.first[0];
   int scaleInvFracvBits = p.second;

  int postMacIntBits = 29;
  int postMacFracBits = 31 - postMacIntBits;

  int resultFracBits = postMacFracBits;
  int shift = scaleInvFracvBits + xFracBits - resultFracBits;
  if (shift > 31){
       shift = 31;
       resultFracBits = scaleInvFracvBits + xFracBits - 31;
  }

  auto* Y = ctx->Output(0, X->Shape());
  int8_t* yData = Y->MutableData<int8_t>();
  size_t tensor_size = X->Shape().Size();
  for (size_t i = 0; i < tensor_size; ++i) {
   int32_t product = fixedPointMultiply(x_data[i], scaleInvQfp, shift);
    // Rounding
    int32_t productRound = fxRoundPosInf(static_cast<int32_t>(product), static_cast<uint8_t>(resultFracBits));

    // Clip and apply zero-point
    yData[i] = static_cast<int8_t>(std::min(std::max(productRound + zp, static_cast<int32_t>(std::numeric_limits<int8_t>::min())), static_cast<int32_t>(std::numeric_limits<int8_t>::max())));
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
