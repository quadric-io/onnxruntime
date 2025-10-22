#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#include "core/mlas/inc/mlas.h"
#include "fixed_point.h"

namespace onnxruntime {

namespace contrib {

struct GeluFixedPointAttrs {
  std::string approximate;
  explicit GeluFixedPointAttrs(const OpKernelInfo& info) {
    auto status = info.GetAttr<std::string>("approximate", &approximate);
    if (!status.IsOK())
      approximate = "none"; /* Default approximation is none */
  }
  ~GeluFixedPointAttrs() = default;
};

// --- GeluFixedPoint
class GeluFixedPoint final : public OpKernel {
 public:
  explicit GeluFixedPoint(const OpKernelInfo& info) : OpKernel(info), gelu__fxp_attrs_(info) {}
  Status Compute(OpKernelContext* ctx) const override;

 private:
  GeluFixedPointAttrs gelu__fxp_attrs_;
};

ONNX_OPERATOR_KERNEL_EX(
    GeluFixedPoint,
    kQuadricDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())    // Input tensor
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())    // Input frac bits
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())    // Output frac bits
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),  // Output
    GeluFixedPoint);

Status GeluFixedPoint::Compute(OpKernelContext* ctx) const {
  const Tensor* inp = ctx->Input<Tensor>(0);
  const Tensor* inpFracBitsTensor = ctx->Input<Tensor>(1);
  const Tensor* outFracBits = ctx->Input<Tensor>(2);

  // Validate inputs
  ORT_ENFORCE(inp != nullptr, "Input is null");
  ORT_ENFORCE(inpFracBitsTensor != nullptr, "inpFracBits is null");
  ORT_ENFORCE(outFracBits != nullptr, "outFracBits is null");

  // input, scale, bias data
  const std::int32_t* inp_data = inp->Data<std::int32_t>();

  const std::int8_t inpFbits = *(inpFracBitsTensor->Data<std::int8_t>());
  const std::int8_t outFbits = *(outFracBits->Data<std::int8_t>());

  // Allocate output tensor
  auto* out = ctx->Output(0, inp->Shape());
  std::int32_t* out_data = out->MutableData<int32_t>();

  // Frac bits analysis is done inside tvm.
  constexpr std::uint8_t erfFracBits = 30;
  constexpr std::uint8_t erfPlusOneFracBits = 29;
  std::uint8_t xmulFracBits = std::max(inpFbits, outFbits) - 2;
  constexpr std::uint8_t sqrt2InvFracBits = 31;
  std::int8_t erfInShift = inpFbits + sqrt2InvFracBits - inpFbits;

  std::int8_t xmulShift = inpFbits + erfPlusOneFracBits - xmulFracBits;

  // convert to fixed-point
  constexpr double sqrt2Inv = 0.70710677;
  std::vector<double> sqrt2InvVec = {sqrt2Inv};
  auto s = dataToQfp(sqrt2InvVec, sqrt2InvFracBits, 32, false);
  std::int32_t sqrt2InvQFp = static_cast<int32_t>(s.first[0]);

  constexpr double point5 = 0.5;
  std::vector<double> point5Vec = {point5};
  auto p = dataToQfp(point5Vec, xmulFracBits, 32, false);
  std::int32_t Point5QFp = static_cast<int32_t>(p.first[0]);

  std::int8_t finalShift = xmulFracBits + xmulFracBits - outFbits;

  size_t tensor_size = inp->Shape().Size();

  for (size_t i = 0; i < tensor_size; i++) {
    std::int32_t erfIn = chimera::fixedPointMultiply(inp_data[i], sqrt2InvQFp, erfInShift);
    std::int32_t erfOut = erf(erfIn);
    std::int32_t erfPlusOne = ((erfOut >> (erfFracBits - erfPlusOneFracBits)) + 1) << erfPlusOneFracBits;
    std::int32_t xmulOut = chimera::fixedPointMultiply(inp_data[i], erfPlusOne, xmulShift);
    out_data[i] = chimera::fixedPointMultiply(xmulOut, Point5QFp, finalShift);
  }
  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
