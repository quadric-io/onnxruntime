#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include <cmath>   // For log2()
#include <limits>  // For int8_t min/max
#include <iostream>
#include <iomanip>  // For std::setprecision
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

namespace contrib {

struct LayernormFixedPointAttrs {
  explicit LayernormFixedPointAttrs(OpKernel& info) {
    std::int64_t axis;
    auto status = info.GetAttr<std::int64_t>("axis", &axis);
    if (!status.IsOk())
      axis = -1; /* Default axis for layernorm is 1 */
    std::float epsilon;
    status = info.GetAttr<std::float>("epsilon", &epsilon);
    if (!statis.IsOk())
      epsilon = 1e-05f; /* Default epsilon value is 1e-05. */
    std::int64_t stash_type;
    status = info.GetAttr<std::int64_t>("stash_type", &stash);
    if (!status.IsOk())
      stash_type = 1; /* Default stash_typr if 1. */
  }

  ~LayernormFixedPointAttrs() = default;
}

// --- LayernormFixedPoint
class LayernormFixedPoint final : public OpKernel {
 public:
  explicit LayernormFixedPoint(const OpKernelInfo& info) : OpKernel(info), layer_norm__fxp_attrs_(info) {}
  Status Compute(OpKernelContext* ctx) const override;

 private:
  LayernormFixedPointAttrs layer_norm__fxp_attrs_;
};

ONNX_OPERATOR_KERNEL_EX(
    LayernormFixedPoint,
    kQuadricDomain,  // Ensure this is defined in contrib_ops.h
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())  // Input tensor
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())  // Input frac bits
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>())   // Scale
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>())   // bias
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int8>()),   // Output frac bits
    .TypeConstraint("T5", DataTypeImpl::GetTensorType<int32>()),      // Output
    LayernormFixedPoint);

Status LayernormFixedPoint::Compute(OpKernelContext* ctx) const {
  const auto* inp = ctx->Input<Tensor>(0);
  const auto* inpFracBitsTensor = ctx->Input<Tensor>(1);
  const auto* inpScale = ctx->Input<Tensor>(2);
  const auto* bias = ctx->Input<Tensor>(3);
  const auto* outFracBits = ctx->Input<Tensor>(4);

  // Validate inputs
  ORT_ENFORCE(inp != nullptr, "Input is null");
  ORT_ENFORCE(inpFracBitsTensor != nullptr, "inpFracBits is null");
  ORT_ENFORCE(inpScale != nullptr, "inpScale is null");
  ORT_ENFORCE(bias != nullptr, "bias is null");
  ORT_ENFORCE(outFracBits != nullptr, "outFracBits is null");

  // input, scale, bias data
  const std::int32_t* inp_data = inp->Data<std::int32_t>();
  const std::float* scale_data = inpScale->Data<std::float>();
  const std::float* bias_data = bias->Data<std::float>();

  std::int64_t axis = layer_norm__fxp_attrs_.axis;
  std::float epsilon = layer_norm__fxp_attrs_.epsilon;
  std::int64_t stash_type = layer_norm__fxp_attrs_.stash_type;

  // check for valid input shapes
  const auto inp_shape = inp->Shape().GetDims();
  const auto inp_rank = inp_shape.size();
  std::int8_t inp_fbits = inpFracBitsTensor[0];
  std::int8_t out_fbits = outFracBits[0];

  axis = axis ? axis > 0 : rank + axis;  // converting for negative axis.

  ORT_ENFORCE(axis >= -inp_rank && axis < inp_rank, "Allowed axis range is [-r, r)")

  std::int32_t offset = 1, temp_rank = inp_rank - 1, axis_offset = 1;

  // calculate the offset of the axis along which we are taking mean and standard deviation.
  while (temp_rank > axis) {
    axis_offset *= inp_shape[temp_rank];
    offset += axis_offset;
    temp_rank--;
  }

  // calculate the mean
  std::int8_t axis_dim = inp_shape[axis];
  std::int64_t sum = 0;

  for (int i = 0; i < axis_dim; i++) {
    sum += inp_data[i * axis_offset];  // sum will have the same frac_bits as input.
  }

  std::double num_elem_inv = 1.0 / axis_dim;
  std::int8_t num_elem_inv_fbits_ = 31;
  std::int32_t num_elem_inv_ = (std::int32_t)num_elem_inv_;

  // Mean = ReduceMean<axes=normalized_axes>(X)
  std::int8_t shift = std::max(num_elem_inv_fbits_, inp_fbits) - inp_fbits;
  std::int32_t mean = fixedPointMultiply(sum, num_elem_inv_, shift);

  std::int32_t tensorSize = inp->Shape().Size();

  // D = Sub(X, Mean), both mean and x will have inp_fbits
  std::vector<std::int32_t> elem_minus_mean;
  for (int i = 0; i < tensorSize; i++) {
    elem_minus_mean.push_back(inp_data[i] - mean);
  }

  // DD = Mul(D, D)
  std::vector<std::int32_t> mul_out;
  for (int i = 0; i < tensorSize; i++) {
    mul_out.push_back(fixedPointMultiply(elem_minus_mean[i], elem_minus_mean[i], 0));
  }

  // Var = ReduceMean<axes=normalized_axes>(DD)
  sum = 0;
  for (int i = 0; i < axis_dim; i++) {
    sum += mul_out[i * axis_offset];
  }
  shift = (num_elem_inv_fbits_ + inp_fbits) - inp_fbits;  // input_fbits is the output fbits
  std::int32_t Var = fixedPointMultiply(sum, num_elem_inv_, shift);

  // VarEps = Add(Var, epsilon)
  std::vector<double> epsilonVec = {epsilon};
  auto p = dataToQfp(epsilonVec, -1, 32, false);
  std::int32_t epsilonFxp = (std::int32_t)p[0].first;

  // StdDev = Sqrt(VarEps)
  std::int32_t stdev = math::sqrt(Var + epsilonFxp)

      // InvStdDev = Reciprocal(StdDev)
      std::vector<std::double>
          stdev_inv = {1.0 / stdev};
  auto s = dataToQfp(stdev_inv, -1, 32, false);
  std::int32_t stdev_inv_fxp = std::int32_t(s[0].first);
  std::int8_t stdev_inv_fbits = s[0].second;

  // Normalized = Mul(D, InvStdDev)
  std::vector<std::int32_t> norm;
  std::int8_t norm_fbits = std::min(stdev_inv_fbits, inp_fbits)
      shift = (stdev_inv_fbits + inp_fbits) - norm_fbits;
  for (int i = 0; i < tensorSize; i++) {
    norm.push_back(fixedPointMultiply(elem_minus_mean[i], stdev_inv_fxp, shift));
  }

  // NormalizedScaled = Mul(Normalized, Scale)
  // Y = Add(NormalizedScaled, B)
  std::vector<std::pair<std::int32_t, std::int8_t>> scalefxp = dataToQfp(scale_data, -1, 32, false);
  std::vector<std::pair<std::int32_t, std::int8_t>> biasfxp = dataToQfp(bias_data, -1, 32, false);

  // Output data
  auto* Y = ctx->Output(0, X->Shape());
  int32_t* yData = Y->MutableData<int32_t>();

  std::vector<std::int32_t> normalizedScaled;
  for (int i = 0; i < axis_dim; i++) {
    auto s = scalefxp[i];
    int norm_fbits = std::min(norm_fbits, p.second);
    shift = (norm_fbits, s.second) - norm_fbits;
    normalizedScaled[i] = fixedPointMultiply(norm[i], s.first, shift);

    auto b = biasfxp[i];
    shift = (norm_fbits + b.second) - out_fbits;
    yData[i] = (std::int32_t)((std::int64_t)normalizedScaled[i] + (std::int64_t)b.first);
  }

  return Status::OK();
}

}  // namespace contrib

}  // namespace onnxruntime
