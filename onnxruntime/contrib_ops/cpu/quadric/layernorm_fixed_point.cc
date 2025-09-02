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

struct LayernormFixedPointAttrs {
  std::int64_t axis;
  float epsilon;
  std::int64_t stash_type;
  std::int64_t wt_fbits;
  std::int64_t bias_fbits;
  explicit LayernormFixedPointAttrs(const OpKernelInfo& info) {
    auto status = info.GetAttr<std::int64_t>("axis", &axis);
    if (!status.IsOK())
      axis = -1; /* Default axis for layernorm is 1 */
    status = info.GetAttr<float>("epsilon", &epsilon);
    if (!status.IsOK())
      epsilon = 1e-05f; /* Default epsilon value is 1e-05. */
    status = info.GetAttr<std::int64_t>("stash_type", &stash_type);
    if (!status.IsOK())
      stash_type = 1; /* Default stash_type if 1. */
    status = info.GetAttr<std::int64_t>("wt_fbits", &wt_fbits);
    if (!status.IsOK())
      wt_fbits = 31; /* weight frac bits. */
    status = info.GetAttr<std::int64_t>("bias_fbits", &bias_fbits);
    if (!status.IsOK())
      bias_fbits = 31; /* bias frac bits. */
  }

  ~LayernormFixedPointAttrs() = default;
};

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
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())    // Input tensor
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())    // Input frac bits
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>())     // Scale
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>())     // bias
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int8_t>())    // Output frac bits
        .TypeConstraint("T5", DataTypeImpl::GetTensorType<int32_t>()),  // Output
    LayernormFixedPoint);

void calculateLayernormWidth(const std::int32_t* inp_data, std::int32_t* out_data, TensorShapeVector inpShape, std::int8_t inpFbits, std::int8_t outFbits, std::int32_t epsilonQFp, std::vector<std::int32_t> gamma = {}, std::vector<std::int32_t> beta = {}, std::int32_t wtFbits = 31, std::int32_t bFbits = 31) {
  std::int64_t elemInBatch = inpShape[0];
  std::int64_t elemInChannel = inpShape[1];
  std::int64_t eleminHW = 0;
  // collapse the 4D shapes into 3D if needed.
  if (inpShape.size() == 3)
    eleminHW = inpShape[2];
  else
    eleminHW = inpShape[2] * inpShape[3];

  size_t tensor_size = elemInBatch * elemInChannel * eleminHW;

  std::int32_t logCols = chimera::log2Ceil(eleminHW);
  std::int32_t squareFracBits = std::max(2 * inpFbits - 31, 0);
  std::int32_t inputMinusMeanFracBits = std::max(inpFbits - 1, 0);
  std::int32_t normFracBits = 30 - (logCols / 2);

  std::int32_t dataSquaredIntBits = 31 - squareFracBits;
  std::int32_t dataSquaredFracBits = 2 * inpFbits;
  std::int32_t SumSquareIntBits = dataSquaredIntBits + logCols;
  std::int32_t SumSquareFracBits = 63 - SumSquareIntBits;
  std::int8_t dataSquaredShift = dataSquaredFracBits - SumSquareFracBits;

  std::vector<std::int64_t> SumElem(elemInBatch * elemInChannel);    // vector of sum in Width
  std::vector<std::int64_t> SumSquare(elemInBatch * elemInChannel);  // vector of sumSquare in Width
  std::int64_t DataInt64 = 0;

  for (int b = 0; b < elemInBatch; b++) {
    for (int ch = 0; ch < elemInChannel; ch++) {
      std::int64_t Sum = 0, sumSquare = 0;
      std::int64_t base_index = b * elemInChannel * eleminHW + ch * eleminHW;
      for (int hw = 0; hw < eleminHW; hw++) {
        DataInt64 = static_cast<std::int64_t>(inp_data[base_index + hw]);
        Sum += DataInt64;
        sumSquare += (DataInt64 * DataInt64) >> dataSquaredShift;
      }
      SumElem[b * elemInChannel + ch] = Sum;
      SumSquare[b * elemInChannel + ch] = sumSquare;
    }
  }

  // Mean
  std::vector<std::int64_t> mean(elemInBatch * elemInChannel);
  std::vector<std::int32_t> mean32(elemInBatch * elemInChannel);
  for (int i = 0; i < elemInBatch * elemInChannel; i++) {
    mean[i] = SumElem[i] / static_cast<std::int64_t>(eleminHW);
    mean32[i] = static_cast<std::int32_t>(mean[i]);
  }

  // Mean square
  std::vector<std::int64_t> meanSquare(elemInBatch * elemInChannel);
  for (int i = 0; i < elemInBatch * elemInChannel; i++) {
    meanSquare[i] = SumSquare[i] / static_cast<std::int64_t>(eleminHW);
  }

  std::int32_t Mean2FracBits = 2 * inpFbits;
  std::int32_t MeanSquareShift = Mean2FracBits - SumSquareFracBits;

  std::vector<std::int64_t> variance(elemInBatch * elemInChannel);
  // meanSumSquare and variance has same number of fracbits as square
  for (int i = 0; i < elemInBatch * elemInChannel; i++) {
    std::int64_t meanSumSquareShifted = meanSquare[i] << MeanSquareShift;
    variance[i] = meanSumSquareShifted - (mean[i] * mean[i]);
  }

  std::int8_t VarianceShift = 31;
  std::int8_t epsilonFbits = 31;
  // Standard deviation
  std::vector<std::int32_t> stdDev(elemInBatch * elemInChannel);

  for (int i = 0; i < elemInBatch * elemInChannel; i++) {
    std::int32_t varInt32 = std::int32_t(variance[i] >> VarianceShift);
    std::int32_t intermediate = varInt32 + (epsilonQFp >> (epsilonFbits - squareFracBits));
    stdDev[i] = chimera::sqrt(intermediate, squareFracBits, inpFbits);
  }
  // layer norm: X_norm = (Xi - mean) / std
  std::int32_t inpMinusMeanFbits = inpFbits - 1;
  std::int8_t normShift = normFracBits - (inputMinusMeanFracBits - inpFbits);
  std::vector<std::int32_t> norm(elemInBatch * elemInChannel * eleminHW);

  for (std::int64_t index = 0; index < tensor_size; index++) {
    std::int64_t b_index = index / (elemInChannel * eleminHW);
    std::int64_t ch_index = (index % (elemInChannel * eleminHW)) / eleminHW;
    std::int64_t mean_stddev_index = b_index * elemInChannel + ch_index;
    std::int32_t distance = (inp_data[index] >> (inpFbits - inpMinusMeanFbits)) - (mean32[mean_stddev_index] >> (inpFbits - inputMinusMeanFracBits));
    norm[index] = chimera::fixedPointDiv(distance, stdDev[mean_stddev_index], normShift);
  }

  // fractional bits for gammaMul and betaAdd
  std::int32_t shift = normFracBits + wtFbits - outFbits;
  std::int32_t biasShift = bFbits - outFbits;

  for (std::int64_t index = 0; index < tensor_size; index++) {
    std::int64_t w = index % eleminHW;
    std::int32_t gammaMul = chimera::fixedPointMultiply(norm[index], gamma[w], shift);
    out_data[index] = gammaMul + (beta[w] >> biasShift);
  }
}

void calculateLayernormChannel(const std::int32_t* inp_data, std::int32_t* out_data, TensorShapeVector inpShape, std::int8_t inpFbits, std::int8_t outFbits, std::int32_t epsilonQFp, std::vector<std::int32_t> gamma = {}, std::vector<std::int32_t> beta = {}, std::int32_t wtFbits = 31, std::int32_t bFbits = 31) {
  std::int64_t elemInBatch = inpShape[0];
  std::int64_t elemInChannel = inpShape[1];
  std::int64_t eleminHW = 0;
  // collapse the 4D shapes into 3D if needed.
  if (inpShape.size() == 3)
    eleminHW = inpShape[2];
  else
    eleminHW = inpShape[2] * inpShape[3];

  size_t tensor_size = elemInBatch * elemInChannel * eleminHW;

  std::int32_t logChn = chimera::log2Ceil(elemInChannel);
  std::int32_t squareFracBits = std::max(2 * inpFbits - 31, 0);
  std::int32_t inputMinusMeanFracBits = std::max(inpFbits - 1, 0);
  std::int32_t normFracBits = 30 - (logChn / 2);

  std::int32_t dataSquaredIntBits = 31 - squareFracBits;
  std::int32_t dataSquaredFracBits = 2 * inpFbits;
  std::int32_t SumSquareIntBits = dataSquaredIntBits + logChn;
  std::int32_t SumSquareFracBits = 63 - SumSquareIntBits;
  std::int8_t dataSquaredShift = dataSquaredFracBits - SumSquareFracBits;

  std::vector<std::int64_t> SumElem(elemInBatch * eleminHW);    // vector of sum in Width
  std::vector<std::int64_t> SumSquare(elemInBatch * eleminHW);  // vector of sumSquare in Width
  std::int64_t DataInt64 = 0;

  for (int b = 0; b < elemInBatch; b++) {
    for (int hw = 0; hw < eleminHW; hw++) {
      std::int64_t Sum = 0, sumSquare = 0;

      for (std::int64_t index = 0; index < elemInChannel; index++) {
        DataInt64 = static_cast<std::int64_t>(inp_data[index * eleminHW + hw]);
        Sum += DataInt64;
        sumSquare += (DataInt64 * DataInt64) >> dataSquaredShift;
      }
      SumElem[b * eleminHW + hw] = Sum;
      SumSquare[b * eleminHW + hw] = sumSquare;
    }
  }

  // Mean
  std::vector<std::int64_t> mean(elemInBatch * eleminHW);
  std::vector<std::int32_t> mean32(elemInBatch * eleminHW);
  for (int i = 0; i < elemInBatch * eleminHW; i++) {
    mean[i] = SumElem[i] / static_cast<std::int64_t>(elemInChannel);
    mean32[i] = static_cast<std::int32_t>(mean[i]);
  }

  // Mean square
  std::vector<std::int64_t> meanSquare(elemInBatch * eleminHW);
  for (int i = 0; i < elemInBatch * eleminHW; i++) {
    meanSquare[i] = SumSquare[i] / static_cast<std::int64_t>(elemInChannel);
  }

  std::int32_t Mean2FracBits = 2 * inpFbits;
  std::int32_t MeanSquareShift = Mean2FracBits - SumSquareFracBits;

  std::vector<std::int64_t> variance(elemInBatch * eleminHW);
  // meanSumSquare and variance has same number of fracbits as square
  for (int i = 0; i < elemInBatch * eleminHW; i++) {
    std::int64_t meanSumSquareShifted = meanSquare[i] << MeanSquareShift;
    variance[i] = meanSumSquareShifted - (mean[i] * mean[i]);
  }

  std::int8_t VarianceShift = 31;
  std::int8_t epsilonFbits = 31;
  // Standard deviation
  std::vector<std::int32_t> stdDev(elemInBatch * eleminHW);

  for (int i = 0; i < elemInBatch * eleminHW; i++) {
    std::int32_t varInt32 = std::int32_t(variance[i] >> VarianceShift);
    std::int32_t intermediate = varInt32 + (epsilonQFp >> (epsilonFbits - squareFracBits));
    stdDev[i] = chimera::sqrt(intermediate, squareFracBits, inpFbits);
  }
  // layer norm: X_norm = (Xi - mean) / std
  std::int32_t inpMinusMeanFbits = inpFbits - 1;
  std::int8_t normShift = normFracBits - (inputMinusMeanFracBits - inpFbits);
  std::vector<std::int32_t> norm(elemInBatch * elemInChannel * eleminHW);

  for (std::int64_t index = 0; index < tensor_size; index++) {
    std::int64_t b_index = index / (elemInChannel * eleminHW);
    std::int64_t hw_index = index % eleminHW;
    std::int64_t mean_stddev_index = b_index * eleminHW + hw_index;
    std::int32_t distance = (inp_data[index] >> (inpFbits - inpMinusMeanFbits)) - (mean32[mean_stddev_index] >> (inpFbits - inputMinusMeanFracBits));
    norm[index] = chimera::fixedPointDiv(distance, stdDev[mean_stddev_index], normShift);
  }

  // fractional bits for gammaMul and betaAdd
  std::int32_t shift = normFracBits + wtFbits - outFbits;
  std::int32_t biasShift = bFbits - outFbits;

  for (std::int64_t index = 0; index < tensor_size; index++) {
    std::int32_t gamma_index = (index / eleminHW) % elemInChannel;
    std::int32_t gammaMul = chimera::fixedPointMultiply(norm[index], gamma[gamma_index], shift);
    out_data[index] = gammaMul + (beta[gamma_index] >> biasShift);
  }
}

Status LayernormFixedPoint::Compute(OpKernelContext* ctx) const {
  const Tensor* inp = ctx->Input<Tensor>(0);
  const Tensor* inpFracBitsTensor = ctx->Input<Tensor>(1);
  const Tensor* inpScale = ctx->Input<Tensor>(2);
  const Tensor* bias = ctx->Input<Tensor>(3);
  const Tensor* outFracBits = ctx->Input<Tensor>(4);

  // Validate inputs
  ORT_ENFORCE(inp != nullptr, "Input is null");
  ORT_ENFORCE(inpFracBitsTensor != nullptr, "inpFracBits is null");
  ORT_ENFORCE(inpScale != nullptr, "inpScale is null");
  ORT_ENFORCE(bias != nullptr, "bias is null");
  ORT_ENFORCE(outFracBits != nullptr, "outFracBits is null");

  // input, scale, bias data
  const std::int32_t* inp_data = inp->Data<std::int32_t>();
  const float* wt_data = inpScale->Data<float>();
  const float* bias_data = bias->Data<float>();

  std::int64_t axis = layer_norm__fxp_attrs_.axis;
  float epsilon = layer_norm__fxp_attrs_.epsilon;
  std::int32_t wtFbits = std::int32_t(layer_norm__fxp_attrs_.wt_fbits);
  std::int32_t bFbits = std::int32_t(layer_norm__fxp_attrs_.bias_fbits);

  // check for valid input shapes
  TensorShapeVector inpShape = ToShapeVector(inp->Shape().GetDims());
  const auto inp_rank = inpShape.size();
  const std::int8_t inpFbits = *(inpFracBitsTensor->Data<std::int8_t>());
  const std::int8_t outFbits = *(outFracBits->Data<std::int8_t>());

  // converting for negative axis.
  if (axis < 0) {
    axis += inp_rank;
  };

  ORT_ENFORCE(axis >= 0 && axis < inp_rank, "Allowed axis range is [-r, r), input_rank ", inp_rank, " axis ", layer_norm__fxp_attrs_.axis);

  // Allocate output tensor
  auto* out = ctx->Output(0, inp->Shape());
  std::int32_t* out_data = out->MutableData<int32_t>();
  // quantize scale(gamma) and bias(beta) tensor
  std::vector<double> scale(wt_data, wt_data + inpShape[axis]);
  auto qs = dataToQfp(scale, wtFbits, 32, false);
  std::vector<std::int32_t> quantScale = qs.first;

  std::vector<double> biasTensor(bias_data, bias_data + inpShape[axis]);
  auto qb = dataToQfp(biasTensor, bFbits, 32, false);
  std::vector<std::int32_t> quantBias = qb.first;

  // convert epsilon to fixed-point
  std::vector<double> epsilonVec = {epsilon};
  auto e = dataToQfp(epsilonVec, -1, 32, false);
  std::int32_t epsilonQFp = static_cast<int32_t>(e.first[0]);

  if (axis == 1) {
    calculateLayernormChannel(inp_data, out_data, inpShape, inpFbits, outFbits, epsilonQFp, quantScale, quantBias, wtFbits, bFbits);
  } else if (axis == 2) {
    calculateLayernormWidth(inp_data, out_data, inpShape, inpFbits, outFbits, epsilonQFp, quantScale, quantBias, wtFbits, bFbits);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid Layernorm axes. Only width and channels allowed. axis should be 1 or 2 got", axis);
  }
  return Status::OK();
}

}  // namespace contrib

}  // namespace onnxruntime
