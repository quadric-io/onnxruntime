#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include <cmath>   // For log2()
#include <limits>  // For int8_t min/max
#include <iostream>
#include <iomanip>  // For std::setprecision
#include "core/mlas/inc/mlas.h"
#include "fixed_point.h"

namespace onnxruntime {

namespace contrib {

struct LayernormFixedPointAttrs {
  std::int64_t axis;
  float epsilon;
  std::int64_t stash_type;
  std::int64_t gbFbits;
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
    status = info.GetAttr<std::int64_t>("gbFbits", &gbFbits);
    if (!status.IsOK())
      gbFbits = 31; /* gamma beta frac bits. */
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

template <typename shapeType>
void calculateLayernormWidth(const std::int32_t* inp_data, std::int32_t* out_data, shapeType inpShape, std::int8_t inpFbits, std::int8_t outFbits, std::int32_t epsilonQFp, std::vector<std::int32_t> gamma = {}, std::vector<std::int32_t> beta = {}, std::int8_t gbFbits = 31) {
  // Frac bits for mean and stdDev is equal to input fbits.
  std::int8_t squareFracBits = std::max(2 * inpFbits - 31, 0);

  std::int64_t elemInChannel = inpShape[0];
  std::int64_t elemInHeight = inpShape[1];
  std::int64_t eleminWidth = inpShape[2];

  std::vector<std::int64_t> SumElem(elemInChannel * elemInHeight);    // vector of sum in Width
  std::vector<std::int64_t> SumSquare(elemInChannel * elemInHeight);  // vector of sumSquare in Width
  std::int64_t DataInt64 = 0;

  for (int ch = 0; ch < elemInChannel; ch++) {
    for (int h = 0; h < elemInHeight; h++) {
      std::int64_t Sum = 0, sumSquare = 0;
      for (int wid = 0; wid < eleminWidth; wid++) {
        DataInt64 = static_cast<std::int64_t>(inp_data[ch * elemInHeight * eleminWidth + h * eleminWidth + wid]);
        Sum += DataInt64;
        sumSquare += (DataInt64 * DataInt64) >> inpFbits;
      }
      SumElem.push_back(Sum);
      SumSquare.push_back(sumSquare);
    }
  }

  std::int8_t reciprocalFbits = 31;
  std::int32_t reciprocalOfaxis = 1.0 / eleminWidth;

  // Mean
  std::vector<std::int32_t> mean(elemInChannel * elemInHeight);
  for (int i = 0; i < elemInChannel * elemInHeight; i++) {
    mean.push_back(static_cast<std::int32_t>(SumElem[i] * static_cast<std::int64_t>(reciprocalOfaxis) >> reciprocalFbits));
  }

  std::int8_t shiftBitsCount = inpFbits + reciprocalFbits - squareFracBits;
  std::vector<std::int32_t> variance(elemInChannel * elemInHeight);
  // meanSumSquare and variance has same number of fracbits as square
  for (int i = 0; i < elemInChannel * elemInHeight; i++) {
    std::int32_t meanSumSquare = static_cast<std::int32_t>(SumSquare[i] * static_cast<std::int64_t>(reciprocalOfaxis) >> shiftBitsCount);
    variance.push_back(meanSumSquare - (mean[i] * mean[i]));
  }

  std::int8_t epsilonFbits = 31;
  // Standard deviation
  std::vector<std::int32_t> stdDev(elemInChannel * elemInHeight);
  for (int i = 0; i < elemInChannel * elemInHeight; i++) {
    stdDev.push_back(sqrt(variance[i] + (epsilonQFp >> (epsilonFbits - inpFbits))));
  }
  // layer norm: X_norm = (Xi - mean) / std
  std::int32_t inpMinusMeanFbits = inpFbits - 1;
  std::int8_t normFracBits = 30 - std::int8_t((log2Ceil(eleminWidth) / 2));

  for (int ch = 0; ch < elemInChannel; ch++) {
    for (int h = 0; h < elemInHeight; h++) {
      for (int w = 0; w < eleminWidth; w++) {
        std::int32_t distance = inp_data[ch * elemInHeight * eleminWidth + h * eleminWidth + w] - mean[ch * elemInHeight + h];
        out_data[ch * elemInHeight * eleminWidth + h * eleminWidth + w] = std::int32_t((std::int64_t(distance) << (normFracBits - (inpMinusMeanFbits - inpFbits))) / std::int64_t(stdDev[ch * elemInHeight + h]));
      }
    }
  }

  if (gamma.size() != 0 && beta.size() != 0) {
    // fractional bits for gammaMul and betaAdd
    std::int8_t outMinusOneFracBits = outFbits - 1;
    std::int8_t gammaMulFracBits = std::min(outFbits, gbFbits) - 1;
    std::int8_t betaAddFracBits = std::min(outMinusOneFracBits, gbFbits);

    for (int ch = 0; ch < elemInChannel; ch++) {
      for (int h = 0; h < elemInHeight; h++) {
        for (int w = 0; w < eleminWidth; w++) {
          std::int8_t shift = normFracBits + gbFbits - gammaMulFracBits;
          std::int32_t gammaMul = fixedPointMultiply(out_data[ch * elemInHeight * eleminWidth + h * eleminWidth + w], gamma[w], shift);
          // betaAddFracBits is always less than gammaMulFracBits
          out_data[ch * elemInHeight * eleminWidth + h * eleminWidth + w] = std::int32_t((std::int64_t(gammaMul) << (gammaMulFracBits - betaAddFracBits)) + std::int64_t(beta[w]));
        }
      }
    }
  }
}

template <typename shapeType>
void calculateLayerNormChannel(const std::int32_t* inp_data, std::int32_t* out_data, shapeType inpShape, std::int8_t inpFbits, std::int8_t outFbits, std::int32_t epsilonQFp, std::vector<std::int32_t> gamma = {}, std::vector<std::int32_t> beta = {}, std::int8_t gbFbits = 31) {
  // Frac bits for mean and stdDev is equal to input fbits.
  std::int8_t squareFracBits = std::max(2 * inpFbits - 31, 0);

  std::int64_t elemInChn = inpShape[0];
  std::int64_t elemInHeightWidth = inpShape[1] * inpShape[2];
  std::vector<std::int64_t> SumElem(elemInHeightWidth);    // vector of sum in channels
  std::vector<std::int64_t> SumSquare(elemInHeightWidth);  // vector of sumSquare in channels
  std::int64_t DataInt64 = 0;

  for (int i = 0; i < elemInHeightWidth; i++) {
    std::int64_t Sum = 0, sumSquare = 0;
    for (int ch = 0; ch < elemInChn; ch++) {
      DataInt64 = static_cast<std::int64_t>(inp_data[ch * elemInHeightWidth + i]);
      Sum += DataInt64;
      sumSquare += (DataInt64 * DataInt64) >> inpFbits;
    }
    SumElem.push_back(Sum);
    SumSquare.push_back(sumSquare);
  }

  std::int8_t reciprocalFbits = 31;
  std::int32_t reciprocalOfaxis = 1.0 / elemInChn;

  // Mean
  std::vector<std::int32_t> mean(elemInHeightWidth);
  for (int i = 0; i < elemInHeightWidth; i++) {
    mean.push_back(static_cast<std::int32_t>(SumElem[i] * (static_cast<std::int64_t>(reciprocalOfaxis) >> reciprocalFbits)));
  }

  std::int8_t shiftBitsCount = inpFbits + reciprocalFbits - squareFracBits;
  std::vector<std::int32_t> variance(elemInHeightWidth);
  // meanSumSquare and variance has same number of fracbits as square
  for (int i = 0; i < elemInHeightWidth; i++) {
    std::int32_t meanSumSquare = static_cast<std::int32_t>(SumSquare[i] * (static_cast<std::int64_t>(reciprocalOfaxis) >> shiftBitsCount));
    variance.push_back(meanSumSquare - (mean[i] * mean[i]));
  }

  std::int8_t epsilonFbits = 31;

  // Standard deviation
  std::vector<std::int32_t> stdDev(elemInHeightWidth);
  for (int i = 0; i < elemInHeightWidth; i++) {
    stdDev.push_back(sqrt(variance[i] + (epsilonQFp >> (epsilonFbits - inpFbits))));
  }
  // layer norm: X_norm = (Xi - mean) / std
  std::int32_t inpMinusMeanFbits = inpFbits - 1;
  std::int8_t normFracBits = 30 - std::int8_t((log2Ceil(elemInChn) / 2));

  for (int ch = 0; ch < elemInChn; ch++) {
    for (int i = 0; i < elemInHeightWidth; i++) {
      std::int32_t distance = inp_data[ch * elemInHeightWidth + i] - mean[i];
      out_data[ch * elemInHeightWidth + i] = std::int32_t((std::int64_t(distance) << (normFracBits - (inpMinusMeanFbits - inpFbits))) / std::int64_t(stdDev[i]));
    }
  }

  if (gamma.size() != 0 && beta.size() != 0) {
    // fractional bits for gammaMul and betaAdd
    std::int8_t outMinusOneFracBits = outFbits - 1;
    std::int8_t gammaMulFracBits = std::min(outFbits, gbFbits) - 1;
    std::int8_t betaAddFracBits = std::min(outMinusOneFracBits, gbFbits);

    for (int ch = 0; ch < elemInChn; ch++) {
      for (int i = 0; i < elemInHeightWidth; i++) {
        std::int8_t shift = normFracBits + gbFbits - gammaMulFracBits;
        std::int32_t gammaMul = fixedPointMultiply(out_data[ch * elemInHeightWidth + i], gamma[ch], shift);
        // betaAddFracBits is always less than gammaMulFracBits
        out_data[ch * elemInHeightWidth + i] = std::int32_t((std::int64_t(gammaMul) << (gammaMulFracBits - betaAddFracBits)) + std::int64_t(beta[ch]));
      }
    }
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
  const float* scale_data = inpScale->Data<float>();
  const float* bias_data = bias->Data<float>();

  std::int64_t axis = layer_norm__fxp_attrs_.axis;
  float epsilon = layer_norm__fxp_attrs_.epsilon;
  std::int8_t gbFbits = std::int8_t(layer_norm__fxp_attrs_.gbFbits);

  // check for valid input shapes
  const auto inpShape = inp->Shape().GetDims();
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
  std::vector<double> scale(scale_data, scale_data + inpShape[axis]);
  auto qs = dataToQfp(scale, gbFbits, 32, false);
  std::vector<std::int32_t> quantScale = qs.first;

  std::vector<double> biasTensor(bias_data, bias_data + inpShape[axis]);
  auto qb = dataToQfp(biasTensor, gbFbits, 32, false);
  std::vector<std::int32_t> quantBias = qb.first;

  // convert epsilon to fixed-point
  std::vector<double> epsilonVec = {epsilon};
  auto e = dataToQfp(epsilonVec, -1, 32, false);
  std::int32_t epsilonQFp = static_cast<int32_t>(e.first[0]);

  if (axis == 1) {
    calculateLayerNormChannel(inp_data, out_data, inpShape, inpFbits, outFbits, epsilonQFp, quantScale, quantBias, gbFbits);
  } else if (axis == 2) {
    calculateLayernormWidth(inp_data, out_data, inpShape, inpFbits, outFbits, epsilonQFp, quantScale, quantBias, gbFbits);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid Layernorm axes. Only width and channels allowed. axis should be 1 or 2 got", axis);
  }
  return Status::OK();
}

}  // namespace contrib

}  // namespace onnxruntime
