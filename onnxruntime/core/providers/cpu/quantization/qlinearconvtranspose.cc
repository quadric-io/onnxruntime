/**
* Copyright (c) 2014-present, Quadric, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#include "core/mlas/inc/mlas.h"
#include "core/common/safeint.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/common/inlined_containers_fwd.h"
#include "core/framework/transpose_helper.h"
#include "core/providers/utils.h"
#include "core/framework/tensorprotoutils.h"

#include "core/providers/cpu/nn/conv_transpose_attributes.h"

namespace onnxruntime {

template <typename ActType>
class QLinearConvTranspose : public OpKernel {
 public:
  explicit QLinearConvTranspose(const OpKernelInfo& info) : OpKernel(info), conv_transpose_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 protected:
  Status DoConvTranspose(OpKernelContext* context) const;

 private:

  static float ComputeOutputScale(OpKernelContext* context) {
    const Tensor* X_scale = context->Input<Tensor>(InputTensors::IN_X_SCALE);
    const Tensor* W_scale = context->Input<Tensor>(InputTensors::IN_W_SCALE);
    const Tensor* Y_scale = context->Input<Tensor>(InputTensors::IN_Y_SCALE);
    ORT_ENFORCE(IsScalarOr1ElementVector(X_scale),
                "QLinearConv : input scale must be a scalar or 1D tensor of size 1");
    ORT_ENFORCE(IsScalarOr1ElementVector(Y_scale),
                "QLinearConv : result scale must be a scalar or 1D tensor of size 1");
    ORT_ENFORCE(IsScalarOr1ElementVector(W_scale),
                "QLinearConv : filter scale must be a scalar or 1D tensor of size 1");

    auto X_scale_value = *(X_scale->Data<float>());
    auto Y_scale_value = *(Y_scale->Data<float>());
    auto W_scale_value = *(W_scale->Data<float>());

    return X_scale_value * W_scale_value / Y_scale_value;
  }

  enum InputTensors : int {
    IN_X = 0,
    IN_X_SCALE = 1,
    IN_X_ZERO_POINT = 2,
    IN_W = 3,
    IN_W_SCALE = 4,
    IN_W_ZERO_POINT = 5,
    IN_Y_SCALE = 6,
    IN_Y_ZERO_POINT = 7,
    IN_BIAS = 8

  };
  enum OutputTensors : int {
    OUT_Y = 0
  };

  ConvTransposeAttributes conv_transpose_attrs_;

  // for pre-packing usage
  TensorShape filter_shape_;
  BufferUniquePtr transposed_filter_;
};

template <typename T>
Status QLinearConvTranspose<T>::PrePack(const Tensor& /*tensor*/, int /*input_idx*/, AllocatorPtr /*alloc*/,
                                        /*out*/ bool& is_packed,
                                        /*out*/ PrePackedWeights* /*prepacked_weights*/
) {
  is_packed = false;
  return Status::OK();
}

template <typename T>
Status QLinearConvTranspose<T>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& /*prepacked_buffers*/,
                                                          int /*input_idx*/,
                                                          /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;
  return Status::OK();
}

template <typename T>
Status QLinearConvTranspose<T>::Compute(OpKernelContext* context) const {
  return QLinearConvTranspose<T>::DoConvTranspose(context);
}

template <typename T>
Status QLinearConvTranspose<T>::DoConvTranspose(OpKernelContext* context) const {
  typedef int32_t ActType;
  size_t num_inputs = OpKernel::Node().InputDefs().size();
  ConvTransposeAttributes::Prepare p;
  bool has_bias = num_inputs == 9;
  ORT_RETURN_IF_ERROR(conv_transpose_attrs_.PrepareForCompute(context, has_bias, p, false, nullptr, true));

  const int64_t input_image_size = p.input_shape.Size();

  // Bail out early if one of the dimensions is zero.
  if (p.Y->Shape().Size() == 0) {
    return Status::OK();
  }

  // Quantization parameters, only support symmetric quant for now
  float scale_value = ComputeOutputScale(context);

  const int64_t X_offset = p.num_input_channels / conv_transpose_attrs_.group * input_image_size;
  const int64_t Y_offset = p.Y->Shape().Size() / p.Y->Shape()[0] / conv_transpose_attrs_.group;
  const int64_t W_offset = p.F->Shape().Size() / conv_transpose_attrs_.group;
  const int64_t kernel_size = TensorShape(p.kernel_shape).Size();
  const int64_t kernel_dim = p.num_output_channels / conv_transpose_attrs_.group * kernel_size;
  const int64_t output_size = (p.Y->Shape().Slice(2)).Size();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  const int64_t col_buffer_size = kernel_dim * p.input_shape.Size();
  auto col_data = alloc->Alloc(SafeInt<size_t>(sizeof(ActType)) * col_buffer_size);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(alloc));

  // Pre-transpose the weight matrix because MatMul does not take transpose params
  const int64_t trans_filt_size = p.num_input_channels / conv_transpose_attrs_.group * kernel_dim;
  auto trans_filt_data = alloc->Alloc(SafeInt<size_t>(sizeof(T)) * trans_filt_size);
  BufferUniquePtr trans_filt(trans_filt_data, BufferDeleter(alloc));
  ActType* col_buffer_data = static_cast<ActType*>(col_buffer.get());

  const T* Xdata = p.X->Data<T>();
  const T* filter_data = p.F->Data<T>();
  T* Ydata = p.Y->MutableData<T>();
  TensorShape output_shape = p.Y->Shape().Slice(2);
  MlasTranspose(
      filter_data,
      static_cast<T*>(trans_filt.get()),
      p.num_input_channels / conv_transpose_attrs_.group,
      kernel_dim);

  //Compute the GEMM in int32 for now, MlassGemm requires uint8 input and MlasSymmQgemmBatch is ARM only
  //TODO: investigate ways of offseting the data with a virtual zero point
  auto inp_i32_buffer = alloc->Alloc(SafeInt<size_t>(sizeof(ActType)) * p.X->Shape().Size());
  BufferUniquePtr inp_i32(inp_i32_buffer, BufferDeleter(alloc));
  ActType* inp_i32_data = static_cast<ActType*>(inp_i32.get());
  for(std::int64_t i = 0; i < p.X->Shape().Size(); i++){
    inp_i32_data[i] = Xdata[i];
  }
  auto wt_i32_buffer = alloc->Alloc(SafeInt<size_t>(sizeof(ActType)) * p.F->Shape().Size());
  BufferUniquePtr wt_i32(wt_i32_buffer, BufferDeleter(alloc));
  ActType* wt_i32_data = static_cast<ActType*>(wt_i32.get());
  for(std::int64_t i = 0; i < p.F->Shape().Size(); i++){
    wt_i32_data[i] = static_cast<T*>(trans_filt.get())[i];
  }
  auto out_i32_buffer = alloc->Alloc(SafeInt<size_t>(sizeof(ActType)) * p.Y->Shape().Size());
  BufferUniquePtr out_i32(out_i32_buffer, BufferDeleter(std::move(alloc)));
  ActType* out_i32_data = static_cast<ActType*>(out_i32.get());

  for (auto image_id = 0; image_id < p.N; ++image_id) {
    for (int group_id = 0; group_id < conv_transpose_attrs_.group; ++group_id) {
      math::MatMul(kernel_dim, input_image_size, p.num_input_channels / conv_transpose_attrs_.group, 
             wt_i32_data + group_id * W_offset, 
             inp_i32_data + group_id * X_offset, 
             col_buffer_data, 
             nullptr);

      // Col2im is only registered for float, though I see no reason for that
      // TODO: use this nasty hack for now, then change col2im to
      // typed registration and add support for int32_t
      math::Col2im<float, CPUMathUtil, StorageOrder::NCHW>(
          reinterpret_cast<const float *>(col_buffer_data),
          p.num_output_channels / conv_transpose_attrs_.group,
          p.Y->Shape()[2],
          p.Y->Shape()[3],
          p.kernel_shape[0],
          p.kernel_shape[1],
          p.dilations[0],
          p.dilations[1],
          p.pads[0],
          p.pads[1],
          p.pads[2],
          p.pads[3],
          p.strides[0],
          p.strides[1],
          reinterpret_cast<float *>(out_i32_data + group_id * Y_offset),
          &CPUMathUtil::Instance());
    }

    if (p.B != nullptr) {
      auto out_i32_matrix = EigenMatrixMap<ActType>(out_i32_data, output_size, p.num_output_channels);
      auto Bvec = ConstEigenVectorMap<ActType>(p.B->Data<ActType>(), p.num_output_channels);
      out_i32_matrix.rowwise() += Bvec.transpose();
    }

    MlasRequantizeOutput(out_i32_data,
                        p.Y->Shape()[2] * p.Y->Shape()[3],
                        Ydata,
                        p.Y->Shape()[2] * p.Y->Shape()[3],
                        nullptr,
                        &scale_value,
                        false,
                        (T)0,
                        0,
                        0,
                        p.num_output_channels,
                        p.Y->Shape()[2] * p.Y->Shape()[3]);
    Xdata += X_offset * conv_transpose_attrs_.group;
    Ydata += Y_offset * conv_transpose_attrs_.group;
  }

  return Status::OK();
}

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {

// Register the operator with int8 inputs and weights
ONNX_OPERATOR_KERNEL_EX(
    QLinearConvTranspose,
    kMSDomain,
    1,
    //int8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConvTranspose<int8_t>);

}  // namespace contrib
#endif

}  // namespace onnxruntime
