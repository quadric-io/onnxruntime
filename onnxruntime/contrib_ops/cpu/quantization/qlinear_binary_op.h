// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class QLinearAdd final : public OpKernel {
 public:
  QLinearAdd(const OpKernelInfo& info) : OpKernel(info) {
  gpnpu_flag_ = (info.GetConfigOptions().GetConfigOrDefault(kOrtSessionOptionsGpnpuMode, "0") == "1");
  }

  Status Compute(OpKernelContext* context) const override;

  private:
    bool gpnpu_flag_{false};
};

template <typename T>
class QLinearMul final : public OpKernel {
 public:
  QLinearMul(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
