// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/controlflow/utils.h"

#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/session/onnxruntime_cxx_api.h"
#include <vector>
#include <unordered_set>

namespace onnxruntime {

struct QuadricCustomOp : public controlflow::IControlFlowKernel {
  QuadricCustomOp(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  virtual Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                            const std::string& attribute_name,
                                            const SessionState& subgraph_session_state) override;

  struct Info {
    Info(const onnxruntime::Node& node, const GraphViewer& subgraph_in);
    const GraphViewer& subgraph;

    int num_inputs;
    int num_outputs;

    std::unordered_set<std::string> subgraph_input_names;
    std::vector<bool> used_inputs;
    std::vector<std::string> subgraph_output_names;
  };
  
 private:
  std::unique_ptr<Info> info_;
  std::unique_ptr<FeedsFetchesManager> feeds_fetches_manager_;
};

}  // namespace onnxruntime
