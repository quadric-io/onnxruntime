// Copyright (c) Quadric, Inc. All rights reserved.
// Licensed under the MIT License.

#include "quadric_custom_op.h"
#include "core/common/common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"
#include "core/framework/session_options.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {

ONNX_OPERATOR_KERNEL_EX(QuadricCustomOp, kQuadricDomain, 1, kCpuExecutionProvider, KernelDefBuilder(), QuadricCustomOp);

QuadricCustomOp::Info::Info(const onnxruntime::Node& node, const GraphViewer& subgraph_in) : subgraph(subgraph_in), used_inputs(node.InputDefs().size(), false) {
  num_inputs = static_cast<int>(node.InputDefs().size());
  num_outputs = static_cast<int>(node.OutputDefs().size());

  auto& subgraph_inputs = subgraph.GetInputs();
  auto num_subgraph_inputs = subgraph_inputs.size();

  for (size_t i = 0; i < num_subgraph_inputs; ++i) {
    auto& input = subgraph_inputs[i];
    subgraph_input_names.insert(input->Name());
  }

  // This is commented out because we include initializers as inputs to the custom op, but
  // *NOT* the sub-graph. As a result, the number of inputs differs. Unfortunately, ORT doesn't do
  // a great job of telling us whether something is truly an initializer or not, so we can't
  // effectively check whether an input is an initializer or not.
  /*ORT_ENFORCE(num_subgraph_inputs == static_cast<size_t>(num_inputs),
              "'QuadricCustomOp' node has ", num_inputs, " inputs which doesn't match the subgraph's ",
              num_subgraph_inputs, " inputs.");
  */

  auto& subgraph_outputs = subgraph.GetOutputs();
  auto num_subgraph_outputs = subgraph_outputs.size();

  // outputs should always match up, so enforce that.
  ORT_ENFORCE(num_subgraph_outputs == static_cast<size_t>(num_outputs),
              "'QuadricCustomOp' node has ", num_outputs, " outputs which doesn't match the subgraph's ",
              num_subgraph_outputs, " outputs.");

  subgraph_output_names.reserve(num_subgraph_outputs);
  for (size_t i = 0; i < num_subgraph_outputs; ++i) {
    auto& output = subgraph_outputs[i];
    subgraph_output_names.push_back(output->Name());
  }
}

class QuadricCustomOpImpl {
 public:
  QuadricCustomOpImpl(OpKernelContextInternal& context,
                      const SessionState& session_state,
                      const QuadricCustomOp::Info& info);

  Status Initialize();
  Status Execute(const FeedsFetchesManager& ffm);

 private:
  OpKernelContextInternal& context_;
  const SessionState& session_state_;
  const QuadricCustomOp::Info& info_;

  Status AllocateOutputTensors();
  
  enum class AllocationType {
    Delayed,  // allocation of If output will be done by subgraph execution
    SubgraphOutput
  };

  // track where the fetches provided to subgraph execution were allocated.
  std::vector<std::pair<AllocationType, OrtValue>> outputs_;
};

QuadricCustomOpImpl::QuadricCustomOpImpl(OpKernelContextInternal& context,
                                         const SessionState& session_state,
                                         const QuadricCustomOp::Info& info) : context_(context),
                                                                              session_state_(session_state),
                                                                              info_(info) {}

Status QuadricCustomOpImpl::Initialize() {
  auto status = AllocateOutputTensors();
  ORT_RETURN_IF_ERROR(status);

  return Status::OK();
}

Status QuadricCustomOpImpl::AllocateOutputTensors() {
  // This function mostly copied from if.cc
  Status status = Status::OK();
  int index = 0;

  const GraphViewer& subgraph = session_state_.GetGraphViewer();
  
  const auto& graph_outputs = subgraph.GetOutputs();

  for (auto& graph_output : graph_outputs) {
    const auto* graph_output_type = graph_output->TypeAsProto();

    ORT_ENFORCE(graph_output_type->has_tensor_type() || graph_output_type->has_sequence_type(), "Only tensors or tensor sequences are supported");
    if (graph_output_type->has_tensor_type()) {
      auto* graph_output_shape = graph_output->Shape();
      bool symbolic_dim_in_shape = false;

      if (graph_output_shape) {
        TensorShape output_shape = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*graph_output_shape);

        // if size < 0 we have a symbolic dimension and need to use a temporary OrtValue in the subgraph execution
        if (output_shape.Size() < 0) {
          symbolic_dim_in_shape = true;
        } else {
          auto* tensor = context_.Output(index, output_shape);

          if (!tensor)
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for ", graph_output->Name());

          outputs_.push_back({AllocationType::SubgraphOutput, *context_.GetOutputMLValue(index)});
        }
      }

      if (!graph_output_shape || symbolic_dim_in_shape) {
        // we still need a value to put in the feeds we give to the execution frame, so just use an empty MLValue
        outputs_.push_back({AllocationType::Delayed, {}});
      }
    } else if (graph_output_type->has_sequence_type()) {
      auto* seq_tensor = context_.Output<TensorSeq>(index);
      if (!seq_tensor)
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for ", graph_output->Name());
      outputs_.push_back({AllocationType::SubgraphOutput, *context_.GetOutputMLValue(index)});
    }
    ++index;
  }

  return Status::OK();
}

Status QuadricCustomOpImpl::Execute(const FeedsFetchesManager& ffm) {
  Status status = Status::OK();

  auto num_inputs = context_.InputCount();
  std::vector<OrtValue> feeds;
  feeds.reserve(num_inputs);

  // This will contain used inputs, so some/all initializers may not be present
  for (int i = 0; i < num_inputs; ++i) {
    if(info_.used_inputs[i]) {
      feeds.push_back(*context_.GetInputMLValue(i));
    }
  }

  std::vector<OrtValue> fetches;
  std::unordered_map<size_t, IExecutor::CustomAllocator> fetch_allocators;

  fetches.reserve(info_.num_outputs);
  for (int i = 0; i < info_.num_outputs; ++i) {
    fetches.push_back(outputs_[i].second);

    if (outputs_[i].first == AllocationType::Delayed) {
      // functor to forward the allocation request from the subgraph to the If node's context so that the
      // allocation plan for the If node's output is used.
      fetch_allocators[i] = [this, i, &fetches](const TensorShape& shape, const OrtDevice& location,
                                                OrtValue& ort_value, bool& allocated) {
        // if the device the QuadricCustomOp output is allocated on does not match the required device for the subgraph output
        // we don't update the provided OrtValue and return false for 'allocated'.
        // the execution frame will allocate a buffer on the required device, and the fetches copy
        // logic in utils::ExecuteSubgraph will handle moving it into the tensor we allocated here.

        auto* tensor = context_.Output(i, shape);
        if (!tensor)
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for QuadricCustomOp output ", i);

        const OrtValue& value = *context_.GetOutputMLValue(i);

        if (tensor->Location().device == location) {
          // return OrtValue for allocated tensor
          ort_value = value;
          allocated = true;
        } else {
          // put the allocated value into fetches so the copy logic in utils::ExecuteGraphImpl can use it
          fetches[i] = value;
        }

        return Status::OK();
      };
    }
  }

  status = utils::ExecuteSubgraph(session_state_, ffm, feeds, fetches, fetch_allocators,
                                  ExecutionMode::ORT_SEQUENTIAL, context_.GetTerminateFlag(),
                                  context_.Logger(), context_.GetComputeStream());

  ORT_RETURN_IF_ERROR(status);

  return status;
}

QuadricCustomOp::QuadricCustomOp(const OpKernelInfo& info) : IControlFlowKernel(info) {
  ONNX_NAMESPACE::GraphProto proto;
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("sub_graph", &proto).IsOK());
  ORT_IGNORE_RETURN_VALUE(proto);
}

Status QuadricCustomOp::Compute(OpKernelContext* ctx) const {
  auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  auto* session_state = ctx_internal->SubgraphSessionState("sub_graph");
  ORT_ENFORCE(session_state, "Subgraph SessionState was not found for sub_graph attribute.");

  QuadricCustomOpImpl impl{*ctx_internal, *session_state, *info_};
  auto status = impl.Initialize();
  ORT_RETURN_IF_ERROR(status);

  status = impl.Execute(*feeds_fetches_manager_);

  return Status::OK();
}

Status QuadricCustomOp::SetupSubgraphExecutionInfo(const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) {
  const auto& node = Node();
  info_ = std::make_unique<QuadricCustomOp::Info>(node, subgraph_session_state.GetGraphViewer());

  const auto& subgraph_map = subgraph_session_state.GetOrtValueNameIdxMap();

  std::vector<std::string> feed_names;

  const auto& input_defs = node.InputDefs();
  for (size_t i = 0, end = info_->num_inputs; i < end; ++i) {
    const auto* input = input_defs[i];
    // Not all subgraph inputs will have names that correspond to the node's inputs. The inputs
    // that diverge like this are limited *only* to initializers and we don't need to create
    // feeds for them. Furthermore, since they are not actually used by the custom op (and
    // not even by the sub-graph since the subgraph contains its own version of initializers)
    // they end up getting removed from the graph during an optimization step and so we can't
    // prove that it's an initializer using Graph::IsInitializedTensor

    if (info_->subgraph_input_names.find(input->Name()) != info_->subgraph_input_names.end()) {
      feed_names.push_back(input->Name());
      info_->used_inputs[i] = true;
    }
  }

  std::unique_ptr<FeedsFetchesManager> ffm;
  ORT_RETURN_IF_ERROR(FeedsFetchesManager::Create(feed_names, info_->subgraph_output_names, subgraph_map, ffm));
  ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(subgraph_session_state, *ffm));

  // find the location all the feeds will be coming from
  std::vector<OrtDevice> feed_locations;
  feed_locations.resize(feed_names.size());
  for (size_t i = 0, end = feed_names.size(); i < end; ++i) {
    const auto& location = utils::FindDeviceForValue(session_state, feed_names[i]);
    feed_locations[i] = location;
  }

  std::vector<const OrtDevice*> fetch_locations;
  fetch_locations.reserve(info_->num_outputs);

  // we need the allocator info for each output from the QuadricCustomOp node
  // as the subgraph execution will write directly into those buffers
  const auto& outputs = node.OutputDefs();
  for (int i = 0, end = info_->num_outputs; i < end; ++i) {
    const auto& alloc_info = utils::FindDeviceForValue(session_state, outputs[i]->Name());
    fetch_locations.push_back(&alloc_info);
  }

  utils::FinalizeFeedFetchCopyInfo(*ffm, feed_locations, fetch_locations);

  feeds_fetches_manager_ = std::move(ffm);

  return Status::OK();
}

}  // namespace onnxruntime
