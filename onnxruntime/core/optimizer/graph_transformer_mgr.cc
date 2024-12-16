// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/rule_based_graph_transformer.h"

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

common::Status GraphTransformerManager::SetSteps(unsigned steps) {
  steps_ = steps;
  return Status::OK();
}

common::Status GraphTransformerManager::GetSteps(unsigned& steps) const {
  steps = steps_;
  return Status::OK();
}

common::Status GraphTransformerManager::ApplyTransformers(Graph& graph, TransformerLevel level,
                                                          const logging::Logger& logger) const {
  const auto& transformers = level_to_transformer_map_.find(level);
  if (transformers == level_to_transformer_map_.end()) {
    return Status::OK();
  }

  for (unsigned step = 0; step < steps_; ++step) {
    bool graph_changed = false;
    for (const auto& transformer : transformers->second) {
      if (step > 0 && transformer->ShouldOnlyApplyOnce())
        continue;

      bool modified = false;
      ORT_RETURN_IF_ERROR(transformer->Apply(graph, modified, logger));
    
    std::cerr << "Start graph search" << std::endl;
    for (const auto& node : graph.Nodes()) {
        if (node.Name() == "DequantizeLinear_1012" or node.Name() == "DequantizeLinear_1009") {
            std::cerr << node.Name() << std::endl;
        }
        if (node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type() == 2) {
            std::cerr << node.Name() << std::endl;
        }
    }
    std::cerr << "End graph search" << std::endl;


      graph_changed = graph_changed || modified;
    }
    if (!graph_changed) {
      break;
    }
  }

  return Status::OK();
}

common::Status GraphTransformerManager::Register(std::unique_ptr<GraphTransformer> transformer,
                                                 TransformerLevel level) {
  const auto& name = transformer->Name();
  if (transformers_info_.find(name) != transformers_info_.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "This transformer is already registered " + name);
  }

  transformers_info_[name] = transformer.get();
  level_to_transformer_map_[level].push_back(std::move(transformer));
  return Status::OK();
}
}  // namespace onnxruntime
