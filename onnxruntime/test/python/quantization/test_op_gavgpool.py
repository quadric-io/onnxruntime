#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static  # noqa: F401


class TestOpGlobalAveragePool(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_gavgpool(self, output_model_path, input_shape, weight_shape, output_shape):
        #      (input)
        #         |
        #  GlobalAveragePool
        #         |
        #       Expand
        #         |
        #        Conv
        #         |
        #  GlobalAveragePool
        #         |
        #      (output)
        input_name = "input"
        expand_input = "expand_input"
        conv_input = "conv_input"
        gavgpool_input_2nd = "gavgpool_input"
        output_name = "output"
        initializers = []

        # make 1st GlobalAveragePool node
        gavgpool_node_1 = onnx.helper.make_node("GlobalAveragePool", [input_name], [expand_input])

        # make Expand node
        expand_shape_name = "expand_shape"
        initializers.append(onnx.numpy_helper.from_array(np.array(input_shape, dtype=np.int64), name=expand_shape_name))
        expand_node = onnx.helper.make_node("Expand", [expand_input, expand_shape_name], [conv_input])

        # make Conv node
        weight_name = "conv_weight"
        conv_name = "conv_node"
        conv_weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(conv_weight_data, name=weight_name))
        conv_node = onnx.helper.make_node("Conv", [conv_input, weight_name], [gavgpool_input_2nd], name=conv_name)

        # make 1st GlobalAveragePool node
        gavgpool_node_2 = onnx.helper.make_node("GlobalAveragePool", [gavgpool_input_2nd], [output_name])

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        graph_name = "GAveragePool_test"
        graph = helper.make_graph(
            [gavgpool_node_1, expand_node, conv_node, gavgpool_node_2],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def quantize_gavgpool_test(self, activation_type, weight_type, extra_options={}):  # noqa: B006
        np.random.seed(1)
        model_fp32_path = "gavg_pool_fp32.onnx"
        data_reader = self.input_feeds(1, {"input": [1, 8, 33, 33]})
        self.construct_model_gavgpool(model_fp32_path, [1, 8, 33, 33], [16, 8, 3, 3], [1, 16, 1, 1])

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_q8_path = f"gavg_pool_{activation_type_str}{weight_type_str}.onnx"

        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_q8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )

        quant_nodes = {
            "QLinearConv": 1,
            "GlobalAveragePool": 0,
            "QLinearGlobalAveragePool": 2,
            "QuantizeLinear": 2,
            "DequantizeLinear": 2,
        }
        check_op_type_count(self, model_q8_path, **quant_nodes)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        qnode_io_qtypes.update(
            {
                "QLinearGlobalAveragePool": [
                    ["i", 2, activation_proto_qtype],
                    ["i", 4, activation_proto_qtype],
                ]
            }
        )
        check_qtype_by_node_type(self, model_q8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_q8_path, data_reader.get_next())

    def test_quantize_gavgpool(self):
        self.quantize_gavgpool_test(QuantType.QUInt8, QuantType.QUInt8, extra_options={})

    def test_quantize_gavgpool_s8s8(self):
        self.quantize_gavgpool_test(
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )


if __name__ == "__main__":
    unittest.main()
