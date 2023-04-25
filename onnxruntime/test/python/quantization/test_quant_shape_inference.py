#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import tempfile
import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from onnxruntime.tools import symbolic_shape_infer


class TestQLinearOpsShapeInfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="test_quant_shape_infer.")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def get_model(self, node, input_shape, initializer):
        # Create a single node model
        return helper.make_model(
            opset_imports=[
                helper.make_operatorsetid("", 12),
                helper.make_operatorsetid("com.microsoft", 1)
            ],
            graph=helper.make_graph(
                name="qlinear_test",
                inputs=[helper.make_tensor_value_info("input", TensorProto.INT8, shape=input_shape)],
                outputs=[helper.make_tensor_value_info("output", TensorProto.INT8, shape=[])],
                initializer=initializer,
                value_info=[],
                nodes=[node]
            ),
        )

    def infer_out_shape(self, model):
        inf_onnx = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(
            in_mp=model,
            auto_merge=True,
            int_max=100000,
            guess_output_rank=True,
        )
        out_shape_proto = inf_onnx.graph.value_info[-1].type.tensor_type.shape
        return [sh.dim_value for sh in out_shape_proto.dim]

    def test_shape_qlinear_add(self):
        model = self.get_model(
            helper.make_node(
                "QLinearAdd",
                inputs=[
                    "input",
                    "input_scale",
                    "input_zero_point",
                    "add_bias_quantized",
                    "add_bias_scale",
                    "add_bias_zero_point",
                    "add_out_scale",
                    "add_out_zero_point",
                ],
                outputs=["output"],
                name="quant_node",
                domain="com.microsoft"),
            [1, 8, 14, 14],
            [
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="input_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="input_zero_point"),
                numpy_helper.from_array(np.ones([8, 1, 1]).astype("int8"), name="add_bias_quantized"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="add_bias_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="add_bias_zero_point"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="add_out_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="add_out_zero_point"),
            ]

        )
        self.assertEqual(self.infer_out_shape(model),
                         [1, 8, 14, 14],
                         "Wrong shape infered for quantized network output")

    def test_shape_qlinear_mult(self):
        model = self.get_model(
            helper.make_node(
                "QLinearMul",
                inputs=[
                    "mul_bias_quantized",
                    "mul_bias_scale",
                    "mul_bias_zero_point",
                    "input",
                    "input_scale",
                    "input_zero_point",
                    "mul_out_scale",
                    "mul_out_zero_point",
                ],
                outputs=["output"],
                name="quant_node",
                domain="com.microsoft"),
            [1, 8, 14, 14],
            [
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="input_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="input_zero_point"),
                numpy_helper.from_array(np.ones([8, 1, 1]).astype("int8"), name="mul_bias_quantized"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="mul_bias_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="mul_bias_zero_point"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="mul_out_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="mul_out_zero_point"),
            ]
        )
        self.assertEqual(self.infer_out_shape(model),
                         [1, 8, 14, 14],
                         "Wrong shape infered for quantized network output")

    def test_shape_qlinear_concat(self):
        model = self.get_model(
            helper.make_node(
                "QLinearConcat",
                inputs=[
                    "concat_out_scale",
                    "concat_out_zero_point",
                    "input",
                    "input_scale",
                    "input_zero_point",
                    "concat_bias_quantized",
                    "concat_out_scale",
                    "concat_out_zero_point",
                ],
                outputs=["output"],
                name="quant_node",
                domain="com.microsoft",
                axis=1,
            ),
            [1, 8, 14, 14],
            [
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="input_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="input_zero_point"),
                numpy_helper.from_array(np.ones([1, 8, 14, 14]).astype("int8"), name="concat_bias_quantized"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="concat_bias_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="concat_bias_zero_point"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="concat_out_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="concat_out_zero_point"),
            ]
        )
        self.assertEqual(self.infer_out_shape(model),
                         [1, 16, 14, 14],
                         "Wrong shape infered for quantized network output")

    def test_shape_qlinear_leaky_relu(self):
        model = self.get_model(
            helper.make_node(
                "QLinearLeakyRelu",
                inputs=[
                    "input",
                    "input_scale",
                    "input_zero_point",
                    "relu_out_scale",
                    "relu_out_zero_point",
                ],
                outputs=["output"],
                name="quant_node",
                domain="com.microsoft",
                alpha=0.009999999776482582,
            ),
            [1, 16, 14, 14],
            [
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="input_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="input_zero_point"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="relu_out_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="relu_out_zero_point"),
            ]
        )
        self.assertEqual(self.infer_out_shape(model),
                         [1, 16, 14, 14],
                         "Wrong shape infered for quantized network output")

    def test_shape_qlinear_average_pool(self):
        model = self.get_model(
            helper.make_node(
                "QLinearAveragePool",
                inputs=[
                    "input",
                    "input_scale",
                    "input_zero_point",
                    "pool_out_scale",
                    "pool_out_zero_point",
                ],
                outputs=["output"],
                name="quant_node",
                domain="com.microsoft",
                kernel_shape=[2, 2],
                strides=[2, 2],
            ),
            [1, 16, 14, 14],
            [
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="input_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="input_zero_point"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="pool_out_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="pool_out_zero_point"),
            ]
        )
        self.assertEqual(self.infer_out_shape(model),
                         [1, 16, 7, 7],
                         "Wrong shape infered for quantized network output")

    def test_shape_qlinear_sigmoid(self):
        model = self.get_model(
            helper.make_node(
                "QLinearSigmoid",
                inputs=[
                    "input",
                    "input_scale",
                    "input_zero_point",
                    "sigmoid_out_scale",
                    "sigmoid_out_zero_point",
                ],
                outputs=["output"],
                name="quant_node",
                domain="com.microsoft",
            ),
            [1, 16, 7, 7],
            [
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="input_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="input_zero_point"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="sigmoid_out_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="sigmoid_out_zero_point"),
            ]
        )
        self.assertEqual(self.infer_out_shape(model),
                         [1, 16, 7, 7],
                         "Wrong shape infered for quantized network output")

    def test_shape_qlinear_global_average_pool(self):
        model = self.get_model(
            helper.make_node(
                "QLinearGlobalAveragePool",
                inputs=[
                    "input",
                    "input_scale",
                    "input_zero_point",
                    "gap_out_scale",
                    "gap_out_zero_point",
                ],
                outputs=["output"],
                name="quant_node",
                domain="com.microsoft",
                channels_last=0,
            ),
            [1, 16, 7, 7],
            [
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="input_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="input_zero_point"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="gap_out_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="gap_out_zero_point"),
            ]
        )
        self.assertEqual(self.infer_out_shape(model),
                         [1, 16, 1, 1],
                         "Wrong shape infered for quantized network output")

    def test_shape_qlinear_gemm(self):
        model = self.get_model(
            helper.make_node(
                "QGemm",
                inputs=[
                    "input",
                    "input_scale",
                    "input_zero_point",
                    "gemm_wt_quantized",
                    "gemm_wt_scale",
                    "gemm_wt_zero_point",
                    "gemm_bias_quantized",
                    "gemm_out_scale",
                    "gemm_out_zero_point",
                ],
                outputs=["output"],
                name="quant_node",
                domain="com.microsoft",
                alpha=1.0,
                transB=1,
            ),
            [1, 16],
            [
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="input_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="input_zero_point"),
                numpy_helper.from_array(np.ones([32, 16]).astype("int8"), name="gemm_wt_quantized"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="gemm_wt_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="gemm_wt_zero_point"),
                numpy_helper.from_array(np.ones([32]).astype("int32"), name="gemm_bias_quantized"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="gemm_out_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="gemm_out_zero_point"),
            ]
        )
        self.assertEqual(self.infer_out_shape(model),
                         [1, 32],
                         "Wrong shape infered for quantized network output")

    def test_shape_qlinear_softmax(self):
        model = self.get_model(
            helper.make_node(
                "QLinearSoftmax",
                inputs=[
                    "input",
                    "input_scale",
                    "input_zero_point",
                    "softmax_out_scale",
                    "softmax_out_zero_point",
                ],
                outputs=["output"],
                name="quant_node",
                domain="com.microsoft",
            ),
            [1, 32],
            [
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="input_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="input_zero_point"),
                numpy_helper.from_array(np.array(0.007874015718698502, dtype="float32"), name="softmax_out_scale"),
                numpy_helper.from_array(np.array(0, dtype="int8"), name="softmax_out_zero_point"),
            ]
        )
        self.assertEqual(self.infer_out_shape(model),
                         [1, 32],
                         "Wrong shape infered for quantized network output")

if __name__ == "__main__":
    unittest.main()
