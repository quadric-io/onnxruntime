#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxruntime
from onnxruntime.tools import symbolic_shape_infer


def order_repeated_field(repeated_proto, key_name, order):
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))

def make_node(op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs):
    node = helper.make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
    if doc_string == '':
        node.doc_string = ''
    order_repeated_field(node.attribute, 'name', kwargs.keys())
    return node

def make_graph(*args, doc_string=None, **kwargs):
    graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if doc_string == '':
        graph.doc_string = ''
    return graph

class TestQLinearOpsShapeInfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="test_quant_shape_infer.")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def get_model(self):
        # Model containing all QLinear operators
        return helper.make_model(
            opset_imports=[
                helper.make_operatorsetid('', 12),
                helper.make_operatorsetid('com.microsoft.experimental', 1),
                helper.make_operatorsetid('ai.onnx.ml', 3),
                helper.make_operatorsetid('ai.onnx.training', 1),
                helper.make_operatorsetid('com.microsoft', 1),
                helper.make_operatorsetid('com.ms.internal.nhwc', 18),
                helper.make_operatorsetid('ai.onnx.preview.training', 1),
                helper.make_operatorsetid('com.microsoft.nchwc', 1),
                helper.make_operatorsetid('org.pytorch.aten', 1),
            ],
            ir_version=4,
            producer_name='onnx.quantize',
            producer_version='0.1.0',
            graph=make_graph(
                name='torch-jit-export',
                inputs=[helper.make_tensor_value_info('input.1', TensorProto.FLOAT, shape=[1, 8, 14, 14])],
                outputs=[helper.make_tensor_value_info('dense_out', TensorProto.FLOAT, shape=[1, 32])],
                initializer=[
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='add_out_zero_point'),
                    numpy_helper.from_array(np.array(0.015748031437397003, dtype='float32'), name='add_out_scale'),
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='input.1_zero_point'),
                    numpy_helper.from_array(np.array(0.007874015718698502, dtype='float32'), name='input.1_scale'),
                    numpy_helper.from_array(np.array(0.007874015718698502, dtype='float32'), name='add_bias_scale'),
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='add_bias_zero_point'),
                    numpy_helper.from_array(np.ones([8]).astype('int8'), name='add_bias_quantized'),
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='mul_out_zero_point'),
                    numpy_helper.from_array(np.array(0.007874015718698502, dtype='float32'), name='mul_out_scale'),
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='concat_out_zero_point'),
                    numpy_helper.from_array(np.array(0.015748031437397003, dtype='float32'), name='concat_out_scale'),
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='relu_out_zero_point'),
                    numpy_helper.from_array(np.array(0.015748031437397003, dtype='float32'), name='relu_out_scale'),
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='pool_out_zero_point'),
                    numpy_helper.from_array(np.array(0.015748031437397003, dtype='float32'), name='pool_out_scale'),
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='sigmoid_out_zero_point'),
                    numpy_helper.from_array(np.array(0.00693541020154953, dtype='float32'), name='sigmoid_out_scale'),
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='add_out_2_zero_point'),
                    numpy_helper.from_array(np.array(0.014809425920248032, dtype='float32'), name='add_out_2_scale'),
                    numpy_helper.from_array(np.array(0.007874015718698502, dtype='float32'), name='shape_checkpont_1_wt_scale'),
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='shape_checkpont_1_wt_zero_point'),
                    numpy_helper.from_array(np.ones([1, 16, 7, 7]).astype('int8'), name='shape_checkpont_1_wt_quantized'),
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='gap_out_zero_point'),
                    numpy_helper.from_array(np.array(0.014809425920248032, dtype='float32'), name='gap_out_scale'),
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='dense_out_zero_point'),
                    numpy_helper.from_array(np.array(0.2353924661874771, dtype='float32'), name='dense_out_scale'),
                    numpy_helper.from_array(np.array(0.007874015718698502, dtype='float32'), name='dense_wt_scale'),
                    numpy_helper.from_array(np.array(0, dtype='int8'), name='dense_wt_zero_point'),
                    numpy_helper.from_array(np.ones([32, 16]).astype('int8'), name='dense_wt_quantized'),
                    numpy_helper.from_array(np.ones([32]).astype('int32'), name='dense_bias_quantized'),
                ],
                value_info=[
                    helper.make_tensor_value_info('add_out', TensorProto.FLOAT, shape=[1, 8, 14, 14]),
                    helper.make_tensor_value_info('mul_out', TensorProto.FLOAT, shape=[1, 8, 14, 14]),
                    helper.make_tensor_value_info('concat_out', TensorProto.FLOAT, shape=[1, 16, 14, 14]),
                    helper.make_tensor_value_info('relu_out', TensorProto.FLOAT, shape=[1, 16, 14, 14]),
                    helper.make_tensor_value_info('pool_out', TensorProto.FLOAT, shape=[1, 16, 7, 7]),
                    helper.make_tensor_value_info('sigmoid_out', TensorProto.FLOAT, shape=[1, 16, 7, 7]),
                    helper.make_tensor_value_info('add_out_2', TensorProto.FLOAT, shape=[1, 16, 7, 7]),
                    helper.make_tensor_value_info('gap_out', TensorProto.FLOAT, shape=[1, 16, 1, 1]),
                    helper.make_tensor_value_info('flatten_out', TensorProto.FLOAT, shape=[1, 16]),
                    helper.make_tensor_value_info('dense_out', TensorProto.FLOAT, shape=[1, 32]),
                ],
                nodes=[
                    make_node('QuantizeLinear', inputs=['input.1', 'input.1_scale', 'input.1_zero_point'], outputs=['input.1_quantized'], name='input.1_QuantizeLinear'),
                    make_node(
                        'QLinearAdd',
                        inputs=['input.1_quantized', 'input.1_scale', 'input.1_zero_point', 'add_bias_quantized', 'add_bias_scale', 'add_bias_zero_point', 'add_out_scale', 'add_out_zero_point'],
                        outputs=['add_out_quantized'],
                        name='20_quant',
                        domain='com.microsoft',
                    ),
                    make_node(
                        'QLinearMul',
                        inputs=['add_bias_quantized', 'add_bias_scale', 'add_bias_zero_point', 'input.1_quantized', 'input.1_scale', 'input.1_zero_point', 'mul_out_scale', 'mul_out_zero_point'],
                        outputs=['mul_out_quantized'],
                        name='21_quant',
                        domain='com.microsoft',
                    ),
                    make_node(
                        'QLinearConcat',
                        inputs=['concat_out_scale', 'concat_out_zero_point', 'add_out_quantized', 'add_out_scale', 'add_out_zero_point', 'mul_out_quantized', 'mul_out_scale', 'mul_out_zero_point'],
                        outputs=['concat_out_quantized'],
                        name='22_quant',
                        domain='com.microsoft',
                        axis=1,
                    ),
                    make_node(
                        'QLinearLeakyRelu',
                        inputs=['concat_out_quantized', 'concat_out_scale', 'concat_out_zero_point', 'relu_out_scale', 'relu_out_zero_point'],
                        outputs=['relu_out_quantized'],
                        name='23_quant',
                        domain='com.microsoft',
                        alpha=0.009999999776482582,
                    ),
                    make_node(
                        'QLinearAveragePool',
                        inputs=['relu_out_quantized', 'relu_out_scale', 'relu_out_zero_point', 'pool_out_scale', 'pool_out_zero_point'],
                        outputs=['pool_out_quantized'],
                        name='24_quant',
                        domain='com.microsoft',
                        kernel_shape=[2, 2],
                        strides=[2, 2],
                    ),
                    make_node(
                        'QLinearSigmoid',
                        inputs=['pool_out_quantized', 'pool_out_scale', 'pool_out_zero_point', 'sigmoid_out_scale', 'sigmoid_out_zero_point'],
                        outputs=['sigmoid_out_quantized'],
                        name='25_quant',
                        domain='com.microsoft',
                    ),
                    make_node(
                        'QLinearAdd',
                        inputs=[
                            'sigmoid_out_quantized',
                            'sigmoid_out_scale',
                            'sigmoid_out_zero_point',
                            'shape_checkpont_1_wt_quantized',
                            'shape_checkpont_1_wt_scale',
                            'shape_checkpont_1_wt_zero_point',
                            'add_out_2_scale',
                            'add_out_2_zero_point',
                        ],
                        outputs=['add_out_2_quantized'],
                        name='26_quant',
                        domain='com.microsoft',
                    ),
                    make_node(
                        'QLinearGlobalAveragePool',
                        inputs=['add_out_2_quantized', 'add_out_2_scale', 'add_out_2_zero_point', 'gap_out_scale', 'gap_out_zero_point'],
                        outputs=['gap_out_quantized'],
                        name='27_quant',
                        domain='com.microsoft',
                        channels_last=0,
                    ),
                    make_node('Flatten', inputs=['gap_out_quantized'], outputs=['flatten_out_quantized'], name='28'),
                    make_node(
                        'QGemm',
                        inputs=['flatten_out_quantized', 'gap_out_scale', 'gap_out_zero_point', 'dense_wt_quantized', 'dense_wt_scale', 'dense_wt_zero_point', 'dense_bias_quantized', 'dense_out_scale', 'dense_out_zero_point'],
                        outputs=['dense_out_quantized'],
                        name='29_quant',
                        domain='com.microsoft',
                        alpha=1.0,
                        transB=1,
                    ),
                    make_node('DequantizeLinear', inputs=['dense_out_quantized', 'dense_out_scale', 'dense_out_zero_point'], outputs=['dense_out'], name='dense_out_DequantizeLinear'),
                ],
            ),
        )

    def test_qlinear_ops_shape_infer(self):
        # Test shape inference for the QLinear operators. Check the output shape for correctness, if
        # shapes for intermediate operators are not correct they will trigger asserts due to shape
        # mismatch in following operations such as Add
        model = self.get_model()

        inf_onnx = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(
            in_mp=model,
            auto_merge=True,
            int_max=100000,
            guess_output_rank=True,
        )

        out_shape_proto = inf_onnx.graph.value_info[-1].type.tensor_type.shape
        out_shape = [sh.dim_value for sh in out_shape_proto.dim]
        self.assertEqual(out_shape, [1, 32], "Wrong shape infered for quantized network output")

if __name__ == "__main__":
    unittest.main()
