#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import os
import sys
import glob

# allows imports from directory of file
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import generate_normal_inputs, get_onnx_const

batch_size = 1
h=16
w=32
channels=2048

class TestQLinearConv(unittest.TestCase):
    def setUp(self):
        # Create a specific ONNX model with a single QLinearConv node
        self.model_path = "qlinearadd_model.onnx"
        self.create_qlinearadd_model(self.model_path)

    def create_qlinearadd_model(self, output_model_path):
        a_scale, a_zero_point = 0.2039528638124466, -14
        b_scale, b_zero_point = 0.003937007859349251, 0
        y_scale, y_zero_point = 0.1019764319062233, -6

        # Create input shapes
        input_shape = [batch_size, channels, h, w]

        # Define node names
        input_a_name = "input_a"
        input_a_scale_name = "input_a_scale"
        input_a_zp_name = "input_a_zero_point"
        input_b_name = "input_b"
        input_b_scale_name = "input_b_scale"
        input_b_zp_name = "input_b_zero_point"
        output_scale_name = "output_scale"
        output_zp_name = "output_zero_point"
        output_name = "output"

        a_sc = get_onnx_const(input_a_scale_name, a_scale)
        a_zp = get_onnx_const(input_a_zp_name, a_zero_point)
        b_sc = get_onnx_const(input_b_scale_name, b_scale)
        b_zp = get_onnx_const(input_b_zp_name, b_zero_point)
        y_sc = get_onnx_const(output_scale_name, y_scale)
        y_zp = get_onnx_const(output_zp_name, y_zero_point)

        # Create QLinearAdd node
        qlinear_add_node = onnx.helper.make_node(
            "QLinearAdd",
            inputs=[
                input_a_name, input_a_scale_name, input_a_zp_name,
                input_b_name, input_b_scale_name, input_b_zp_name,
                output_scale_name, output_zp_name
            ],
            outputs=[output_name],
            domain="com.microsoft"
        )

        # Define input and output tensors
        input_a_tensor = helper.make_tensor_value_info(input_a_name, TensorProto.INT8, input_shape)
        b = get_onnx_const(input_b_name, generate_normal_inputs(input_shape, np.int8, 0, 32))
        input_b_tensor = helper.make_tensor_value_info(input_b_name, TensorProto.INT8, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.INT8, input_shape)

        # Create graph
        graph_name = "com.microsoft.QLinearAdd_test"
        graph = helper.make_graph(
            [qlinear_add_node],
            graph_name,
            [input_a_tensor],
            [output_tensor],
            initializer=[a_sc, a_zp, b, b_sc, b_zp, y_sc, y_zp],
        )

        # Create model
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.microsoft", 1),helper.make_opsetid("", 12)])
        model.ir_version = 8  # use stable onnx ir version

        # Save model
        onnx.checker.check_model(model, True)
        onnx.save(model, output_model_path)

    def tearDown(self):
        # Delete the ONNX file after testing
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_accuracy(self):
        # CPU Session
        session_options_cpu = ort.SessionOptions()
        session_options_cpu.enable_gpnpu = False
        session_cpu = ort.InferenceSession(
            self.model_path,
            sess_options=session_options_cpu,
            providers=["CPUExecutionProvider"]
        )

        # GPNPU Session
        session_options_gpnpu = ort.SessionOptions()
        session_options_gpnpu.enable_gpnpu = True
        session_gpnpu = ort.InferenceSession(
            self.model_path,
            sess_options=session_options_gpnpu,
            providers=["CPUExecutionProvider"]
        )

        # Prepare input
        input_a_info = session_cpu.get_inputs()[0]
        shape_tuple_a = tuple(dim if isinstance(dim, int) else 1 for dim in input_a_info.shape)
        info = np.iinfo(np.int8)
        min_val = info.min
        max_val = info.max
        x_data_a = np.random.randint(
            low=min_val, high=max_val, size=shape_tuple_a, dtype=np.int8
        )
        input_dict = {input_a_info.name: x_data_a}

        # Run CPU inference
        output_cpu = session_cpu.run(
            [session_cpu.get_outputs()[0].name],
            input_dict
        )[0]

        # Run GPNPU inference
        output_gpnpu = session_gpnpu.run(
            [session_gpnpu.get_outputs()[0].name],
            input_dict
        )[0]

        # Calculate max difference
        max_diff = np.max(np.abs(output_cpu - output_gpnpu))

        self.assertLessEqual(max_diff, 1)

if __name__ == '__main__':
    unittest.main()
