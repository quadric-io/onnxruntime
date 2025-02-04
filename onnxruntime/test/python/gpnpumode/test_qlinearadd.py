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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper import get_onnx_const, generate_normal_inputs


batch_size = 1
h=128
w=128
channels=8

class TestQLinearAdd(unittest.TestCase):
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
        print(input_shape)
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

    # def tearDown(self):
    #     # Delete the ONNX file after testing
    #     if os.path.exists(self.model_path):
    #         os.remove(self.model_path)

    def test_qlinearconv_inference(self):
        session_options = ort.SessionOptions()
        session_options.enable_gpnpu = False
        print(f"Flag enable_gpnpu: {session_options.enable_gpnpu}")

        # Create an inference session
        session1 = ort.InferenceSession(self.model_path, sess_options=session_options, providers=["CPUExecutionProvider"])
        print(f"Check flag enable_gpnpu: {session1.get_session_options().enable_gpnpu}")

        session_options.enable_gpnpu = True
        session2 = ort.InferenceSession(self.model_path, sess_options=session_options, providers=["CPUExecutionProvider"])
        print(f"Check flag enable_gpnpu: {session2.get_session_options().enable_gpnpu}")

        # Get information about both inputs
        input_a_info = session1.get_inputs()[0]
        # input_b_info = session.get_inputs()[1]

        print(f"Model input names: {input_a_info.name}")
        print(f"Model input shapes: {input_a_info.shape}")

        # Create random INT8 data matching the input shapes
        shape_tuple_a = tuple(dim if isinstance(dim, int) else 1 for dim in input_a_info.shape)

        # Generate random data for both inputs
        x_data_a = np.random.randint(
            low=-128, high=128, size=shape_tuple_a, dtype=np.int8
        )

        # Create input dictionary with both inputs
        input_dict = {
            input_a_info.name: x_data_a
        }

        # Run inference
        output_name1 = session1.get_outputs()[0].name
        print(f"Process ID: {os.getpid()}")
        output_data1 = session1.run([output_name1], input_dict)[0]
        output_name2 = session2.get_outputs()[0].name
        print(f"Process ID: {os.getpid()}")
        output_data2 = session2.run([output_name2], input_dict)[0]

        # Print shapes and types
        print(f"Input A data shape: {x_data_a.shape}, dtype: {x_data_a.dtype}")
        # print(f"Output data shape: {output_data1.shape}, dtype: {output_data1.dtype}")
        print("Output data (truncated):\n", output_data1.flatten()[:50], "...\n")
        print("Output data (truncated):\n", output_data2.flatten()[:50], "...\n")
        # print("hi")
        difference = output_data1 - output_data2
        max_diff = np.max(np.abs(difference))
        print(max_diff)

        difference = output_data1 - output_data2

        max_diff = np.max(np.abs(difference))

        # Check the output shape and type
        self.assertEqual(output_data1.shape, (batch_size, channels, h, w))
        self.assertEqual(output_data1.dtype, np.int8)
        self.assertLessEqual(max_diff, 1)

if __name__ == '__main__':
    unittest.main()
