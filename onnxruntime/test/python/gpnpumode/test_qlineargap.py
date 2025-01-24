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
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper import get_onnx_const, generate_normal_inputs

N, C, H, W = 1, 2, 3, 3

class TestQGemm(unittest.TestCase):
    def setUp(self):
        # Create a specific ONNX model with a single QGemm node
        self.model_path = "qlineargap.onnx"
        self.create_qgemm_model(self.model_path)

    def create_qgemm_model(self, output_model_path):
        # Define the quantization parameters for X
        x_scale = 0.1
        x_zero_point = 128

        # Create tensor for input X (quantized data)
        X = helper.make_tensor_value_info("X", TensorProto.INT8, [N, C, H, W])
        Y = helper.make_tensor_value_info("Y", TensorProto.INT8, [N, C, 1, 1])


        # Define quantization parameters for output Y
        y_scale = 0.2
        y_zero_point = 128

        # Step 2: Create the QLinearGlobalAveragePool node
        node = helper.make_node(
            'QLinearGlobalAveragePool',  # Operator name
            inputs=['X', 'x_scale', 'x_zero_point', 'y_scale', 'y_zero_point'],  # Input tensors
            outputs=['Y'],  # Output tensor
            channels_last=0,  # Attribute indicating whether the channels are last in the shape (1 = True)
            domain="com.microsoft"
        )

        # Step 3: Define the scale and zero point tensors for input/output
        x_scale_tensor = helper.make_tensor('x_scale', TensorProto.FLOAT, [1], [x_scale])
        x_zero_point_tensor = helper.make_tensor('x_zero_point', TensorProto.INT8, [1], [x_zero_point])
        y_scale_tensor = helper.make_tensor('y_scale', TensorProto.FLOAT, [1], [y_scale])
        y_zero_point_tensor = helper.make_tensor('y_zero_point', TensorProto.INT8, [1], [y_zero_point])

        # Step 4: Create the graph (composed of the node and input/output tensors)
        graph = helper.make_graph(
            [node],  # List of nodes (here, just our QLinearGlobalAveragePool node)
            'QLinearGlobalAveragePoolModel',  # Name of the graph
            [X],  # Inputs
            [Y],
            initializer=[x_scale_tensor, x_zero_point_tensor, y_scale_tensor, y_zero_point_tensor]
        )

        # Step 5: Create the model (version 1)
        model = helper.make_model(
            graph,
            producer_name='onnx-example',
            opset_imports=[helper.make_opsetid('', 1)]  # Operator set version
        )

        # Step 6: Save the model to file
        onnx.save(model, output_model_path)

    def tearDown(self):
        # Delete the ONNX file after testing
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

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

        # print(f"Model input names: {input_a_info.name}")
        # print(f"Model input shapes: {input_a_info.shape}")

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
        # print(f"Process ID: {os.getpid()}")
        t1 = time.time()
        output_data1 = session1.run([output_name1], input_dict)[0]
        t2 = time.time()
        output_name2 = session2.get_outputs()[0].name
        # print(f"Process ID: {os.getpid()}")
        t3 = time.time()
        output_data2 = session2.run([output_name2], input_dict)[0]
        t4 = time.time()

        print("CPU  ", t2-t1)
        print("GPNPU", t4-t3)

        # Print shapes and types
        print(f"Input A data shape: {x_data_a.shape}, dtype: {x_data_a.dtype}")
        print(f"Output data shape: {output_data1.shape}, dtype: {output_data1.dtype}")
        # print("Output data (truncated):\n", output_data1.flatten()[:50], "...\n")
        # print("Output data (truncated):\n", output_data2.flatten()[:50], "...\n")
        # print("hi")
        difference = output_data1 - output_data2
        max_diff = np.max(np.abs(difference))
        print(max_diff)

        difference = output_data1 - output_data2

        max_diff = np.max(np.abs(difference))

        # Check the output shape and type
        self.assertEqual(output_data1.shape, (N,C,1,1)) # only N C dims, rest are 1
        self.assertEqual(output_data1.dtype, np.int8)
        self.assertLessEqual(max_diff, 1)

if __name__ == '__main__':
    unittest.main()
