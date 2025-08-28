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
import glob
from onnx import version_converter


N, C, H, W = 1, 2048, 7, 7

class TestQLinearGAP(unittest.TestCase):
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
            opset_imports=[helper.make_opsetid("com.microsoft", 1), helper.make_opsetid('', 12)]  # Operator set version
        )

        # Step 6: Save the model to file
        onnx.save(model, output_model_path)

    def tearDown(self):
        # Delete the ONNX file after testing
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def accuracy_test(self, num_iterations=100):
        for _ in range(num_iterations):
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
            x_data_a = np.random.randint(
                low=-128, high=128, size=shape_tuple_a, dtype=np.int8
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

    def test_accuracy(self):
        # Run test
        self.accuracy_test(num_iterations=1)

if __name__ == '__main__':
    unittest.main()
