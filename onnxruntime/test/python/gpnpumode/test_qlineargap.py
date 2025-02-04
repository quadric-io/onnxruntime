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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper import json_to_df, load_json

N, C, H, W = 1, 2048, 7, 7

class TestQGemm(unittest.TestCase):
    def setUp(self):
        # Create a specific ONNX model with a single QGemm node
        self.model_path = "qlineargap.onnx"
        self.create_qgemm_model(self.model_path)
        self.cpu_jsons = []
        self.gpnpu_jsons = []

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

    # def tearDown(self):
    #     # Delete the ONNX file and JSON files after testing
    #     if os.path.exists(self.model_path):
    #         os.remove(self.model_path)
    #     for json_file in glob.glob("*.json"):
    #         os.remove(json_file)

    def performance_and_accuracy_test(self, num_iterations=100):
        for _ in range(num_iterations):
            # CPU Session
            session_options_cpu = ort.SessionOptions()
            session_options_cpu.enable_gpnpu = False
            session_options_cpu.enable_profiling = True
            session_options_cpu.profile_file_prefix = "cpu"
            session_cpu = ort.InferenceSession(
                self.model_path,
                sess_options=session_options_cpu,
                providers=["CPUExecutionProvider"]
            )

            # GPNPU Session
            session_options_gpnpu = ort.SessionOptions()
            session_options_gpnpu.enable_gpnpu = True
            session_options_gpnpu.enable_profiling = True
            session_options_gpnpu.profile_file_prefix = "gpnpu"
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

            # Time and run CPU inference
            output_cpu = session_cpu.run(
                [session_cpu.get_outputs()[0].name],
                input_dict
            )[0]
            json_name_cpu = session_cpu.end_profiling()
            self.cpu_jsons.append(json_name_cpu)

            # Time and run GPNPU inference
            output_gpnpu = session_gpnpu.run(
                [session_gpnpu.get_outputs()[0].name],
                input_dict
            )[0]
            json_name_gpnpu = session_gpnpu.end_profiling()
            self.gpnpu_jsons.append(json_name_gpnpu)

            # Calculate max difference
            max_diff = np.max(np.abs(output_cpu - output_gpnpu))
            print(max_diff)

            self.assertLessEqual(max_diff, 1)

    def test_performance_and_accuracy(self):
        # Run test
        self.performance_and_accuracy_test(num_iterations=10)
        self.json_time_profiling()

    def json_time_profiling(self):
        def get_time(jsons):
            times = []
            for json in jsons:
                cpu_df, gpu_df = json_to_df(load_json(json), lambda x: True)
                times.extend(cpu_df[cpu_df['name'] == 'QLinearGlobalAveragePool']['duration'].values)
            return np.mean(np.array(times)), np.std(np.array(times))
        cpu_mean_time, cpu_std_time = get_time(self.cpu_jsons)
        gpnpu_mean_time, gpnpu_std_time = get_time(self.gpnpu_jsons)
        print(f"CPU Time:   {cpu_mean_time:8.3f} ± {cpu_std_time:.3f} ms")
        print(f"GPNPU Time: {gpnpu_mean_time:8.3f} ± {gpnpu_std_time:.3f} ms")


if __name__ == '__main__':
    unittest.main()
