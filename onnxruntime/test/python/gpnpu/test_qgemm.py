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

from helper_onnx import get_onnx_const
from helper import generate_normal_inputs

m = 1
k = 2024
n = 1000

class TestQGemm(unittest.TestCase):
    def setUp(self):
        # Create a specific ONNX model with a single QGemm node
        self.model_path = "qgemm_model.onnx"
        self.create_qgemm_model(self.model_path)
        self.cpu_jsons = []
        self.gpnpu_jsons = []

    def create_qgemm_model(self, output_model_path):
        a_scale, a_zero_point = 0.2039528638124466, -14
        b_scale, b_zero_point = 0.003937007859349251, 0
        y_scale, y_zero_point = 0.1019764319062233, -6

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

        a_sc = get_onnx_const(input_a_scale_name, a_scale, TensorProto.FLOAT)
        a_zp = get_onnx_const(input_a_zp_name, a_zero_point, TensorProto.INT8)
        b_sc = get_onnx_const(input_b_scale_name, b_scale, TensorProto.FLOAT)
        b_zp = get_onnx_const(input_b_zp_name, b_zero_point, TensorProto.INT8)
        y_sc = get_onnx_const(output_scale_name, y_scale, TensorProto.FLOAT)
        y_zp = get_onnx_const(output_zp_name, y_zero_point, TensorProto.INT8)
        # Define input and output tensors
        input_a_tensor = helper.make_tensor_value_info(input_a_name, TensorProto.INT8, [m, k])
        output_tensor = helper.make_tensor_value_info("out", TensorProto.INT8, [m, n])
        b = get_onnx_const(input_b_name, generate_normal_inputs([n, k], np.int8, 0, 32))
        y = get_onnx_const(output_name, generate_normal_inputs([n, ], np.int32, 0, 32))


        # Create QGEMM node
        qgemm_node = onnx.helper.make_node(
            "QGemm",
            inputs=[input_a_name, input_a_scale_name, input_a_zp_name,
                input_b_name, input_b_scale_name, input_b_zp_name,
                output_name,
                output_scale_name, output_zp_name],
            outputs=["out"],
            alpha=1.0,
            transA=0,
            transB=1,
            domain="com.microsoft"
        )

        # Create graph
        graph_name = "com.microsoft.QGEMM_test"
        graph = helper.make_graph(
            [qgemm_node],
            graph_name,
            [input_a_tensor],
            [output_tensor],
            initializer=[a_sc, a_zp, b, b_sc, b_zp, y, y_sc, y_zp],
        )

        # Create model
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.microsoft", 1),helper.make_opsetid("", 12)])
        model.ir_version = 8  # use stable onnx ir version

        # Save model
        onnx.checker.check_model(model, True)
        onnx.save(model, output_model_path)

    def tearDown(self):
        # Delete the ONNX file and JSON files after testing
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_accuracy(self, num_iterations=10):
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

            # Time and run CPU inference
            output_cpu = session_cpu.run(
                [session_cpu.get_outputs()[0].name],
                input_dict
            )[0]

            # Time and run GPNPU inference
            output_gpnpu = session_gpnpu.run(
                [session_gpnpu.get_outputs()[0].name],
                input_dict
            )[0]

            # Calculate max difference
            max_diff = np.max(np.abs(output_cpu - output_gpnpu))

            self.assertLessEqual(max_diff, 1)


if __name__ == "__main__":
    unittest.main()
