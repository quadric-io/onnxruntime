import unittest
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import os
import itertools
import shutil
import pathlib
import sys
import math

# These helper functions are assumed to be defined in testing_common.py
# and `out_fbits`, `wt_frac_bits`, `bias_frc_bits` are constants
# from that module.
from testing_common import (
    apply_float_casting,
    apply_fixed_point_casting,
)

# The parameters for this test are directly ported from sdk, but this test runs for both width and channelwise layernorm.
out_fbits = 29
wt_frac_bits = 30
bias_frc_bits = 31


class TestLayernormFixedPoint(unittest.TestCase):
    def setUp(self):
        """
        Set up shared test parameters and create a test directory.
        """
        self.output_dir = os.path.dirname(os.path.abspath(__file__))
        self.fx_model_folder = os.path.join(self.output_dir, "layernorm_fxp_onnx")
        self.epsilon = 9.999999974752427e-7
        os.makedirs(self.fx_model_folder, exist_ok=True)
        self.out_data_frac_bits = out_fbits

    def tearDown(self):
        """
        Remove the generated ONNX files and the test directory.
        """
        if os.path.exists(self.fx_model_folder):
            shutil.rmtree(self.fx_model_folder)

    def _create_model_fixed_point(
        self,
        model_path,
        input_shape,
        inp_fbits,
        axis,
        scale,
        bias,
    ):
        """
        Creates an ONNX model with a single LayernormFixedPoint node.
        """
        input_tensor = helper.make_tensor_value_info("x", TensorProto.INT32, input_shape)
        output_tensor = helper.make_tensor_value_info("output", TensorProto.INT32, input_shape)

        inp_fbits_tensor = helper.make_tensor(name="x_frac_bits", data_type=TensorProto.INT8, dims=[], vals=[inp_fbits])
        scale_tensor = helper.make_tensor(
            name="scale", data_type=TensorProto.FLOAT, dims=[input_shape[axis]], vals=scale.flatten().tolist()
        )
        bias_tensor = helper.make_tensor(
            name="bias", data_type=TensorProto.FLOAT, dims=[input_shape[axis]], vals=bias.flatten().tolist()
        )
        out_fbits_tensor = helper.make_tensor(
            name="out_frac_bits", data_type=TensorProto.INT8, dims=[], vals=[self.out_data_frac_bits]
        )

        layernorm_fxp_attrs = {
            "axis": axis,
            "epsilon": self.epsilon,
            "stash_type": -1,
            "wt_fbits": wt_frac_bits,
            "bias_fbits": bias_frc_bits,
        }

        node = helper.make_node(
            "LayernormFixedPoint",
            inputs=["x", "x_frac_bits", "scale", "bias", "out_frac_bits"],
            outputs=["output"],
            domain="com.quadric",
            **layernorm_fxp_attrs,
        )

        graph = helper.make_graph(
            [node],
            "test_graph",
            [input_tensor],
            [output_tensor],
            initializer=[inp_fbits_tensor, scale_tensor, bias_tensor, out_fbits_tensor],
        )

        onnx_model = helper.make_model(
            graph,
            opset_imports=[helper.make_operatorsetid("com.quadric", 1)],
        )
        onnx.save(onnx_model, model_path)

    def _calculate_expected_output(self, np_input, axis, scale_tensor, bias_tensor):
        """
        Calculates the expected output using NumPy's floating-point arithmetic.
        """
        # Calculate mean and variance for the specified axis
        mean = np.mean(np_input, axis=axis, keepdims=True, dtype=np.float64)
        var = np.var(np_input, axis=axis, keepdims=True, dtype=np.float64)
        std = np.sqrt(var + self.epsilon, dtype=np.float64)

        # Reshape scale and bias for broadcasting
        # The scale/bias tensor shape must be broadcastable with the normalized tensor.
        # e.g., if input is (1, 197, 32) and axis is 1 (channels), scale is (197,)
        # Reshaping it to (1, 197, 1) allows correct broadcasting.
        if len(np_input.shape) == 3:
            if axis == 1 or axis == -2:
                scale_reshaped = scale_tensor.reshape(1, scale_tensor.shape[0], 1)
                bias_reshaped = bias_tensor.reshape(1, bias_tensor.shape[0], 1)
            else:  # axis == 2 or axis == -1 (width)
                scale_reshaped = scale_tensor.reshape(1, 1, scale_tensor.shape[0])
                bias_reshaped = bias_tensor.reshape(1, 1, bias_tensor.shape[0])
        else:
            # Handle other dimensions if needed
            scale_reshaped = scale_tensor
            bias_reshaped = bias_tensor

        distance = np_input - mean
        norm = distance / std
        scaled_ip = norm * scale_reshaped
        out_tensor_expected = scaled_ip + bias_reshaped

        return out_tensor_expected

    def _run_test_case(self, asym, in_frac_bits, channels, width, axis):
        """
        A helper method to run a single test case.
        """
        input_shape = [1, channels, width]
        model_name = f"layernorm_c{channels}_w{width}_a{axis}_inf{in_frac_bits}_asym{asym}.onnx"
        model_path = os.path.join(self.fx_model_folder, model_name)

        # Generate random input data
        limit = 2 ** (31 - in_frac_bits) - 1
        bot_limit = -limit
        if in_frac_bits == 31:
            limit = 0.99
        if asym:
            bot_limit = -limit / 2

        min_value_fxp = apply_fixed_point_casting(np.float64(bot_limit), in_frac_bits)
        max_value_fxp = apply_fixed_point_casting(np.float64(limit), in_frac_bits)
        in_tensor_fxp = np.random.randint(
            low=min_value_fxp,
            high=max_value_fxp,
            size=input_shape,
            dtype=np.int32,
        )

        # These ranges are based on existing test cases inside sdk.
        scale_tensor = np.random.uniform(0.5, 0.7, size=input_shape[axis]).astype(np.float32)
        bias_tensor = np.random.uniform(-0.01, 0.01, size=input_shape[axis]).astype(np.float32)

        self._create_model_fixed_point(model_path, input_shape, in_frac_bits, axis, scale_tensor, bias_tensor)

        np_input = apply_float_casting(in_tensor_fxp, in_frac_bits)

        so = ort.SessionOptions()
        sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        output = sess.run(None, {"x": in_tensor_fxp})[0]
        self.assertEqual(output.dtype, np.int32)
        print(f"Test passed for shape {input_shape} with axis {axis}")

        out_tensor_expected = self._calculate_expected_output(np_input, axis, scale_tensor, bias_tensor)
        out_tensor = apply_float_casting(output, self.out_data_frac_bits)

        num_elem_along_axis = np_input.shape[axis]
        norm_frac_bits = 30 - int(math.ceil(math.log2(num_elem_along_axis)) / 2)
        atol = 2 ** (-norm_frac_bits + 4)

        np.testing.assert_allclose(
            out_tensor, out_tensor_expected, atol=atol, rtol=1e-5, err_msg="Quantized output mismatch!"
        )

    def test_layernorm_fixed_point_inference(self):
        """
        Main test method that runs all defined test cases.
        """
        TEST_CASES = [
            # Width-wise layernorm tests
            {"asym": [0, 1], "in_frac_bits": range(16, 32), "channels": [197], "width": [32, 512, 768], "axis": -1},
            # Channel-wise layernorm tests
            {"asym": [0, 1], "in_frac_bits": range(16, 32), "channels": [32, 512, 768], "width": [197], "axis": 1},
        ]

        for case in TEST_CASES:
            # Generate all combinations of parameters for this test case
            param_combinations = itertools.product(
                case["asym"], case["in_frac_bits"], case["channels"], case["width"], [case["axis"]]
            )
            for params in param_combinations:
                with self.subTest(params=params):
                    self._run_test_case(*params)


if __name__ == "__main__":
    unittest.main()
