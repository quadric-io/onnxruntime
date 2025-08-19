import unittest
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import os
import itertools
from testing_common import (
    apply_float_casting,
    apply_fixed_point_casting,
    get_onnx_initializers,
)
import pathlib
import sys
import os


class TestLayernormFixedPoint(unittest.TestCase):
    def setUp(self):
        """
        Setup for tests.
        """
        self.output_dir = os.path.dirname(os.path.abspath(__file__))
        # This is fixed for now, but this is supposed to be populated by tvm based on output data float ranges.
        self.fx_model_folder = os.path.join(self.output_dir, "layernorm_fxp_onnx")

        self.out_data_frac_bits = 29
        os.makedirs(self.fx_model_folder, exist_ok=True)

        self.epsilon = 9.999999974752427e-7
        self.gammabetaFbits = 31

    def create_model_fixed_point(
        self,
        model_path,
        input_shape,
        inp_fbits_val,  # Changed to _val to avoid name collision
        axis,
        scale_val,  # Changed to _val
        bias_val,  # Changed to _val
    ):
        """
        Create an ONNX model with a single LayernormFixedPoint node.
        """
        # Create input & output tensors
        input_tensor = helper.make_tensor_value_info("input", TensorProto.INT32, input_shape)
        output_tensor = helper.make_tensor_value_info("output", TensorProto.INT32, input_shape)

        # Initializer tensors (constants) - these should be TensorProto objects
        inp_fbits = helper.make_tensor(name="inp_frac_bits", data_type=TensorProto.INT8, dims=(), vals=[inp_fbits_val])
        scale = helper.make_tensor(
            name="Scale", data_type=TensorProto.FLOAT, dims=(input_shape[axis],), vals=scale_val.flatten().tolist()
        )
        bias = helper.make_tensor(
            name="B", data_type=TensorProto.FLOAT, dims=(input_shape[axis],), vals=bias_val.flatten().tolist()
        )
        out_fbits = helper.make_tensor(
            name="out_frac_bits", data_type=TensorProto.INT8, dims=(), vals=[self.out_data_frac_bits]
        )

        layernorm_fxp_attrs = {"axis": axis, "epsilon": self.epsilon, "stash_type": -1, "gbFbits": self.gammabetaFbits}

        # Define the node (LayernormFixedPoint op)
        node = helper.make_node(
            "LayernormFixedPoint",  # Custom op name
            # Inputs to the node are the *names* of the tensors, not the tensors themselves
            inputs=["input", "inp_frac_bits", "Scale", "B", "out_frac_bits"],
            outputs=["output"],
            domain="com.quadric",
            **layernorm_fxp_attrs,
        )

        # Create ONNX graph
        graph = helper.make_graph(
            [node],
            "test_graph",
            [input_tensor],
            [output_tensor],
            initializer=[inp_fbits, scale, bias, out_fbits],
        )

        # Create ONNX model
        onnx_model = helper.make_model(
            graph,
            opset_imports=[helper.make_operatorsetid("com.quadric", 1)],
        )

        # Save model
        onnx.save(onnx_model, model_path)

    def tearDown(self):
        """
        Remove the generated ONNX file after tests.
        """
        if os.path.exists(self.fx_model_folder):
            for file in os.listdir(self.fx_model_folder):
                os.remove(os.path.join(self.fx_model_folder, file))
            os.rmdir(self.fx_model_folder)

    def test_layernorm_fixed_point_inference(self):
        """
        Run inference on LayernormFixedPoint for various shapes and axes.
        """
        # Define parameters for shape generation
        asym = [0]  # True, False
        in_frac_bits = [16]
        channels = [197]
        width = [32]
        axes = [-1]

        test_cases = list(itertools.product(asym, in_frac_bits, channels, width, axes))

        for asym, in_frac_bits, channels, width, axis in test_cases:
            with self.subTest(shape=(channels, width, width), axis=axes):
                input_shape = [channels, width, width]
                model_name = f"layernorm_c{channels}_h{width}_a{axis}_inf{in_frac_bits}_asym{asym}.onnx"
                model_path = os.path.join(self.output_dir, "layernorm_fxp_onnx", model_name)

                # corresponding onnx file for weights and bias
                float_onnx_file = self.output_dir + "/layernorm_onnx/" + f"layernorm_{width}.onnx"

                initilziers = get_onnx_initializers(str(float_onnx_file), ["encoder.ln.weight", "encoder.ln.bias"])

                scale_tensor = initilziers[0]
                bias_tensor = initilziers[1]

                # Create the model
                self.create_model_fixed_point(model_path, input_shape, in_frac_bits, axis, scale_tensor, bias_tensor)

                # sym/asym input distributions helps catch overflow bugs where symmetric
                # distributions will not.
                limit = 2 ** (31 - in_frac_bits) - 1
                bot_limit = -limit
                if in_frac_bits == 31:
                    limit = 0.99
                if asym:
                    bot_limit = -limit / 2

                # Generate random input data
                min_value_fxp = apply_fixed_point_casting(np.float64(bot_limit), in_frac_bits)
                max_value_fxp = apply_fixed_point_casting(np.float64(limit), in_frac_bits)

                in_tensor_fxp = np.random.randint(
                    low=min_value_fxp,
                    high=max_value_fxp,
                    size=input_shape,
                    dtype=np.int32,
                )
                np_input = apply_float_casting(in_tensor_fxp, in_frac_bits)

                # Inference
                # try:
                so = ort.SessionOptions()
                sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])

                output = sess.run(None, {"input": in_tensor_fxp})[0]
                self.assertEqual(output.shape, tuple(input_shape))
                self.assertEqual(output.dtype, np.int32)
                print(f"Test passed for shape {input_shape} with axis {axis}")

                mean = np.mean(np_input, axis=axis, keepdims=True, dtype=np.float64)
                mean_squares = np.mean(np_input**2, axis=-1, keepdims=True, dtype=np.float64)
                var = mean_squares - mean**2
                std = np.sqrt(var + self.epsilon, dtype=np.float64)
                norm = (np_input - mean) / std
                out_tensor_expected = norm * scale_tensor + bias_tensor

                out_tensor = apply_float_casting(output, self.out_data_frac_bits)
                np.set_printoptions(threshold=sys.maxsize)

                num_elem_along_axis = np_input.shape[axis]
                norm_frac_bits = 30 - int(np.ceil(np.log(num_elem_along_axis)) / 2)

                # norm_error_expectation ~= 2^(-(30 - log2Ceil(width) / 2))
                # norm_error_expectation = 2^-27 for width == 32 and 2^-25 for width == 768
                # Add 3 to give enough leeway for randomness to pass
                atol = 2 ** (-norm_frac_bits + 4)
                abs_err = np.abs(out_tensor - out_tensor_expected)
                max_err = np.max(abs_err)
                err_count = (abs_err > atol).sum()

                print(f"Max absolute error: {max_err}")
                print(f"Total mismatches: {(100 * err_count / abs_err.size):.3f}% ({err_count}/{abs_err.size})")


if __name__ == "__main__":
    unittest.main()
