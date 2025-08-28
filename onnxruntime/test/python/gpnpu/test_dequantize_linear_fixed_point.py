import unittest
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto
import os
import sys

# allows imports from directory of file
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import check_shape_inference


class TestDequantizeLinearFixedPoint(unittest.TestCase):
    def setUp(self):
        """
        Create an ONNX model with a single DequantizeLinearFixedPoint node.
        """
        self.input_output_shape = [1, 1, 2, 3]
        self.scale_value = 0.10242629051208496
        self.zero_point_value = 5
        self.input_data = np.array([-128, 1, 2, 3, 4, 127], dtype=np.int8).reshape(self.input_output_shape)
        self.output_frac_bits = 27


    def _create_model_fixed_point(self, output_info_defined=True):
        # Create input & output tensors
        input_tensor = helper.make_tensor_value_info("input", TensorProto.INT8, self.input_output_shape)

        if output_info_defined:
            output_tensor = helper.make_tensor_value_info("output", TensorProto.INT32, self.input_output_shape)
        else:
            output_tensor = helper.make_tensor_value_info("output", TensorProto.UNDEFINED, None)

        # Initializer tensors (constants)
        y_frac_bits = helper.make_tensor(name="y_frac_bits", data_type=TensorProto.INT8, dims=(), vals=[self.output_frac_bits])
        dq_s = helper.make_tensor(name="dq_s", data_type=TensorProto.FLOAT, dims=(), vals=[self.scale_value])
        dq_zp = helper.make_tensor(name="dq_zp", data_type=TensorProto.INT8, dims=(), vals=[self.zero_point_value])

        # Define the node (DequantizeLinearFixedPoint op)
        node = helper.make_node(
            "DequantizeLinearFixedPoint",  # Custom op name
            inputs=["input", "y_frac_bits", "dq_s", "dq_zp"],
            outputs=["output"],
            domain="com.quadric"  # Custom domain
        )

        # Create ONNX graph
        graph = helper.make_graph(
            [node],
            "test_graph",
            [input_tensor],
            [output_tensor],
            initializer=[y_frac_bits, dq_s, dq_zp]
        )

        # Create ONNX model
        onnx_model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_operatorsetid("", 13),  # Standard ONNX opset
                helper.make_operatorsetid("com.quadric", 1)  # Custom domain
            ],
            ir_version=7
        )

        return onnx_model

    def _create_model_float(self):
        # Create input & output tensors
        input_tensor = helper.make_tensor_value_info("input", TensorProto.INT8, self.input_output_shape)
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, self.input_output_shape)

        # Initializer tensors (constants)
        dq_s = helper.make_tensor(name="dq_s", data_type=TensorProto.FLOAT, dims=(), vals=[self.scale_value])
        dq_zp = helper.make_tensor(name="dq_zp", data_type=TensorProto.INT8, dims=(), vals=[self.zero_point_value])

        # Define the node (DequantizeLinear op)
        node = helper.make_node(
            "DequantizeLinear",  # Custom op name
            inputs=["input", "dq_s", "dq_zp"],
            outputs=["output"],
            domain="com.microsoft"
        )

        # Create ONNX graph
        graph = helper.make_graph(
            [node],
            "test_graph",
            [input_tensor],
            [output_tensor],
            initializer=[dq_s, dq_zp]
        )

        # Create ONNX model
        onnx_model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_operatorsetid("", 13),  # Standard ONNX opset
                helper.make_operatorsetid("com.microsoft", 1)  # Custom domain
            ],
            ir_version=7
        )

        return onnx_model

    def test_dequantize_linear_fixed_point_inference(self):
        """
        Run inference on DequantizeLinearFixedPoint and validate output.
        """
        # Fixed-point dequantization
        model_fixed = self._create_model_fixed_point()
        so = ort.SessionOptions()
        sess = ort.InferenceSession(model_fixed.SerializeToString(), so, providers=["CPUExecutionProvider"])
        output_fixed_point_int32 = sess.run(None, {"input": self.input_data})[0]
        output_fixed_point_as_float = output_fixed_point_int32.astype(np.float32) * 2.**-self.output_frac_bits # convert to float

        # Floating-point dequantization
        so = ort.SessionOptions()
        model_float = self._create_model_float()
        sess = ort.InferenceSession(model_float.SerializeToString(), so, providers=["CPUExecutionProvider"])
        output_float = sess.run(None, {"input": self.input_data})[0]

        # Check output shape & values
        self.assertEqual(output_fixed_point_as_float.shape, output_float.shape)
        np.testing.assert_array_equal(output_fixed_point_as_float, output_float, err_msg="Dequantized output mismatch!")
        max_diff = np.max(np.abs(output_fixed_point_as_float - output_float))
        atol = 1e-6
        self.assertLessEqual(max_diff, atol, f"Max diff between fixed-point and float dequantized output: {max_diff}")

    def test_shape_inference(self):
        model = self._create_model_fixed_point(output_info_defined=False)
        expected_shapes = [
            helper.make_tensor_value_info("output", TensorProto.INT32, self.input_output_shape),
        ]
        check_shape_inference(model, expected_shapes)


if __name__ == "__main__":
    unittest.main()
