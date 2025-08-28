import unittest
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto
import os
import sys

# allows imports from directory of file
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import check_shape_inference
class TestQuantizeLinearFixedPoint(unittest.TestCase):
    def setUp(self):
        """
        Create an ONNX model with a single QuantizeLinearFixedPoint node.
        """
        self.input_output_shape = [1, 1, 2, 3]
        self.input_data_frac_bits = 27
        self.input_data = np.array([-15, 1.4, 2, 3.4,
      14.5, 15.5], dtype=np.float32).reshape(self.input_output_shape)
        self.s_value = 0.01865844801068306
        self.zp_value = -14

        self.expected_output = np.array([[-128, 61, 93, 127, 127, 127]], dtype=np.int8).reshape(self.input_output_shape)

    def _create_model_fixed_point(self, output_info_defined=True):
        # Create input & output tensors
        input_tensor = helper.make_tensor_value_info("input", TensorProto.INT32, self.input_output_shape)
        if output_info_defined:
            output_tensor = helper.make_tensor_value_info("output", TensorProto.INT8, self.input_output_shape)
        else:
            # This is useful for shape inference test
            output_tensor = helper.make_tensor_value_info("output", TensorProto.UNDEFINED, None)

        x_frac_bits = helper.make_tensor(name="x_frac_bits", data_type=TensorProto.INT8, dims=(), vals=[self.input_data_frac_bits])
        s = helper.make_tensor(name="s", data_type=TensorProto.FLOAT, dims=(), vals=[self.s_value])
        zp = helper.make_tensor(name="zp", data_type=TensorProto.INT8, dims=(), vals=[self.zp_value])

        # Define the node (QuantizeLinearFixedPoint op)
        node = helper.make_node(
            "QuantizeLinearFixedPoint",  # Custom op name
            inputs=["input", "x_frac_bits", "s", "zp"],
            outputs=["output"],
            domain="com.quadric"
        )

        # Create ONNX graph
        graph = helper.make_graph(
            [node],
            "test_graph",
            [input_tensor],
            [output_tensor],
            initializer=[x_frac_bits, s, zp]
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
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, self.input_output_shape)
        output_tensor = helper.make_tensor_value_info("output", TensorProto.INT8, self.input_output_shape)

        s = helper.make_tensor(name="s", data_type=TensorProto.FLOAT, dims=(), vals=[self.s_value])
        zp = helper.make_tensor(name="zp", data_type=TensorProto.INT8, dims=(), vals=[self.zp_value])

        # Define the node (QuantizeLinear op)
        node = helper.make_node(
            "QuantizeLinear",
            inputs=["input", "s", "zp"],
            outputs=["output"],
            domain="com.microsoft"
        )

        # Create ONNX graph
        graph = helper.make_graph(
            [node],
            "test_graph",
            [input_tensor],
            [output_tensor],
            initializer=[s, zp]
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


    def test_quantize_linear_fixed_point_inference(self):
        """
        Run inference on QuantizeLinearFixedPoint and validate output.
        """
        # fixed point model
        model_fixed_point = self._create_model_fixed_point()
        so = ort.SessionOptions()
        sess = ort.InferenceSession(model_fixed_point.SerializeToString(), so, providers=["CPUExecutionProvider"])
        input_data_fixed_point = (self.input_data * (2**self.input_data_frac_bits)).astype(np.int32)
        output_fixed_point = sess.run(None, {"input": input_data_fixed_point})[0]

        # floating point model
        model_float = self._create_model_float()
        sess = ort.InferenceSession(model_float.SerializeToString(), so, providers=["CPUExecutionProvider"])
        output_float = sess.run(None, {"input": self.input_data})[0]

        # Check output shape & values
        self.assertEqual(output_fixed_point.shape, output_float.shape)
        np.testing.assert_array_equal(output_fixed_point, output_float , err_msg="Quantized output mismatch!")

    def test_shape_inference(self):
        model = self._create_model_fixed_point(output_info_defined=False)
        expected_shapes = [
            helper.make_tensor_value_info("output", TensorProto.INT8, self.input_output_shape),
        ]
        check_shape_inference(model, expected_shapes)


if __name__ == "__main__":
    unittest.main()
