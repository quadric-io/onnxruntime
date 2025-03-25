import unittest
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import os

class TestQuantizeLinearFixedPoint(unittest.TestCase):
    def setUp(self):
        """
        Create an ONNX model with a single QuantizeLinearFixedPoint node.
        """
        self.model_fixed_point_path = "quantize_linear_fixed_point_test.onnx"
        self.model_float_path = "quantize_linear_float_test.onnx"
        self.input_output_shape = [1, 1, 2, 3]
        self.input_data_frac_bits = 27
        self.input_data = np.array([-15, 1.4, 2, 3.4,
      14.5, 15.5], dtype=np.float32).reshape(self.input_output_shape)
        self.expected_output = np.array([[-128, 61, 93, 127, 127, 127]], dtype=np.int8).reshape(self.input_output_shape)
        self.create_model_fixed_point(self.model_fixed_point_path)
        self.create_model_float(self.model_float_path)

    def create_model_fixed_point(self, model_path):

        # Create input & output tensors
        input_tensor = helper.make_tensor_value_info("input", TensorProto.INT32, self.input_output_shape)
        output_tensor = helper.make_tensor_value_info("output", TensorProto.INT8, self.input_output_shape)

        # Initializer tensors (constants)
        s_value = 0.01865844801068306
        zp_value = -14

        x_frac_bits = helper.make_tensor(name="x_frac_bits", data_type=TensorProto.INT8, dims=(), vals=[self.input_data_frac_bits])
        s = helper.make_tensor(name="s", data_type=TensorProto.FLOAT, dims=(), vals=[s_value])
        zp = helper.make_tensor(name="zp", data_type=TensorProto.INT8, dims=(), vals=[zp_value])

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

        # Save model
        onnx.save(onnx_model, model_path)

    def create_model_float(self, model_path):

        # Create input & output tensors
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, self.input_output_shape)
        output_tensor = helper.make_tensor_value_info("output", TensorProto.INT8, self.input_output_shape)

        # Initializer tensors (constants)
        s_value = 0.01865844801068306
        zp_value = -14

        s = helper.make_tensor(name="s", data_type=TensorProto.FLOAT, dims=(), vals=[s_value])
        zp = helper.make_tensor(name="zp", data_type=TensorProto.INT8, dims=(), vals=[zp_value])

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

        # Save model
        onnx.save(onnx_model, model_path)

    def tearDown(self):
        """
        Remove the generated ONNX file after tests.
        """
        if os.path.exists(self.model_fixed_point_path):
            os.remove(self.model_fixed_point_path)
        if os.path.exists(self.model_float_path):
            os.remove(self.model_float_path)

    def test_quantize_linear_fixed_point_inference(self):
        """
        Run inference on QuantizeLinearFixedPoint and validate output.
        """
        # fixed point model
        so = ort.SessionOptions()
        sess = ort.InferenceSession(self.model_fixed_point_path, so, providers=["CPUExecutionProvider"])
        input_data_fixed_point = (self.input_data * (2**self.input_data_frac_bits)).astype(np.int32)
        output_fixed_point = sess.run(None, {"input": input_data_fixed_point})[0]

        # floating point model
        self.create_model_float(self.model_float_path)
        sess = ort.InferenceSession(self.model_float_path, so, providers=["CPUExecutionProvider"])
        output_float = sess.run(None, {"input": self.input_data})[0]

        # Check output shape & values
        self.assertEqual(output_fixed_point.shape, output_float.shape)
        np.testing.assert_array_equal(output_fixed_point, output_float , err_msg="Quantized output mismatch!")

if __name__ == "__main__":
    unittest.main()
