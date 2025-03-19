import unittest
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import os

class TestDequantizeLinearFixedPoint(unittest.TestCase):
    def setUp(self):
        """
        Create an ONNX model with a single DequantizeLinearFixedPoint node.
        """
        self.model_path = "dequantize_linear_fixed_point_test.onnx"
        self.input_shape = [5]
        self.scale_value = 0.10242629051208496
        self.zero_point_value = 5

        # Input data: int8 values
        self.input_data = np.array([-128, 1, 2, 3, 127], dtype=np.int8)

        # Expected output (int32, using fixed-point arithmetic)
        self.expected_output =  np.array([-1828407392,  -54989696,  -41242272,  -27494848,  1677185728],  dtype=np.int32)

        self.create_model(self.model_path)

    def create_model(self, model_path):
        # Create input & output tensors
        input_tensor = helper.make_tensor_value_info("input", TensorProto.INT8, self.input_shape)
        output_tensor = helper.make_tensor_value_info("output", TensorProto.INT32, self.input_shape)

        # Initializer tensors (constants)
        dq_s = helper.make_tensor(name="dq_s", data_type=TensorProto.FLOAT, dims=(), vals=[self.scale_value])
        dq_zp = helper.make_tensor(name="dq_zp", data_type=TensorProto.INT8, dims=(), vals=[self.zero_point_value])

        # Define the node (DequantizeLinearFixedPoint op)
        node = helper.make_node(
            "DequantizeLinearFixedPoint",  # Custom op name
            inputs=["input", "dq_s", "dq_zp"],
            outputs=["output"],
            domain="com.quadric"  # Custom domain
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
                helper.make_operatorsetid("com.quadric", 1)  # Custom domain
            ],
            ir_version=7
        )


        # Save model
        onnx.save(onnx_model, model_path)

    def tearDown(self):
        """
        Remove the generated ONNX file after tests.
        """
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_dequantize_linear_fixed_point_inference(self):
        """
        Run inference on DequantizeLinearFixedPoint and validate output.
        """
        so = ort.SessionOptions()
        sess = ort.InferenceSession(self.model_path, so, providers=["CPUExecutionProvider"])

        # Run inference
        output = sess.run(None, {"input": self.input_data})[0]

        # Check output shape & values
        self.assertEqual(output.shape, self.expected_output.shape)
        np.testing.assert_array_equal(output, self.expected_output, err_msg="Dequantized output mismatch!")

if __name__ == "__main__":
    unittest.main()
