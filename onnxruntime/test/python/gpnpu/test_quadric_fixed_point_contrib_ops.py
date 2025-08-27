import unittest
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import os

if os.path.exists(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "python",
        "tools",
        "symbolic_shape_infer.py",
    )
):
    # Allow running this test script without installing onnxruntime package.
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "python", "tools"))
    from symbolic_shape_infer import SymbolicShapeInference
else:
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


def _check_shapes(graph, inferred_graph, vis):  # type: (GraphProto, GraphProto, List[ValueInfoProto]) -> None
    names_in_vis = {x.name for x in vis}
    vis = list(x for x in graph.value_info if x.name not in names_in_vis) + vis
    inferred_vis = list(inferred_graph.value_info)
    vis = list(sorted(vis, key=lambda x: x.name))
    inferred_vis = list(sorted(inferred_vis, key=lambda x: x.name))
    if vis == inferred_vis:
        return
    # otherwise some custom logic to give a nicer diff
    vis_names = {x.name for x in vis}
    inferred_vis_names = {x.name for x in inferred_vis}
    assert vis_names == inferred_vis_names, (vis_names, inferred_vis_names)
    for vi, inferred_vi in zip(vis, inferred_vis):
        assert vi == inferred_vi, f"\n{vi}\n{inferred_vi}\n"
    raise AssertionError()

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

    def create_model_fixed_point(self, model_path, output_info_defined=True):

        # Create input & output tensors
        input_tensor = helper.make_tensor_value_info("input", TensorProto.INT32, self.input_output_shape)
        if output_info_defined:
            output_tensor = helper.make_tensor_value_info("output", TensorProto.INT8, self.input_output_shape)
        else:
            # This is useful for shape inference test
            output_tensor = helper.make_tensor_value_info("output", TensorProto.UNDEFINED, None)

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
        return onnx_model

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
        return onnx_model

    def tearDown(self):
        """
        Remove the generated ONNX file after tests.
        """
        if os.path.exists(self.model_fixed_point_path):
            os.remove(self.model_fixed_point_path)
        if os.path.exists(self.model_float_path):
            os.remove(self.model_float_path)
        if os.path.exists(self.fixed_point_model_no_output_info_path):
            os.remove(self.fixed_point_model_no_output_info_path)

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

    def test_shape_inference(self):
        self.fixed_point_model_no_output_info_path = "quantize_linear_fixed_point_test_no_output_info.onnx"
        model = self.create_model_fixed_point(model_path=self.fixed_point_model_no_output_info_path, output_info_defined=False)

        inferred_model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)

        expected_shapes = [
            helper.make_tensor_value_info("output", TensorProto.INT8, self.input_output_shape),
        ]

        _check_shapes(model.graph, inferred_model.graph, expected_shapes)


class TestDequantizeLinearFixedPoint(unittest.TestCase):
    def setUp(self):
        """
        Create an ONNX model with a single DequantizeLinearFixedPoint node.
        """
        self.model_fixed_point_path = "dequantize_linear_fixed_point_test.onnx"
        self.model_float_path = "dequantize_linear_float_test.onnx"
        self.input_output_shape = [1, 1, 2, 3]
        self.scale_value = 0.10242629051208496
        self.zero_point_value = 5
        self.input_data = np.array([-128, 1, 2, 3, 4, 127], dtype=np.int8).reshape(self.input_output_shape)

        self.output_frac_bits = 27

        self.create_model_fixed_point(self.model_fixed_point_path)
        self.create_model_float(self.model_float_path)

    def create_model_fixed_point(self, model_path, output_info_defined=True):
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


        # Save model
        onnx.save(onnx_model, model_path)
    def create_model_float(self, model_path):
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

        if os.path.exists(self.fixed_point_model_no_output_info_path):
            os.remove(self.fixed_point_model_no_output_info_path)

    def test_dequantize_linear_fixed_point_inference(self):
        """
        Run inference on DequantizeLinearFixedPoint and validate output.
        """
        # Fixed-point dequantization
        so = ort.SessionOptions()
        sess = ort.InferenceSession(self.model_fixed_point_path, so, providers=["CPUExecutionProvider"])
        output_fixed_point_int32 = sess.run(None, {"input": self.input_data})[0]
        output_fixed_point_as_float = output_fixed_point_int32.astype(np.float32) * 2.**-self.output_frac_bits # convert to float

        # Floating-point dequantization
        so = ort.SessionOptions()
        sess = ort.InferenceSession(self.model_float_path, so, providers=["CPUExecutionProvider"])
        output_float = sess.run(None, {"input": self.input_data})[0]

        # Check output shape & values
        self.assertEqual(output_fixed_point_as_float.shape, output_float.shape)
        np.testing.assert_array_equal(output_fixed_point_as_float, output_float, err_msg="Dequantized output mismatch!")
        max_diff = np.max(np.abs(output_fixed_point_as_float - output_float))
        atol = 1e-6
        self.assertLessEqual(max_diff, atol, f"Max diff between fixed-point and float dequantized output: {max_diff}")

    def test_shape_inference(self):
        self.fixed_point_model_no_output_info_path = "dequantize_linear_fixed_point_test_no_output_info.onnx"
        model = self.create_model_fixed_point(model_path=self.fixed_point_model_no_output_info_path, output_info_defined=False)

        inferred_model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)

        expected_shapes = [
            helper.make_tensor_value_info("output", TensorProto.INT8, self.input_output_shape),
        ]

        _check_shapes(model.graph, inferred_model.graph, expected_shapes)


if __name__ == "__main__":
    unittest.main()
