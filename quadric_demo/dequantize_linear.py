import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto
from onnxruntime_extensions import (
    onnx_op, PyCustomOpDef, make_onnx_model,
    get_library_path as _get_library_path)

# Define the custom op node
input_tensor = helper.make_tensor_value_info('input', TensorProto.INT8, [3])
output_tensor = helper.make_tensor_value_info('output', TensorProto.INT32, [3])


dq_s_frac_bits_value = 30
dq_s_value = (0.10242629051208496)*2**dq_s_frac_bits_value
dq_zp_value = 5
dq_output_frac_bits_value = 16

dq_s = onnx.helper.make_tensor(name="dq_s", data_type=onnx.TensorProto.INT32, dims=(), vals=[dq_s_value])
dq_s_frac_bits = onnx.helper.make_tensor(name="dq_s_frac_bits", data_type=onnx.TensorProto.INT8, dims=(), vals=[dq_s_frac_bits_value])
dq_zp = onnx.helper.make_tensor(name="dq_zp", data_type=onnx.TensorProto.INT8, dims=(), vals=[dq_zp_value])
dq_output_frac_bits = onnx.helper.make_tensor(name="dq_output_frac_bits", data_type=onnx.TensorProto.INT8, dims=(), vals=[dq_output_frac_bits_value])

node = helper.make_node(
    'DequantizeLinearFixedPoint',  # Custom op name
    inputs=['input', 'dq_s', 'dq_zp','dq_output_frac_bits'],
    outputs=['output'],
    domain='ai.onnx.contrib',  # Custom domain
)
# Create the graph and model
graph = helper.make_graph(
    [node],
    'test_graph',
    [input_tensor],
    [output_tensor],
    initializer=[dq_s, dq_zp, dq_output_frac_bits]
)

# Add opset import for the custom domain
opset_imports = [
    helper.make_opsetid("", 13),  # Default domain (ONNX)
    helper.make_opsetid("ai.onnx.contrib", 1)  # Custom domain
]
onnx_model = helper.make_model(
    graph, opset_imports=[helper.make_operatorsetid('ai.onnx.contrib', 1)], ir_version=7)

# Save the model
onnx_model_path = 'quantize_linear_fixed_point.onnx'
onnx.save(onnx_model, onnx_model_path)


def fx_mulitply(A, A_fp, B, B_fp, C_fp):
    shift = A_fp + B_fp - C_fp
    return (A * B) >> shift

# Register the custom op
@onnx_op(op_type="DequantizeLinearFixedPoint",
            inputs=[
                PyCustomOpDef.dt_int8, # x
                PyCustomOpDef.dt_int32, # s
                PyCustomOpDef.dt_int8, # s_frac_bits
                PyCustomOpDef.dt_int8, # zp
                PyCustomOpDef.dt_int8, # output_frac_bits
                PyCustomOpDef.dt_int8
            ],
            outputs=[PyCustomOpDef.dt_int32])
def dequantize_linear_fixed_point(x, s, s_frac_bits, zp, output_frac_bits):
    y_temp = x.astype(np.int32) - zp.astype(np.int32)  # floating point representation
    y_temp = y_temp * s # in s_frac_bits
    y_fixed = (y_temp << (output_frac_bits - s_frac_bits)).astype(np.int32) # fixed point representation
    return y_fixed

# Run model
so = ort.SessionOptions()
so.register_custom_ops_library(_get_library_path())
sess = ort.InferenceSession(onnx_model.SerializeToString(), so, providers=['CPUExecutionProvider'])
input = np.array(
    [1, 2, 3]).astype(np.int8)
output = sess.run(None, {'input': input})

print('Input:', input)
print('Output:', output[0])
