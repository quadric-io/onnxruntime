import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto
from onnxruntime_extensions import (
    onnx_op, PyCustomOpDef, make_onnx_model,
    get_library_path as _get_library_path)
from tvm.contrib.epu.chimera_job.chimera_job import ChimeraJob
from interval import interval
import math
from typing import Union
# create new mode with standard dequantizeLinear
input_tensor = helper.make_tensor_value_info('input', TensorProto.INT32, [5])
output_tensor = helper.make_tensor_value_info('output', TensorProto.INT8, [5])

# globals
post_mac_int_bits = 29
post_mac_frac_bits = 31 - post_mac_int_bits


x_frac_bits_value = 27
s_value = 0.01865844801068306
zp_value = -14

x_frac_bits = onnx.helper.make_tensor(name="x_frac_bits", data_type=onnx.TensorProto.INT8, dims=(), vals=[x_frac_bits_value])
s = onnx.helper.make_tensor(name="s", data_type=onnx.TensorProto.FLOAT, dims=(), vals=[s_value])
zp = onnx.helper.make_tensor(name="zp", data_type=onnx.TensorProto.INT8, dims=(), vals=[zp_value])


node = helper.make_node(
    'QuantizeLinearFixedPoint',  # Custom op name
    inputs=['input','x_frac_bits', 's','zp'],
    outputs=['output'],
    domain='com.quadric'
)

graph = helper.make_graph(
    [node],
    'test_graph',
    [input_tensor],
    [output_tensor],
    initializer=[x_frac_bits, s, zp]
)

onnx_model = helper.make_model(graph, opset_imports=[
    helper.make_operatorsetid("", 13),  # Standard ONNX domain
    helper.make_operatorsetid("com.microsoft", 1),  # Ensure com.microsoft is explicitly included
    helper.make_operatorsetid("com.quadric", 1)  # Custom domain
], ir_version=7)
onnx_model_path = 'quantize_linear_fixed_point.onnx'
onnx.save(onnx_model, onnx_model_path)

# Helpers
def get_dequantized_range(scale, zero_point):
    # Gets range of dequantization of int8 tensor
    ii8 = np.iinfo(np.int8)
    bounds = [(ii8.min - zero_point) * scale, (ii8.max - zero_point) * scale]
    return interval(bounds)

def frac_bits_from_range(_interval):
    # Given the range of data, this function calculates the bits are needed to
    # represent the range with signed int32 data type
    assert len(_interval) == 1, f"Expected single tuple in an Interval: {_interval}"
    i = _interval[0]
    range_min = abs(float(i[0]))
    range_max = range_min if len(i) == 1 else abs(float(i[1]))
    if range_min > range_max:
        int_bits = math.ceil(math.log2(range_min)) if range_min >= 1.0 else 0
    else:
        int_bits = math.ceil(math.log2(range_max + 1)) if range_max >= 1.0 else 0
    return 31 - int_bits

def get_dequantized_frac_bits(scale, zero_point):
    return frac_bits_from_range(get_dequantized_range(scale, zero_point))


def fixed_point_to_scalar(multiplier: int, shift: int):
    return multiplier / (1 << 31) * (2**shift)


def fixed_point_to_qfp(multiplier: int, shift: int):
    scalar = fixed_point_to_scalar(multiplier, shift)
    return data_to_qfp(scalar)


def data_to_qfp(
    data: Union[np.ndarray, int, float], frac_bits=None, qfp_size=32, scalar_as_float=True
):
    def derive_fractional_bits(scalar):
        value_bits = qfp_size - 1

        _, int_part = math.modf(scalar)
        int_part = abs(int_part)

        int_bits = 0 if int_part == 0 else int(math.log2(int_part)) + 1
        frac_bits = value_bits - int_bits

        assert frac_bits >= 0, "Scalar cannot be represented in qfp format."

        return frac_bits

    def scalar_to_qfp(value, frac_bits):
        frac, integer = math.modf(value)

        integer = int(abs(integer)) << frac_bits
        frac = round(abs(frac) * (1 << frac_bits))

        qfp = integer + frac
        if value < 0:
            qfp *= -1

        return int(qfp)

    if isinstance(data, np.ndarray) and data.size != 1:
        frac_bits = frac_bits if frac_bits != None else derive_fractional_bits(np.max(np.abs(data)))
        qfp = np.vectorize(scalar_to_qfp)(data, frac_bits)
    else:
        frac_bits = frac_bits if frac_bits != None else derive_fractional_bits(data)
        if scalar_as_float:
            # In the case where the value is an immediate return
            # it as float to be consumed directly by the codegen
            qfp = float(data)
        else:
            # Case when converting to integer constant in Relay to enable
            # post quantized CPU inference
            qfp = scalar_to_qfp(data, frac_bits)

    return qfp, frac_bits

def round_to_pos_inf(x, frac_bits):
    zp5 = 1 << (frac_bits - 1)
    return (x + zp5) >> frac_bits
# Register the custom op
@onnx_op(op_type="QuantizeLinearFixedPoint",
            inputs=[
                PyCustomOpDef.dt_int32, # x
                PyCustomOpDef.dt_int8, # x_frac_bits
                PyCustomOpDef.dt_float, # s
                PyCustomOpDef.dt_int8, # zp
            ],
            outputs=[PyCustomOpDef.dt_int8])
def quantize_linear_fixed_point(x, x_frac_bits, s, zp):
    print("s", s)
    scale_inv = 1./s
    print("scale_inv", scale_inv)
    scale_inv_value, scale_inv_frac_bits = data_to_qfp(scale_inv, scalar_as_float=False)
    result_frac_bits = post_mac_frac_bits
    mul_post_shift = scale_inv_frac_bits + x_frac_bits - result_frac_bits
    if mul_post_shift > 31:
        mul_post_shift = 31
        result_frac_bits = scale_inv_frac_bits + x_frac_bits - 31
    print("scale_inv_value", scale_inv_value)
    print("scale_inv_frac_bits", scale_inv_frac_bits)
    print("mul_post_shift", mul_post_shift)
    print("result_frac_bits", result_frac_bits)
    y_temp = np.int64(x) * np.int64(scale_inv_value) # in s_frac_bits
    if mul_post_shift > 0:
        y_temp = y_temp >> mul_post_shift
    else:
        y_temp = y_temp << -mul_post_shift
    print("y_temp (before round)", y_temp)
    y_temp = round_to_pos_inf(y_temp, result_frac_bits)
    print("y_temp (after round)", y_temp)
    y_fixed = np.int8(np.clip(y_temp + zp, -128, 127))
    print("y_temp (after clip)", y_temp)
    return y_fixed

# Run model
so = ort.SessionOptions()
#so.register_custom_ops_library(_get_library_path())
sess = ort.InferenceSession(onnx_model_path, so, providers=['CPUExecutionProvider'])
input = np.array(
    [-128.345, 1.4, 2, 3.4, 127.6]).astype(np.float32)
input = (input * (2**27)).astype(np.int32)
output = sess.run(None, {'input': input})

print('Input:', input)
print('Output:', output[0])
