import onnx
import numpy as np
from onnx import helper, TensorProto
from tvm.contrib.epu.chimera_job.chimera_job import ChimeraJob
from pathlib import Path


# create new mode with standard dequantizeLinear
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [5])
output_tensor = helper.make_tensor_value_info('output', TensorProto.INT8, [5])

q_s_value = 0.01865844801068306
q_zp_value = -14
q_s = onnx.helper.make_tensor(name="q_s", data_type=onnx.TensorProto.FLOAT, dims=(), vals=[q_s_value])
q_zp = onnx.helper.make_tensor(name="q_zp", data_type=onnx.TensorProto.INT8, dims=(), vals=[q_zp_value])


node = helper.make_node(
    'QuantizeLinear',  # Custom op name
    inputs=['input','q_s','q_zp'],
    outputs=['output'],
    domain="com.microsoft"
)

graph = helper.make_graph(
    [node],
    'test_graph',
    [input_tensor],
    [output_tensor],
    initializer=[q_s, q_zp]
)

onnx_model2 = helper.make_model(graph, opset_imports=[
    helper.make_operatorsetid("", 13),  # Standard ONNX domain
    helper.make_operatorsetid("com.microsoft", 1)  # Ensure com.microsoft is explicitly included
], ir_version=7)
onnx_model_path = 'quantize_linear_float_in.onnx'
onnx.save(onnx_model2, onnx_model_path)

# input = np.array(
#     [-128, 1, 2, 3, 127]).astype(np.int8)
input = np.array(
    [-128.345, 1.4, 2, 3.4, 127.6]).astype(np.float32)

cgc_job = ChimeraJob(model_p=onnx_model_path, macs_per_pe=8, quiet_iss=False)
#cgc_job.analyze_network()
cgc_job.compile(quiet=True)
outputs = cgc_job.run_inference_harness(inputs={"input": input})

print('cgc_job.experimental_path:', cgc_job.experiment_path)

# raw int8 output (underlying fixedpoint representation)
output_path = Path(cgc_job.experiment_path) / 'output.bin'
output_raw_int8 = np.fromfile(output_path, dtype=np.int8)
print('Input:', input)
print('Output (direct):', output_raw_int8)
print('Output:', outputs["output"])

# [-1828407392   -54989696   -41242272   -27494848  1677185728]
# Output: [-1828407392   -54989696   -41242272   -27494848  1677185728]
