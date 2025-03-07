import onnx
import numpy as np
from onnx import helper, TensorProto
from tvm.contrib.epu.chimera_job.chimera_job import ChimeraJob
from pathlib import Path


# create new mode with standard dequantizeLinear
input_tensor = helper.make_tensor_value_info('input', TensorProto.INT8, [5])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [5])

dq2_s_value = 0.10242629051208496
dq2_zp_value = 5
dq2_s = onnx.helper.make_tensor(name="dq2_s", data_type=onnx.TensorProto.FLOAT, dims=(), vals=[dq2_s_value])
dq2_zp = onnx.helper.make_tensor(name="dq2_zp", data_type=onnx.TensorProto.INT8, dims=(), vals=[dq2_zp_value])


node = helper.make_node(
    'DequantizeLinear',  # Custom op name
    inputs=['input','dq2_s','dq2_zp'],
    outputs=['output'],
    domain="com.microsoft"
)

graph = helper.make_graph(
    [node],
    'test_graph',
    [input_tensor],
    [output_tensor],
    initializer=[dq2_s, dq2_zp]
)

onnx_model2 = helper.make_model(graph, opset_imports=[
    helper.make_operatorsetid("", 13),  # Standard ONNX domain
    helper.make_operatorsetid("com.microsoft", 1)  # Ensure com.microsoft is explicitly included
], ir_version=7)
onnx_model_path = 'quantize_linear_float_out.onnx'
onnx.save(onnx_model2, onnx_model_path)

input = np.array(
    [-128, 1, 2, 3, 127]).astype(np.int8)

cgc_job = ChimeraJob(model_p=onnx_model_path, macs_per_pe=8, quiet_iss=False)
#cgc_job.analyze_network()
cgc_job.compile(quiet=True)
outputs = cgc_job.run_inference_harness(inputs={"input": input})

print('cgc_job.experimental_path:', cgc_job.experiment_path)

# raw int32 output (underlying fixedpoint representation)
output_path = Path(cgc_job.experiment_path) / 'output.bin'
output_raw_int32 = np.fromfile(output_path, dtype=np.int32)
print('Input:', input)
print('Output:', output_raw_int32)
print('Output (float):', outputs["output"])

# [-1828407392   -54989696   -41242272   -27494848  1677185728]
# Output: [-1828407392   -54989696   -41242272   -27494848  1677185728]
