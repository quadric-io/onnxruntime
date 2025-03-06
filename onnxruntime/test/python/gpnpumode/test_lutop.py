# import onnx
# import numpy as np
# import onnxruntime as ort
# from onnx import helper, TensorProto

# # Define the custom op node
# input_tensor = helper.make_tensor_value_info('input', TensorProto.INT8, [3])
# lut_tensor = helper.make_tensor_value_info('lut', TensorProto.INT8, [256])
# output_tensor = helper.make_tensor_value_info('output', TensorProto.INT8, [3])

# node = helper.make_node(
#     'LookupTable',  # Custom op name
#     inputs=['input', 'lut'],
#     outputs=['output'],
#     domain='test.customop'  # Custom domain
# )

# # Create the graph and model
# graph = helper.make_graph(
#     [node],
#     'test_graph',
#     [input_tensor, lut_tensor],
#     [output_tensor]
# )

# # Add opset import for the custom domain
# opset_imports = [
#     helper.make_opsetid("", 13),  # Default domain (ONNX)
#     helper.make_opsetid("test.customop", 1)  # Custom domain
# ]

# model = helper.make_model(graph, opset_imports=opset_imports, producer_name='custom_op_test')

# # Save the model
# onnx.save(model, 'test_model.onnx')

# # Prepare input data
# input_data = np.array([-128, 0, 127], dtype=np.int8)
# lut_data = np.array([127 - i for i in range(256)], dtype=np.int8)  # Example LUT: invert values

# # Run the model with ONNX Runtime
# so = ort.SessionOptions()
# so.register_custom_ops_library('/home/maggies/onnxruntime/build/Linux/Release/libcustom_op_library.so')  # Path to your custom op library

# session = ort.InferenceSession('test_model.onnx', so)
# inputs = {'input': input_data, 'lut': lut_data}
# outputs = session.run(None, inputs)

# print('Input:', input_data)
# print('LUT:', lut_data)
# print('Output:', outputs[0])


import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto

# Define the custom op node
input_tensor = helper.make_tensor_value_info('input', TensorProto.INT8, [3])
output_tensor = helper.make_tensor_value_info('output', TensorProto.INT8, [3])

# Example LUT: invert values
lut_data = np.array([127 - i for i in range(256)], dtype=np.int8)

# Create the LUT tensor attribute
lut_tensor = helper.make_tensor(
    name='lut',
    data_type=TensorProto.INT8,
    dims=[256],
    vals=lut_data
)

node = helper.make_node(
    'LookupTable',  # Custom op name
    inputs=['input'],
    outputs=['output'],
    domain='test.customop',  # Custom domain
    lut=lut_tensor  # LUT as an attribute
)

# Create the graph and model
graph = helper.make_graph(
    [node],
    'test_graph',
    [input_tensor],
    [output_tensor]
)

# Add opset import for the custom domain
opset_imports = [
    helper.make_opsetid("", 13),  # Default domain (ONNX)
    helper.make_opsetid("test.customop", 1)  # Custom domain
]

model = helper.make_model(graph, opset_imports=opset_imports, producer_name='custom_op_test')

# Save the model
onnx.save(model, 'test_model.onnx')

# Prepare input data
input_data = np.array([-128, 0, 127], dtype=np.int8)

# Run the model with ONNX Runtime
so = ort.SessionOptions()
so.register_custom_ops_library('/home/maggies/onnxruntime/build/Linux/Release/libcustom_op_library.so')  # Path to your custom op library

session = ort.InferenceSession('test_model.onnx', so)
inputs = {'input': input_data}
outputs = session.run(None, inputs)

print('Input:', input_data)
print('LUT:', lut_data)
print('Output:', outputs[0])
