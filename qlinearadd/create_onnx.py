# import numpy as np
# import onnx
# from onnx import helper, TensorProto, numpy_helper

# def create_qlinear_add_model(h=128, w=128, channels=8):
#     # Using the scales from the original code
#     a_scale, a_zp = 0.018654844, -14  # Original x_scale, x_zp
#     b_scale, b_zp = 0.044774472, 0    # Original w_scale, w_zp
#     y_scale, y_zp = 0.023529412, -30  # Original y_scale, y_zp

#     # Create input shapes
#     input_shape = [1, channels, h, w]

#     # Create input/output tensors
#     a = helper.make_tensor_value_info('a', TensorProto.INT8, input_shape)
#     b = helper.make_tensor_value_info('b', TensorProto.INT8, input_shape)
#     y = helper.make_tensor_value_info('y', TensorProto.INT8, input_shape)

#     # Create constants for scales and zero points
#     a_scale_tensor = helper.make_tensor('a_scale', TensorProto.FLOAT, [], [a_scale])
#     a_zp_tensor = helper.make_tensor('a_zp', TensorProto.INT8, [], [a_zp])

#     b_scale_tensor = helper.make_tensor('b_scale', TensorProto.FLOAT, [], [b_scale])
#     b_zp_tensor = helper.make_tensor('b_zp', TensorProto.INT8, [], [b_zp])

#     y_scale_tensor = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [y_scale])
#     y_zp_tensor = helper.make_tensor('y_zp', TensorProto.INT8, [], [y_zp])

#     # Create QLinearAdd node with Microsoft domain
#     node = helper.make_node(
#         'QLinearAdd',
#         inputs=[
#             'a', 'a_scale', 'a_zp',
#             'b', 'b_scale', 'b_zp',
#             'y_scale', 'y_zp'
#         ],
#         outputs=['y'],
#         name='QLinearAdd_0',
#         domain='com.microsoft'  # Using Microsoft domain
#     )

#     # Create the graph
#     graph = helper.make_graph(
#         [node],
#         'qlinear_add_model',
#         [a, b],  # Only a and b are inputs, rest are initializers
#         [y],
#         initializer=[
#             a_scale_tensor, a_zp_tensor,
#             b_scale_tensor, b_zp_tensor,
#             y_scale_tensor, y_zp_tensor
#         ]
#     )

#     # Create the model with both Microsoft and ONNX opsets
#     model = helper.make_model(
#         graph,
#         producer_name='corrected_qlinear_add',
#         opset_imports=[
#             helper.make_opsetid("com.microsoft", 1),
#             helper.make_opsetid("", 12)
#         ]
#     )

#     # Verify the model
#     onnx.checker.check_model(model)
#     return model

# # Create and save the model
# model = create_qlinear_add_model()
# onnx.save(model, "correct_qlinear_add.onnx")
# print("Model saved to: correct_qlinear_add.onnx")

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def create_qlinear_add_model(h=128, w=128, channels=8):
    # Using the scales from the original code
    a_scale, a_zp = 0.018654844, -14
    b_scale, b_zp = 0.044774472, 0
    y_scale, y_zp = 0.023529412, -30

    # Create input shapes
    input_shape = [1, channels, h, w]

    # Create input/output tensors - use UINT8 instead of INT8
    a = helper.make_tensor_value_info('a', TensorProto.UINT8, input_shape)
    b = helper.make_tensor_value_info('b', TensorProto.UINT8, input_shape)
    y = helper.make_tensor_value_info('y', TensorProto.UINT8, input_shape)

    # Create constants for scales and zero points
    a_scale_tensor = helper.make_tensor('a_scale', TensorProto.FLOAT, [], [float(a_scale)])
    a_zp_tensor = helper.make_tensor('a_zp', TensorProto.UINT8, [], [int(a_zp + 128)])  # Shift to UINT8

    b_scale_tensor = helper.make_tensor('b_scale', TensorProto.FLOAT, [], [float(b_scale)])
    b_zp_tensor = helper.make_tensor('b_zp', TensorProto.UINT8, [], [int(b_zp + 128)])  # Shift to UINT8

    y_scale_tensor = helper.make_tensor('y_scale', TensorProto.FLOAT, [], [float(y_scale)])
    y_zp_tensor = helper.make_tensor('y_zp', TensorProto.UINT8, [], [int(y_zp + 128)])  # Shift to UINT8

    # Create QLinearAdd node
    node = helper.make_node(
        'QLinearAdd',
        inputs=[
            'a', 'a_scale', 'a_zp',
            'b', 'b_scale', 'b_zp',
            'y_scale', 'y_zp'
        ],
        outputs=['y'],
        name='QLinearAdd_0',
        domain=''  # Using standard domain instead of Microsoft
    )

    # Create the graph
    graph = helper.make_graph(
        [node],
        'qlinear_add_model',
        [a, b],
        [y],
        initializer=[
            a_scale_tensor, a_zp_tensor,
            b_scale_tensor, b_zp_tensor,
            y_scale_tensor, y_zp_tensor
        ]
    )

    # Create the model
    model = helper.make_model(
        graph,
        producer_name='corrected_qlinear_add',
        opset_imports=[helper.make_opsetid("", 10)]  # Using only standard ONNX opset
    )

    # Add IR version explicitly
    model.ir_version = 7

    # Verify the model
    onnx.checker.check_model(model)
    return model

# Create and save the model
model = create_qlinear_add_model()
onnx.save(model, "correct_qlinear_add.onnx")
print("Model saved to: correct_qlinear_add.onnx")
