#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import os

x_scale, x_zp = 0.018654844, -14
w_scale, w_zp = 0.044774472, 0
y_scale, y_zp = 0.023529412, -30


# def canonicalize_conv_params(kernel, strides, padding, dilation):
#     kernel = [kernel, kernel] if not isinstance(kernel, (list, tuple)) else kernel

#     assert len(kernel) == 2, "Unexpected kernel:\n{call}"

#     strides = [strides, strides] if not isinstance(strides, (list, tuple)) else strides

#     assert len(strides) == 2, "Unexpected strides:\n{call}"

#     padding = (
#         [int(padding or 0), int(padding or 0), int(padding or 0), int(padding or 0)]
#         if not isinstance(padding, (list, tuple))
#         else padding
#     )
#     assert len(padding) == 4, "Unexpected padding:\n{call}"

#     dilation = [dilation, dilation] if not isinstance(dilation, (list, tuple)) else dilation

#     assert len(dilation) == 2, "Unexpected dilation:\n{call}"

#     return kernel, strides, padding, dilation

# def conv_output_height_width(kernel, strides, padding, dilation, input_dims):
#     kernel, strides, padding, dilation = canonicalize_conv_params(
#         kernel, strides, padding, dilation
#     )
#     return int(
#         (input_dims[0] + padding[0] + padding[2] - dilation[0] * (kernel[0] - 1) - 1) // strides[0]
#         + 1
#     ), int(
#         (input_dims[1] + padding[1] + padding[3] - dilation[1] * (kernel[1] - 1) - 1) // strides[1]
#         + 1
#     )

# def generate_normal_inputs(shape, dtype, mu=0, sigma=32, a_min=-127, a_max=127):
#     return np.clip(np.rint(np.random.normal(mu, sigma, shape)).astype(dtype), a_min, a_max)

# def get_onnx_linear_conv(
#     op_name,
#     inp,  # Should be a ValueInfo
#     oc,
#     kernel_shape,
#     strides=[1, 1],
#     auto_pad="NOTSET",
#     padding=None,
#     dilations=[1, 1],
#     groups=1,
#     x_scale=1.0,
#     x_zp=0,
#     w_scale=1.0,
#     w_zp=0,
#     y_scale=1.0,
#     y_zp=0,
#     with_bias=True,
# ):
#     kernel_shape = (
#         [kernel_shape, kernel_shape]
#         if not isinstance(kernel_shape, (list, tuple))
#         else kernel_shape
#     )

#     if padding is None and auto_pad == "NOTSET":
#         padding = [int(kernel_shape[0]) // 2] * 4

#     xs = get_onnx_const(f"{op_name}.x_scale", x_scale)
#     xz = get_onnx_const(f"{op_name}.x_zp", x_zp)
#     ws = get_onnx_const(f"{op_name}.w_scale", w_scale)
#     wz = get_onnx_const(f"{op_name}.w_zp", w_zp)
#     ys = get_onnx_const(f"{op_name}.y_scale", y_scale)
#     yz = get_onnx_const(f"{op_name}.y_zp", y_zp)

#     in_dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
#     # FIXME: Need to take into account padding and what not
#     ic = in_dims[1]
#     if padding:
#         out_height, out_width = conv_output_height_width(
#             kernel_shape, strides, padding, dilations, in_dims[-2:]
#         )
#     else:
#         out_height = in_dims[-2] // strides[-2]
#         out_width = in_dims[-1] // strides[-1]
#     out_dims = [1, oc, out_height, out_width]

#     group_size = ic // groups
#     wt_dims = [oc, group_size, kernel_shape[0], kernel_shape[1]]
#     bias_dims = [oc]

#     wt = get_onnx_const(f"{op_name}.wt", generate_normal_inputs(wt_dims, np.int8, 0, 32))
#     bias = get_onnx_const(
#         f"{op_name}.bias",
#         generate_normal_inputs(bias_dims, np.int32, 0, 256, -1024, 1024),
#         onnx.TensorProto.INT32,
#     )

#     out_name = f"{op_name}.output"
#     out = onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.INT8, out_dims)

#     names = [
#         inp.name,
#         f"{op_name}.x_scale",
#         f"{op_name}.x_zp",
#         f"{op_name}.wt",
#         f"{op_name}.w_scale",
#         f"{op_name}.w_zp",
#         f"{op_name}.y_scale",
#         f"{op_name}.y_zp",
#         f"{op_name}.bias",
#     ]
#     initializers = [xs, xz, wt, ws, wz, ys, yz, bias]

#     if auto_pad == "NOTSET":
#         conv = onnx.helper.make_node(
#             "QLinearConv",
#             names,
#             [out_name],
#             name=op_name,
#             dilations=dilations,
#             group=groups,
#             pads=padding,
#             strides=strides,
#             kernel_shape=kernel_shape,
#         )
#     else:
#         conv = onnx.helper.make_node(
#             "QLinearConv",
#             names,
#             [out_name],
#             name=op_name,
#             dilations=dilations,
#             auto_pad=auto_pad,
#             group=groups,
#             strides=strides,
#             kernel_shape=kernel_shape,
#         )

#     return conv, out, initializers

# def get_onnx_const(name, val, dtype=None):
#     if isinstance(val, np.ndarray):
#         dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[val.dtype]
#         dims = val.shape
#     else:
#         if not dtype:
#             dtype = onnx.TensorProto.INT8 if isinstance(val, int) else onnx.TensorProto.FLOAT
#         dims = ()
#         val = [val]

#     return onnx.helper.make_tensor(name=name, data_type=dtype, dims=dims, vals=val)

# def get_onnx(
#     h,
#     w,
#     ic,
#     oc,
#     kernel_size,
#     strides,
#     padding=None,
#     dilation=[1, 1],
#     auto_pad="NOTSET",
#     pad_mode="constant",
#     include_pre_op=False,
#     groups=1,
# ):
#     kernel_size = (
#         [kernel_size, kernel_size] if not isinstance(kernel_size, (list, tuple)) else kernel_size
#     )

#     inp_dims = (1, ic, h, w)

#     ops = []
#     inits = []
#     if include_pre_op:
#         op_name = "inp_relu"
#         relu_min = get_onnx_const(f"{op_name}.min", 0, dtype=onnx.TensorProto.INT8)
#         relu_max = get_onnx_const(f"{op_name}.max", 6, dtype=onnx.TensorProto.INT8)
#         inits = [relu_min, relu_max]
#         inp_pre = onnx.helper.make_tensor_value_info("inp.pre", onnx.TensorProto.INT8, inp_dims)
#         inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.INT8, inp_dims)

#         relu6 = onnx.helper.make_node(
#             "Clip", ["inp.pre", f"{op_name}.min", f"{op_name}.max"], ["inp"], name=op_name
#         )
#         ops.append(relu6)
#     elif pad_mode == "reflect":
#         # Create a pad node ahead of the conv
#         op_name = "inp_pad"
#         inp_pre = onnx.helper.make_tensor_value_info("inp.pre", onnx.TensorProto.INT8, inp_dims)
#         padded_dims = list(inp_dims)
#         padded_dims[-2] = 2 * (kernel_size[0] // 2)
#         padded_dims[-1] = 2 * (kernel_size[1] // 2)
#         inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.INT8, padded_dims)
#         inp_pads = get_onnx_const(
#             "inp.pads",
#             np.array(
#                 [
#                     0,
#                     0,
#                     kernel_size[0] // 2,
#                     kernel_size[1] // 2,
#                     0,
#                     0,
#                     kernel_size[0] // 2,
#                     kernel_size[1] // 2,
#                 ],
#                 dtype=np.int64,
#             ),
#             onnx.TensorProto.INT64,
#         )
#         inits = [inp_pads]
#         pad = onnx.helper.make_node(
#             "Pad", ["inp.pre", "inp.pads"], ["inp"], name=op_name, mode=pad_mode
#         )
#         padding = [0, 0, 0, 0]
#         ops.append(pad)
#     else:
#         inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.INT8, inp_dims)

#     conv, outp, conv_inits = get_onnx_linear_conv(
#         "conv_0",
#         inp,
#         oc,
#         kernel_size,
#         strides,
#         auto_pad=auto_pad,
#         padding=padding,
#         dilations=dilation,
#         groups=groups,
#         x_scale=x_scale,
#         x_zp=x_zp,
#         w_scale=w_scale,
#         w_zp=w_zp,
#         y_scale=y_scale,
#         y_zp=y_zp,
#     )
#     ops.append(conv)
#     inits = inits + conv_inits

#     graph_input = (inp_pre if include_pre_op or pad_mode == "reflect" else inp,)
#     graph = onnx.helper.make_graph(
#         ops,
#         "test_conv",
#         graph_input,
#         [outp],
#         initializer=inits,
#     )

#     model = onnx.helper.make_model(
#         graph,
#         opset_imports=[
#             onnx.helper.make_opsetid("com.microsoft", 1),
#             onnx.helper.make_opsetid("", 12),
#         ],
#     )
#     return model

class TestQLinearConv(unittest.TestCase):
    def setUp(self):
        # Create a specific ONNX model with a single QLinearConv node
        self.model_path = "qlinearadd_model.onnx"
        self.create_qlinearadd_model(self.model_path)

    def create_qlinearadd_model(self, output_path):
        h=128
        w=128
        channels=8
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
        onnx.save(model, output_path)

    def tearDown(self):
        # Delete the ONNX file after testing
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_qlinearconv_inference(self):
        session_options = ort.SessionOptions()
        session_options.enable_gpnpu = False
        print(f"Flag enable_gpnpu: {session_options.enable_gpnpu}")

        # Create an inference session
        session1 = ort.InferenceSession(self.model_path, sess_options=session_options, providers=["CPUExecutionProvider"])
        print(f"Check flag enable_gpnpu: {session1.get_session_options().enable_gpnpu}")

        session_options.enable_gpnpu = True
        session2 = ort.InferenceSession(self.model_path, sess_options=session_options, providers=["CPUExecutionProvider"])
        print(f"Check flag enable_gpnpu: {session2.get_session_options().enable_gpnpu}")

        # Get information about both inputs
        input_a_info = session1.get_inputs()[0]
        # input_b_info = session.get_inputs()[1]

        print(f"Model input names: {input_a_info.name}")
        print(f"Model input shapes: {input_a_info.shape}")

        # Create random INT8 data matching the input shapes
        shape_tuple_a = tuple(dim if isinstance(dim, int) else 1 for dim in input_a_info.shape)

        # Generate random data for both inputs
        x_data_a = np.random.randint(
            low=-128, high=128, size=shape_tuple_a, dtype=np.int8
        )

        # Create input dictionary with both inputs
        input_dict = {
            input_a_info.name: x_data_a
        }

        # Run inference
        output_name1 = session1.get_outputs()[0].name
        print(f"Process ID: {os.getpid()}")
        output_data1 = session1.run([output_name1], input_dict)[0]
        output_name2 = session2.get_outputs()[0].name
        print(f"Process ID: {os.getpid()}")
        output_data2 = session2.run([output_name2], input_dict)[0]

        # Print shapes and types
        print(f"Input A data shape: {x_data_a.shape}, dtype: {x_data_a.dtype}")
        # print(f"Output data shape: {output_data1.shape}, dtype: {output_data1.dtype}")
        print("Output data (truncated):\n", output_data1.flatten()[:50], "...\n")
        print("Output data (truncated):\n", output_data2.flatten()[:50], "...\n")
        # print("hi")
        difference = output_data1 - output_data2
        max_diff = np.max(np.abs(difference))
        print(max_diff)

        BATCH_SIZE = 1
        CHANNELS = 64
        HEIGHT = 56
        WIDTH = 56
        difference = output_data1 - output_data2

        max_diff = np.max(np.abs(difference))

        # Check the output shape and type
        self.assertEqual(output_data1.shape, (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH))
        self.assertEqual(output_data1.dtype, np.int8)
        self.assertLessEqual(max_diff, 1)

if __name__ == '__main__':
    unittest.main()
